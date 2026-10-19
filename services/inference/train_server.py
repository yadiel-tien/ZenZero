import os
import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from core.network.network import Net
from core.utils.replay import ReplayBuffer
from core.utils.types import EnvName
from .engine import InferenceEngine
from .functions import get_checkpoint_path
from core.utils.config import CONFIG, settings
from core.utils.logger import get_logger


class TrainServer(InferenceEngine):
    """训练服务器：进程内推理引擎 + 模型训练。

    selfplay 与本类运行在同一进程中，MCTS 通过 request_queue 直接请求推理，
    训练命令 (fit/更新模型/删除模型/重置统计) 为直接方法调用，无需 socket。
    """

    def __init__(self, model_id: int, env_name: EnvName):
        super().__init__(model_id, env_name)
        self.fit_logger = get_logger('fit')
        # 模型
        self.fit_model = Net(self.eval_model.config, eval_model=False)
        # 优化器和学习率调解器
        self.optimizer = torch.optim.Adam(self.fit_model.parameters(),
                                          lr=1e-3,
                                          weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                    T_max=settings['max_iters'], eta_min=1e-5)
        # 训练步数计数器
        self.total_steps_trained = 0
        # buffer
        self.buffer = ReplayBuffer(500_000, 2048)
        # fit model与eval model一致
        self.load_checkpoint(self.model_index)
        # 可视化记录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=f'runs/{self.env_name}_{timestamp}')

    def load_checkpoint(self, iteration: int) -> None:
        """读取并加载存档"""
        path = get_checkpoint_path(self.env_name, iteration)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            self.fit_model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'total_steps_trained' in checkpoint:
                self.total_steps_trained = checkpoint['total_steps_trained']
            self.fit_logger.info(
                f'Load checkpoint successfully. Iteration: {iteration}, Step: {self.total_steps_trained}.')
        else:
            self.fit_logger.info(f'Checkpoint {iteration} not found.Starting from raw model.')

    def save_checkpoint(self, iteration: int) -> None:
        """保存存档 (原子化操作)"""
        # 创建checkpoint
        checkpoint = {
            'total_steps_trained': self.total_steps_trained,
            'model': self.fit_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.fit_model.config
        }

        # 先保存到临时文件，再重命名 (防止保存过程中断导致文件损坏)
        path = get_checkpoint_path(self.env_name, iteration)
        tmp_path = path + ".tmp"
        try:
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, path)
            self.fit_logger.info(f"Saved checkpoint to {path}")
        except Exception as e:
            self.fit_logger.error(f"Failed to save checkpoint: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def remove_failed_model(self, iteration_to_remove: int) -> None:
        """移除未通过测试的模型"""
        path = get_checkpoint_path(self.env_name, iteration_to_remove)
        if os.path.exists(path):
            os.remove(path)
        self.fit_logger.info(
            f"Model{iteration_to_remove} does not pass evaluation, checkpoint was deleted.")

    def update_eval_model(self, iteration: int) -> None:
        """模型通过测试，将eval model参数同步到fit model"""
        with self.model_lock:
            self.eval_model.load_state_dict(self.fit_model.state_dict())
            self.model_index = iteration
        self.fit_logger.info(f"Evaluation model updated to {iteration}.")

    def fit(self, n_collected_samples: int, iteration: int) -> str:
        """从buffer中获取数据，训练神经网络"""
        start = time.time()
        # 加载最新buffer
        self.buffer.load()
        # 根据获取的数据和相关设置计算训练epoch
        n_training_steps = n_collected_samples * settings['augment_times'] * CONFIG[
            'training_steps_per_sample'] // self.buffer.batch_size
        for step in range(n_training_steps):
            # 批量数据获取
            states, pis, zs = self.buffer.get_batch()
            states = torch.from_numpy(states).float().to(CONFIG['device'])
            pis = torch.from_numpy(pis).float().to(CONFIG['device'])
            zs = torch.from_numpy(zs).float().to(CONFIG['device'])
            print("\033[K", end='')  # 清空之前的尾行推理信息
            self.fit_logger.info(
                f"real z mean: {torch.mean(zs).item():>7.4f},"
                f" std:{torch.std(zs).item():>7.4f},"
                f" q[:3]: {','.join(f'{i.item():>8.4f}' for i in zs[:3])}"
            )
            # 模型前向推理
            policy_logits, values = self.fit_model(states)
            self.fit_logger.info(
                f"pred v mean: {torch.mean(values).item():>7.4f},"
                f" std:{torch.std(values).item():>7.4f},"
                f" v[:3]: {','.join(f'{i.item():>8.4f}' for i in values[:3])}"
            )

            # 交叉熵损失，使policy的结果趋近mcts模拟出来的pi，[B,H*W]->scalar
            policy_loss = - torch.sum(pis * torch.log_softmax(policy_logits, dim=1), dim=1).mean()
            # 均方差损失，使value的结果趋近与mcts模拟出来的z，[B]。
            value_loss = torch.nn.functional.mse_loss(values, zs)

            # 用总的损失进行反向梯度更新
            loss = policy_loss + value_loss
            self.optimizer.zero_grad()  # 清空旧梯度
            loss.backward()  # 反向传播

            torch.nn.utils.clip_grad_norm_(self.fit_model.parameters(), max_norm=1.0)  # 将梯度范数裁剪到1.0
            self.optimizer.step()  # 更新参数

            # 记录数据到tensorboard
            self.writer.add_scalar('Loss/total', loss.item(), self.total_steps_trained)
            self.writer.add_scalar('Loss/policy', policy_loss.item(), self.total_steps_trained)
            self.writer.add_scalar('Loss/value', value_loss.item(), self.total_steps_trained)

            # 更新计步器
            self.total_steps_trained += 1

            self.fit_logger.info(
                f"Step {step + 1}:\n "
                f" loss={loss.item():.4f}, policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}"
            )

        # 更新学习率调解器
        self.writer.add_scalar('Learning Rate', self.scheduler.get_last_lr()[0], self.total_steps_trained)
        self.scheduler.step()
        # 保存存档
        self.save_checkpoint(iteration)
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        duration = time.time() - start
        self.fit_logger.info(f"iteration_to_remove{iteration}:{n_training_steps}轮训练完成，共用时{duration:.2f}秒。")

        return 'Fit done!'

    def shutdown(self) -> None:
        if self.writer:
            self.writer.close()
        super().shutdown()
