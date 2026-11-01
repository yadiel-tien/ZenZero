import json
import os
import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from core.network.network import Net
from core.network.registry import DEFAULT_ARCH, build as build_arch
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

    def __init__(self, model_id: int, env_name: EnvName, buffer: ReplayBuffer | None = None):
        super().__init__(model_id, env_name)
        self.fit_logger = get_logger('fit')
        # 模型：架构跟随 eval_model（envelope 决定），跨档位/跨架构续训由此自动正确
        cfg = self.eval_model.config
        arch = getattr(self.eval_model, 'arch_name', None) or (
            cfg.get('arch') if isinstance(cfg, dict) else None) or DEFAULT_ARCH
        # 架构升级闸门：训练目标架构 = 当前棋种 default_net.arch（preset 只覆写规模键）。
        # 与旧 ckpt 架构不同 → value 头半迁移：同名同形权重（trunk/policy）照用，
        # shape 不符的头重初始化，优化器/调度器从头开始；旧 checkpoint 原样不动。
        target = CONFIG.get(env_name, {}).get('default_net', {}).get('arch', arch) or arch
        self._arch_from = arch if target != arch else None
        if self._arch_from:
            cfg2 = dict(cfg) if isinstance(cfg, dict) else cfg
            cfg2['arch'] = target
            self.fit_model = build_arch(target, cfg2, eval_model=False)
            self.fit_logger.info(f'Arch migration training: {arch} → {target}'
                                 f'（value 头半迁移：不匹配的头重初始化，其余权重继承）')
        else:
            self.fit_model = build_arch(arch, cfg, eval_model=False)
        # 优化器和学习率调解器
        self.optimizer = torch.optim.Adam(self.fit_model.parameters(),
                                          lr=1e-3,
                                          weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                    T_max=settings['max_iters'], eta_min=1e-5)
        # 训练步数计数器
        self.total_steps_trained = 0
        # buffer：由外部注入时与 SelfPlayManager 共享同一实例（同进程无需两份大数组，
        # 象棋旧配置双份合计 ~14GB）；默认容量也跟 config 对齐
        self.buffer = buffer if buffer is not None else ReplayBuffer(settings['buffer_size'], 2048)
        # fit model与eval model一致
        self.load_checkpoint(self.model_index)
        # 可视化记录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=os.path.join(CONFIG['runs_dir'], f'{self.env_name}_{timestamp}'))

    def load_checkpoint(self, iteration: int) -> None:
        """读取并加载存档"""
        path = get_checkpoint_path(self.env_name, iteration, self.profile)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            if self._arch_from:
                from core.network.registry import migrated_load
                used, reinit = migrated_load(self.fit_model, checkpoint['model'])
                self.fit_logger.info(
                    f'Migrated load: reused {used}/{len(self.fit_model.state_dict())} tensors, '
                    f're-init {reinit[:4]}{" …" if len(reinit) > 4 else ""}; '
                    f'optimizer/scheduler fresh start.')
            else:
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
            'config': self.fit_model.config,
            # 棋盘表示版本（feature version）：训练启动闸门据此判断新旧模型是否同表示（v1 存档无此字段）
            'feature_version': int(settings.get('feature_version', 1)),
            # 网络架构名：加载端经 registry 分发（v1 老档无此字段，缺省 resnet_v1）
            'arch': getattr(self.fit_model, 'arch_name', DEFAULT_ARCH)
        }

        # 先保存到临时文件，再重命名 (防止保存过程中断导致文件损坏)
        path = get_checkpoint_path(self.env_name, iteration, self.profile)
        tmp_path = path + ".tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)   # 档位子目录自建（s/l 首存必需要）
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, path)
            self.fit_logger.info(f"Saved checkpoint to {path}")
        except Exception as e:
            self.fit_logger.error(f"Failed to save checkpoint: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def remove_failed_model(self, iteration_to_remove: int) -> None:
        """移除未通过测试的模型"""
        path = get_checkpoint_path(self.env_name, iteration_to_remove, self.profile)
        if os.path.exists(path):
            os.remove(path)
        self.fit_logger.info(
            f"Model{iteration_to_remove} does not pass evaluation, checkpoint was deleted.")

    def update_eval_model(self, iteration: int) -> None:
        """模型通过测试，将eval model参数同步到fit model"""
        # load_state_dict 是写操作：必须持 exclusive 写锁（旧代码 with self.model_lock
        # 会直接 AttributeError——_ModelAccess 只提供 shared()/exclusive() 上下文方法）
        with self.model_lock.exclusive():
            if getattr(self.fit_model, 'arch_name', None) != getattr(self.eval_model, 'arch_name', None):
                # 架构迁移后首次晋升：按训练架构重建评估实例（engine 每请求读属性，独占锁内换引用安全）
                self.fit_logger.info(f'Eval model rebuilt: '
                                     f'{self.eval_model.arch_name} → {self.fit_model.arch_name}')
                self.eval_model = build_arch(self.fit_model.arch_name, self.fit_model.config,
                                             eval_model=True, device=self.eval_model.device)
                self.arch_name = self.fit_model.arch_name
                self._is_wdl = bool(getattr(self.eval_model, 'is_wdl', False))
            else:
                self.eval_model.load_state_dict(self.fit_model.state_dict())
            self.model_index = iteration
        self.fit_logger.info(f"Evaluation model updated to {iteration}.")

    def fit(self, n_collected_samples: int, iteration: int, reload_buffer: bool = True,
            val_buffer: ReplayBuffer | None = None) -> str:
        """从buffer中获取数据，训练神经网络。
        :param reload_buffer: 是否先从磁盘重载 buffer。与 SelfPlayManager 共享实例（现行主路径）
               时应传 False——文件就是 manager 上一秒刚存的，重读几 GB pickle 纯属白干；
               独立使用 TrainServer 时保持 True（旧语义）。
        :param val_buffer: 留出验证集（评估时关闭在线增强），训练后写 TensorBoard Val/*。"""
        start = time.time()
        if reload_buffer:
            # 加载最新buffer
            self.buffer.load()
        # 训练量配平（在线增强语义）：每轮把全仓平均过 training_passes_per_buffer 遍。
        # 旧公式"新增×写入增强份数×30"里那份数是近重复放大器的虚高记账，已废弃
        n_training_steps = max(1, int(self.buffer.size * CONFIG['training_passes_per_buffer']
                                      // self.buffer.batch_size))
        # 设备跟随 fit_model 自身参数，**绝不读全局 CONFIG['device']**：
        # 评估阶段的守擂引擎加载失败会 _degrade_cpu 改写全局（推理端同款事故，见 engine.py 注释），
        # 全局一旦被污染，数据与权重就会分裂在两个设备上崩溃
        fit_device = next(self.fit_model.parameters()).device
        for step in range(n_training_steps):
            # 批量数据获取
            states, pis, zs = self.buffer.get_batch()
            states = torch.from_numpy(states).float().to(fit_device)
            pis = torch.from_numpy(pis).float().to(fit_device)
            zs = torch.from_numpy(zs).float().to(fit_device)
            print("\033[K", end='')  # 清空之前的尾行推理信息
            # zs 是训练目标 v（=0.5z+0.5q̃ 混合后的 value 标签，非纯终局 z——旧日志标签有误导）
            self.fit_logger.info(
                f"target v mean: {torch.mean(zs).item():>7.4f},"
                f" std:{torch.std(zs).item():>7.4f},"
                f" v[:3]: {','.join(f'{i.item():>8.4f}' for i in zs[:3])}"
            )
            # 模型前向推理：WDL 架构取三头 logits，期望分 = p_win - p_loss
            from core.network.archs.resnet_wdl import expected_score, zld_to_class
            is_wdl = getattr(self.fit_model, 'is_wdl', False)
            if is_wdl:
                policy_logits, wdl_logits = self.fit_model.forward_wdl(states)
                values = expected_score(wdl_logits)
            else:
                policy_logits, values = self.fit_model(states)
            self.fit_logger.info(
                f"pred v mean: {torch.mean(values).item():>7.4f},"
                f" std:{torch.std(values).item():>7.4f},"
                f" v[:3]: {','.join(f'{i.item():>8.4f}' for i in values[:3])}"
            )

            # 交叉熵损失，使policy的结果趋近mcts模拟出来的pi，[B,H*W]->scalar
            policy_loss = - torch.sum(pis * torch.log_softmax(policy_logits, dim=1), dim=1).mean()
            # value loss：WDL 向胜/和/负结局分类做交叉熵（z∈{-1,0,1} 即类标）；
            # 老架构对期望分做 MSE 回归
            if is_wdl:
                value_loss = torch.nn.functional.cross_entropy(wdl_logits, zld_to_class(zs))
            else:
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

        # 留出验证集评估：train/val 的 gap 就是过拟合仪表盘（loss→0 时代到此结束）
        self.evaluate_val(val_buffer, fit_device, iteration=iteration)

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

    @torch.no_grad()
    def evaluate_val(self, val_buffer: ReplayBuffer | None, fit_device=None,
                     iteration: int | None = None) -> None:
        """在**留出集全量**上算固定视角（无增强）的 CE / MSE / KL / 目标熵。

        旧实现每轮随机抽 4 批（8,192 个），而留出集有 1.5 万个样本——每次抽到的是不同子集，
        于是 Val 曲线自带 ±0.01 的采样噪声，根本看不出真实变化。
        现在按存储顺序**确定性全覆盖**（batch=留出集 batch_size，约 8 批 ≈ 0.3s），
        并把结果同时写一份 sidecar（logs/fit/val_latest.json）：
        面板不必再从日志尾部猜——一整个 fit 回合 1000+ 行会把 Val 行挤出 tail 窗口，
        这正是"验证集又不显示了"的原因。
        样本不足一批时静默跳过（开局前几轮属正常）。"""
        if val_buffer is None or val_buffer.size < val_buffer.batch_size:
            return
        if fit_device is None:
            fit_device = next(self.fit_model.parameters()).device
        self.fit_model.eval()
        try:
            pol_sum, val_sum, tot_sum, n_batches, kl_sum, ent_sum = 0.0, 0.0, 0.0, 0, 0.0, 0.0
            n_val = int(val_buffer.size)
            bs = int(val_buffer.batch_size)
            for start in range(0, n_val, bs):                # 确定性：按存储顺序全覆盖
                sl = slice(start, min(start + bs, n_val))
                states = torch.from_numpy(val_buffer.state_buffer[sl]).float().to(fit_device)
                pis = torch.from_numpy(val_buffer.policy_buffer[sl]).float().to(fit_device)
                zs = torch.from_numpy(val_buffer.value_buffer[sl]).float().to(fit_device)
                from core.network.archs.resnet_wdl import zld_to_class
                if getattr(self.fit_model, 'is_wdl', False):
                    policy_logits, wdl_logits = self.fit_model.forward_wdl(states)
                else:
                    policy_logits, values = self.fit_model(states)
                log_p = torch.log_softmax(policy_logits, dim=1)
                policy_loss = - torch.sum(pis * log_p, dim=1).mean()
                value_loss = (torch.nn.functional.cross_entropy(wdl_logits, zld_to_class(zs))
                              if getattr(self.fit_model, 'is_wdl', False)
                              else torch.nn.functional.mse_loss(values, zs))
                # 策略交叉熵有硬地板：CE >= H(pi)。目标越软（sims 越少）地板越高，
                # 于是 CE 会"看起来降不动"。真正能学的是 KL(pi_target || pi_model)，
                # 两者分开记，进度才有意义。
                log_pi = torch.log(pis + 1e-8)
                ent = - torch.sum(pis * log_pi, dim=1).mean()
                kl = torch.sum(pis * (log_pi - log_p), dim=1).mean()
                pol_sum += float(policy_loss); val_sum += float(value_loss)
                tot_sum += float(policy_loss + value_loss)
                kl_sum += float(kl); ent_sum += float(ent)
                n_batches += 1
            if n_batches:
                self.writer.add_scalar('Val/policy', pol_sum / n_batches, self.total_steps_trained)
                self.writer.add_scalar('Val/value', val_sum / n_batches, self.total_steps_trained)
                self.writer.add_scalar('Val/total', tot_sum / n_batches, self.total_steps_trained)
                # 目标熵（不可降部分）与 KL（可学部分）分列，避免把地板当瓶颈
                self.writer.add_scalar('Val/policy_entropy', ent_sum / n_batches, self.total_steps_trained)
                self.writer.add_scalar('Val/policy_kl', kl_sum / n_batches, self.total_steps_trained)
                self.fit_logger.info(
                    f'Val loss - policy {pol_sum / n_batches:.4f}, value {val_sum / n_batches:.4f}'
                    f' | KL {kl_sum / n_batches:.4f}, 目标熵 {ent_sum / n_batches:.4f}, 样本 {n_val}')
                self._write_val_sidecar(iteration, n_val,
                                        pol_sum / n_batches, val_sum / n_batches,
                                        kl_sum / n_batches, ent_sum / n_batches)
        finally:
            self.fit_model.train()

    def _write_val_sidecar(self, iteration, n_samples, policy, value, kl, entropy) -> None:
        """留出集指标落一个小 JSON：面板直接读它，不用从几千行日志里刨（原子替换）。"""
        try:
            path = os.path.join(CONFIG['log_dir'], 'fit', 'val_latest.json')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = {'ts': time.time(), 'iso': time.strftime('%Y-%m-%dT%H:%M:%S'),
                       'iteration': iteration, 'step': self.total_steps_trained,
                       'n_samples': int(n_samples),
                       'policy': round(float(policy), 6), 'value': round(float(value), 6),
                       'kl': round(float(kl), 6), 'entropy': round(float(entropy), 6)}
            tmp = path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as fp:
                json.dump(payload, fp, ensure_ascii=False)
            os.replace(tmp, path)
        except OSError:
            pass

    def shutdown(self) -> None:
        if self.writer:
            self.writer.close()
        super().shutdown()
