"""RuleContext：对局"过程性状态"的单一事实源（表示版本 2 引入）。

存放不进 state 数组、但规则判定需要的信息——局面重复：
- pos: 位置键 -> [出现次数, 轮红走时被将军的次数, 轮黑走时被将军的次数, 倒数第二次出现的手数, 最近一次出现的手数]

术语：**手（ply）**= 任一方走一步（红走一子算 1 手，黑回一手算第 2 手）；一手 == 一回合是错的，一回合 = 两手。
代码参数名保留 ply（英文棋类编程标准词），中文行文一律说"手"。

键为 position_key(state, side_to_move)（盘面字节 + 轮走方），
即"同一盘面 + 同一轮走方"记为同一位置。

计数约定（全项目统一，勿改）：
- pos[key][0] 是该位置的**历史出现次数**（真实对局中已经走到过的局面都算，
  包含当前局面本身——由 env.step/开局登记；搜索树里的假设局面不计入，
  判定时按"历史次数 + 1"把当前这一步补上）。
- 被将军的次数按"轮走方"归类记录：轮到谁走、谁被将，就记在谁名下。
  由于同一位置的所有出现都由另一方走出，"每次出现轮走方都被将"
  等价于"对方在反复将军"。
- 长将判负的三个条件（对齐中协棋例总纲"任何情况下均不允许单方面长将"与
  维基"长将＝连续不停照将而形成循环达三次"的定义）：
  ①全部出现均带将；②走子方正处于**连续不停照将**中（run 计数≥3，
  一将一闲交替属于允许着法，不罚）；③**着法循环不变**：最后两次出现等间隔 d，
  且最近 d 手的着法序列与再之前 d 手逐一相同（move_log 周期校验）。
  仅等间隔不算循环——同一局面可以由不同的中间着法走到（绕左圈/绕右圈都会回到原点），
  官方的措辞是"双方着法循环**不变**"，罚的是复读机，不是恰好重逢。
  任一不满足只按普通三次重复判和。
- run[side]：各方当前的连续将军手数（自己将军 +1，走不将军的着清零）。
- 隐含窗口（材料单调性）：吃子使双方合计棋子数严格 -1，而同 key 要求盘面
  字节全等（兵力相同）——两次出现之间不可能隔着吃子。因此同一位置的出现
  全部落在同一个无吃子段内，被 120 步时钟天然封顶，不存在"隔全局散见累计"。
"""


class RuleContext:
    __slots__ = ('pos', 'run', 'move_log')

    def __init__(self) -> None:
        self.pos: dict[bytes, list[int]] = {}
        self.run = [0, 0]
        # 本局着法流水（register 逐手追加，开局/读库时清空）：
        # "循环不变着"的最终裁判——两个区间着法序列逐点相等才算复读循环
        self.move_log: list[int] = []

    def check_run(self, side: int) -> int:
        """side 方当前的连续将军手数（不含尚未登记的本手）"""
        return self.run[side]

    def count(self, key: bytes) -> int:
        """该位置（含轮走方）已经出现过的次数"""
        e = self.pos.get(key)
        return e[0] if e else 0

    def checked_count(self, key: bytes, side_to_move: int) -> int:
        """该位置的历史出现中，轮到 side_to_move 走且正被将军的次数"""
        e = self.pos.get(key)
        return e[1 + side_to_move] if e else 0

    def loop_ok(self, key: bytes, arrival_ply: int, cur_action: int | None) -> bool:
        """"着法循环不变"核查（替代旧的裸等间隔判断）：
        ① 两次间隔相等（d = arrival − 上次 == 上次 − 上上次）；
        ② 最近 d 手（含本次 cur_action）与再之前 d 手的着法序列逐一相同。
        数据不足（历史 <2 次或着法流水 <2d 手）时宽松放行（只可能少罚，不会误罚）。"""
        e = self.pos.get(key)
        if e is None or e[0] < 2:
            return True
        d = arrival_ply - e[4]
        if d <= 0 or d != e[4] - e[3]:
            return False
        seq = self.move_log + ([cur_action] if cur_action is not None and cur_action >= 0 else [])
        if len(seq) < 2 * d:
            return True
        return seq[-d:] == seq[-2 * d:-d]

    def register(self, key: bytes, checked: bool, side_to_move: int, ply: int,
                 action: int | None = None) -> None:
        """记录一次局面出现（在 game_over 判定之后调用）；ply = 当前手数（env.steps 口径），
        action = 走到这里的着法（真实对局必传，手工构造的测试账本可为 None）"""
        if action is not None and action >= 0:
            self.move_log.append(action)
        mover = 1 - side_to_move
        self.run[mover] = self.run[mover] + 1 if checked else 0
        e = self.pos.get(key)
        chk_r = int(checked) if side_to_move == 0 else 0
        chk_b = int(checked) if side_to_move == 1 else 0
        if e is None:
            self.pos[key] = [1, chk_r, chk_b, ply, ply]
        else:
            e[0] += 1
            e[1] += chk_r
            e[2] += chk_b
            e[3], e[4] = e[4], ply

    def reset(self) -> None:
        self.pos.clear()
        self.run = [0, 0]
        self.move_log.clear()
