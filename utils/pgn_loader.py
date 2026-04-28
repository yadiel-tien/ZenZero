import re
import numpy as np
from tqdm import tqdm
from env.chess import ChineseChess  # 确保能导入你的环境


def pgn_to_numpy_dataset(pgn_path: str, output_path: str, sample_step: int):
    """
    读取 PGN，模拟走棋sample_step步后，提取 State ，保存为 numpy 文件
    """
    env = ChineseChess()

    # 用于收集数据的列表
    collected_states = []
    collected_last_actions = []
    seen_states = set()
    # 1. 读取 PGN 文件
    print(f"Reading PGN: {pgn_path}...")
    with open(pgn_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 分割每一局 (简单按 [Game... 分割)
    games_raw = content.split('[Game "Chinese Chess"]')
    print(f"Found {len(games_raw) - 1} games. Processing...")

    # 2. 遍历处理每一局
    for game_str in tqdm(games_raw):
        if not game_str.strip(): continue

        # 提取招法 (C3-C4 格式),得到[('C3', 'C4'), ('C9', 'E7'),...]
        moves = re.findall(r"([A-I]\d)-([A-I]\d)", game_str)

        # 过滤掉太短的对局
        if len(moves) < sample_step:
            continue

        # --- 模拟下棋到指定步数 ---
        env.reset()
        valid_game = True

        for i in range(sample_step):
            src_str, dst_str = moves[i]
            try:
                # 坐标转换 PGN(ICCS) -> Matrix(0-9, 0-8)
                # ICCS: A0(左下) -> I9(右上)
                # Env:  (9,0)(左下) -> (0,8)(右上)

                c_src = ord(src_str[0]) - ord('A')
                r_src = 9 - int(src_str[1])
                c_dst = ord(dst_str[0]) - ord('A')
                r_dst = 9 - int(dst_str[1])

                move = (r_src, c_src, r_dst, c_dst)
                env.step(env.move2action(move))

            except Exception:
                valid_game = False
                break

        if not valid_game:
            continue

        # 存入列表 (保存副本)
        state = env.state.copy()
        s_bytes = state.tobytes()
        if s_bytes not in seen_states:
            seen_states.add(s_bytes)
            collected_states.append(state)
            collected_last_actions.append(env.last_action)

    # 3. 转换为 Numpy 数组并保存
    if not collected_states:
        print("No valid openings found.")
        return

    print(f"Collected  {len(collected_states)} unique states.")

    # Stack 起来: states shape -> (N, 7, 10, 9)
    all_states = np.stack(collected_states, dtype=np.int8)  # state全是整数，int8省空间
    all_actions = np.stack(collected_last_actions, dtype=np.int32)
    output_path = f'{output_path}_step{sample_step}.npz'
    print(f"Saving {len(all_states)} openings to {output_path}...")
    # 保存为压缩的 .npz 文件
    np.savez_compressed(output_path, states=all_states, last_actions=all_actions, steps=sample_step)
    print("Done!")


if __name__ == "__main__":
    # 使用方法
    pgn_to_numpy_dataset("env/dpxq-99813games.pgns", "env/chess", 50)
