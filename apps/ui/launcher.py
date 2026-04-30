import pygame
import random
import math
from core.utils.config import CONFIG, settings
from apps.ui.button import Button
from core.player.ai_client import AIClient
from apps.ui.dropdown import Dropdown

class LauncherUI:
    def __init__(self):
        self.screen = pygame.display.get_surface()
        self.rect = self.screen.get_rect()
        self.state = 'SELECT_GAME'
        self.selected_game = None
        self.selected_mode = None
        self.model_indices = []
        self.best_index = -1
        self.selected_model1 = -1
        self.selected_model2 = -1
        
        # 缓存字体
        self.font_path = self._get_best_font()
        
        self.buttons = []
        self.dropdowns = []
        self.setup_game_buttons()
        
        # 启动背景音乐
        self.start_music()
        
        # 缓存背景渐变，避免每帧重复计算
        self.bg_surface = pygame.Surface(self.rect.size)
        self.render_background_cache()
        
        # 装饰性粒子 (古风墨点/落花) - 改进运动逻辑
        self.particles = []
        for _ in range(35):
            self.particles.append({
                "pos": [random.randint(0, self.rect.width), random.randint(0, self.rect.height)],
                "speed": random.uniform(0.7, 1.4),  # 稍微加快，减少由于取整导致的停顿感
                "size": random.randint(2, 6),
                "phase": random.uniform(0, math.pi * 2),
                "sway_amp": random.uniform(15, 40),
                "sway_speed": random.uniform(0.02, 0.04),
                "color": random.choice([
                    (40, 40, 40),   # 墨黑
                    (139, 0, 0),    # 深红
                    (80, 80, 80),   # 浅灰
                    (110, 100, 90)  # 灰褐
                ])
            })

    def _get_best_font(self):
        for fn in ['stkaiti', 'kaiti', 'simsun', 'arial']:
            font_path = pygame.font.match_font(fn)
            if font_path:
                return font_path
        return None

    def render_background_cache(self):
        paper_center = (250, 245, 225) # 亮
        paper_edge = (230, 220, 200)   # 暗
        self.bg_surface.fill(paper_edge)
        # 绘制多层渐变圆
        for r in range(self.rect.width, 0, -80):
            ratio = r / self.rect.width
            color = [int(paper_edge[c] * ratio + paper_center[c] * (1 - ratio)) for c in range(3)]
            pygame.draw.circle(self.bg_surface, color, self.rect.center, r)

    def setup_game_buttons(self):
        self.state = 'SELECT_GAME'
        self.selected_game = None
        self.model_indices = []
        self.dropdowns = []
        center_x = self.rect.centerx - 120
        self.buttons = [
            Button("Chinese Chess", lambda: self.select_game("ChineseChess"), (center_x, 300), color='red'),
            Button("Gomoku", lambda: self.select_game("Gomoku"), (center_x, 400), color='blue')
        ]

    def setup_mode_buttons(self):
        self.state = 'SELECT_MODE'
        self.dropdowns = []
        center_x = self.rect.centerx - 120
        self.buttons = [
            Button("Local PvP", lambda: self.select_mode("HvH"), (center_x, 280), color='green'),
            Button("Human vs AI", lambda: self.select_mode("HvAI"), (center_x, 360), color='orange'),
            Button("AI vs AI", lambda: self.select_mode("AIvAI"), (center_x, 440), color='grey'),
            Button("Back", self.setup_game_buttons, (30, 30), (100, 40), color='black')
        ]

    def setup_model_buttons(self):
        self.state = 'SELECT_MODEL'
        if not self.model_indices:
            data = AIClient.get_available_models(self.selected_game)
            self.model_indices = data.get('indices', [])
            self.best_index = data.get('best_index', -1)
            if not self.model_indices: self.model_indices = [-1]
        
        # 准备下拉选项 (英文标签)
        options = []
        best_opt_idx = -1
        for idx in sorted(self.model_indices, reverse=True):
            if idx == self.best_index:
                label = f"Strongest AI (v{idx})"
                best_opt_idx = len(options)
            elif idx == -1:
                label = "Raw AI"
            else:
                label = f"AI Version {idx}"
            options.append((label, idx))

        center_x = self.rect.centerx - 120
        self.buttons = [
            Button("Back", self.setup_mode_buttons, (30, 30), (100, 40), color='black'),
            Button("Start Game", self.confirm_selection, (center_x, 600), color='red')
        ]
        
        if self.selected_mode == "HvAI":
            self.dropdowns = [Dropdown(options, (center_x, 300), default_text="Select AI", default_index=best_opt_idx)]
        else: # AI vs AI
            self.dropdowns = [
                Dropdown(options, (center_x, 250), default_text="AI 1 (First)", default_index=best_opt_idx),
                Dropdown(options, (center_x, 350), default_text="AI 2 (Second)", default_index=best_opt_idx)
            ]

    def confirm_selection(self):
        if self.selected_mode == "HvAI":
            val = self.dropdowns[0].selected_value
            if val is not None:
                self.finish(val, -1)
        else:
            v1 = self.dropdowns[0].selected_value
            v2 = self.dropdowns[1].selected_value
            if v1 is not None and v2 is not None:
                self.finish(v1, v2)

    def start_music(self):
        try:
            # 确保音频设备已初始化
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load('./apps/assets/sound/launcher_bgm.wav')
            pygame.mixer.music.set_volume(0.8)
            pygame.mixer.music.play(-1) # 循环播放
        except:
            pass

    def stop_music(self):
        pygame.mixer.music.stop()

    def select_game(self, game):
        self.selected_game = game
        CONFIG['game_name'] = game
        self.setup_mode_buttons()

    def select_mode(self, mode):
        self.selected_mode = mode
        if mode == "HvH":
            self.finish(-1, -1)
        else:
            self.setup_model_buttons()

    def finish(self, m1, m2):
        self.selected_model1 = m1
        self.selected_model2 = m2
        self.state = 'FINISHED'

    def handle_input(self, event):
        # 优先处理下拉菜单
        for dd in self.dropdowns:
            if dd.handle_input(event):
                return

        for btn in self.buttons:
            btn.handle_input(event)

    def draw(self):
        # 1. 绘制背景
        self.screen.blit(self.bg_surface, (0, 0))
        
        # 2. 绘制动态粒子 (墨点/落花)
        for p in self.particles:
            # 更新状态
            p["phase"] += p["sway_speed"]
            p["pos"][1] += p["speed"]
            
            # 计算显示坐标
            display_x = p["pos"][0] + math.sin(p["phase"]) * p["sway_amp"]
            display_y = p["pos"][1]
            
            # 边界循环
            if display_y > self.rect.height + 20:
                p["pos"][1] = -20
                p["pos"][0] = random.randint(0, self.rect.width)
                p["phase"] = random.uniform(0, math.pi * 2)

            pygame.draw.circle(self.screen, p["color"], (int(display_x), int(display_y)), p["size"])

        # 3. 标题 (书法感 + 阴影) - 位置下移
        font_big = pygame.font.Font(self.font_path, 72)
        title_y = 115
        # 绘制浅色阴影
        shadow_surf = font_big.render("ZenZero", True, (200, 190, 170))
        self.screen.blit(shadow_surf, (self.rect.centerx - shadow_surf.get_width()//2 + 3, title_y + 3))
        # 绘制主标题
        title_surf = font_big.render("ZenZero", True, (40, 40, 40))
        self.screen.blit(title_surf, (self.rect.centerx - title_surf.get_width()//2, title_y))
        
        # 装饰性红印章 (位置同步下移)
        seal_x, seal_y = self.rect.centerx + 165, title_y - 5
        # 印章投影
        pygame.draw.rect(self.screen, (160, 150, 130), (seal_x + 2, seal_y + 2, 42, 42), border_radius=6)
        # 印章主体
        pygame.draw.rect(self.screen, (150, 20, 20), (seal_x, seal_y, 42, 42), border_radius=6)
        # 内边框
        pygame.draw.rect(self.screen, (200, 160, 40), (seal_x + 4, seal_y + 4, 34, 34), width=1, border_radius=4)
        
        font_seal = pygame.font.Font(self.font_path, 22)
        seal_txt = font_seal.render("AI", True, (250, 240, 210))
        self.screen.blit(seal_txt, (seal_x + 10, seal_y + 8))

        # 副标题: 优化显示逻辑与长度控制 - 位置同步下移
        status_text = self.state.replace('_', ' ')
        if self.selected_game:
            game_name = "Chess" if self.selected_game == "ChineseChess" else self.selected_game
            status_text = f"{game_name} · {status_text}"
        
        font_small = pygame.font.Font(self.font_path, 24)
        sub_surf = font_small.render(status_text, True, (110, 100, 90))
        
        # 宽度自适应: 如果副标题过长，则缩小字体
        if sub_surf.get_width() > title_surf.get_width() * 0.95:
            font_small = pygame.font.Font(self.font_path, 18)
            sub_surf = font_small.render(status_text, True, (110, 100, 90))

        self.screen.blit(sub_surf, (self.rect.centerx - sub_surf.get_width()//2, title_y + 85))

        # 4. 绘制交互组件
        for btn in self.buttons:
            btn.draw()
            
        for dd in reversed(self.dropdowns):
            dd.draw()
