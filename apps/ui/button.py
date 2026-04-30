import pygame
from core.utils.config import CONFIG, color_key

class Button:
    # 静态类变量，避免每个按钮重复加载
    hover_sound = None
    click_sound = None

    def __init__(self, text, action, pos, size=(240, 60), color: color_key = 'blue'):
        self.text = text
        self.action = action
        self.size = size
        self.rect = pygame.Rect(pos, size)
        self.base_colors = CONFIG['color_themes'][color]
        self.state = 'normal'
        self.screen = pygame.display.get_surface()
        
        # 字体预加载
        self.font = pygame.font.SysFont('Arial', int(self.size[1] * 0.45), bold=True)

        # 加载音效
        if Button.hover_sound is None:
            try:
                Button.hover_sound = pygame.mixer.Sound('./apps/assets/sound/ui_hover.wav')
                Button.hover_sound.set_volume(0.3)
                Button.click_sound = pygame.mixer.Sound('./apps/assets/sound/ui_click.wav')
                Button.click_sound.set_volume(0.5)
            except:
                pass

    def draw(self):
        # 颜色逻辑
        bg_color = self.base_colors[0]
        border_color = self.base_colors[3]
        text_color = (255, 255, 255)
        
        if self.state == 'hover':
            bg_color = self.base_colors[1]
        elif self.state == 'click':
            bg_color = self.base_colors[2]

        # 创建一个带有透明度的Surface用于绘制背景
        button_surface = pygame.Surface(self.size, pygame.SRCALPHA)
        
        # 绘制主背景 (微圆角，模拟木质或漆器)
        pygame.draw.rect(button_surface, bg_color, (0, 0) + self.size, border_radius=4)
        
        # 绘制金纹边框
        border_w = 3 if self.state == 'hover' else 1
        gold_color = (184, 134, 11) # 琥珀金
        pygame.draw.rect(button_surface, gold_color, (0, 0) + self.size, border_radius=4, width=border_w)
        
        # 如果是悬停，增加四个角的装饰
        if self.state == 'hover':
            d = 10
            # 左上角
            pygame.draw.line(button_surface, (255, 215, 0), (0, 0), (d, 0), 3)
            pygame.draw.line(button_surface, (255, 215, 0), (0, 0), (0, d), 3)
            # 右下角
            pygame.draw.line(button_surface, (255, 215, 0), (self.size[0], self.size[1]), (self.size[0]-d, self.size[1]), 3)
            pygame.draw.line(button_surface, (255, 215, 0), (self.size[0], self.size[1]), (self.size[0], self.size[1]-d), 3)

        # 渲染文字 - 动态缩放字体以适应宽度
        font_size = int(self.size[1] * 0.45)

        # 安全获取字体逻辑
        font_name = None
        for fn in ['stkaiti', 'kaiti', 'simsun', 'arial']:
            font_path = pygame.font.match_font(fn)
            if font_path:
                font_name = font_path
                break

        temp_font = pygame.font.Font(font_name, font_size)
        text_surf = temp_font.render(self.text, True, text_color)
        # 如果文字太长，缩小字体
        max_w = self.size[0] - 20
        while text_surf.get_width() > max_w and font_size > 12:
            font_size -= 2
            temp_font = pygame.font.Font(font_name, font_size)
            text_surf = temp_font.render(self.text, True, text_color)


        text_rect = text_surf.get_rect(center=(self.size[0] // 2, self.size[1] // 2))
        button_surface.blit(text_surf, text_rect)

        # 绘制到屏幕
        self.screen.blit(button_surface, self.rect)

    def handle_input(self, event):
        if event.type == pygame.MOUSEMOTION:
            is_hover = self.rect.collidepoint(event.pos)
            if is_hover and self.state != 'hover':
                if Button.hover_sound:
                    Button.hover_sound.play()
                self.state = 'hover'
            elif not is_hover:
                self.state = 'normal'
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.rect.collidepoint(event.pos):
                if Button.click_sound:
                    Button.click_sound.play()
                self.state = 'click'
                self.action()
