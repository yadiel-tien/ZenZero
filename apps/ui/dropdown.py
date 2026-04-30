import pygame
from core.utils.config import CONFIG

class Dropdown:
    # 静态音效
    scroll_sound = None
    select_sound = None

    def __init__(self, options, pos, size=(240, 50), default_text="Select Model", default_index=-1):
        self.options = options # 列表: [(label, value), ...]
        self.pos = pos
        self.size = size
        self.rect = pygame.Rect(pos, size)
        self.screen = pygame.display.get_surface()
        
        self.is_open = False
        self.selected_index = default_index
        self.hovered_index = -1
        self.default_text = default_text
        
        # 滚动相关
        self.scroll_offset = 0
        self.max_visible = 8
        self.item_height = self.size[1]
        
        # 样式配置 (取自蓝色主题)
        self.colors = CONFIG['color_themes']['blue']

        # 加载音效
        if Dropdown.scroll_sound is None:
            try:
                Dropdown.scroll_sound = pygame.mixer.Sound('./apps/assets/sound/ui_scroll.wav')
                Dropdown.scroll_sound.set_volume(0.3)
                Dropdown.select_sound = pygame.mixer.Sound('./apps/assets/sound/ui_click.wav')
                Dropdown.select_sound.set_volume(0.5)
            except:
                pass
        
    def draw(self):
        # 1. 绘制主框
        bg_color = (255, 250, 240) # 象牙白
        pygame.draw.rect(self.screen, bg_color, self.rect, border_radius=4)
        pygame.draw.rect(self.screen, (40, 40, 40), self.rect, border_radius=4, width=1)
        
        # 2. 绘制当前选中的文字 (墨色文字)
        current_text = self.default_text
        if self.selected_index != -1:
            current_text = self.options[self.selected_index][0]
        
        self._draw_text_with_scale(current_text, self.rect, color=(40, 40, 40))
        
        # 3. 绘制右侧小箭头 (墨色)
        arrow_color = (40, 40, 40)
        arrow_center = (self.rect.right - 25, self.rect.centery)
        if self.is_open:
            pygame.draw.polygon(self.screen, arrow_color, [(arrow_center[0]-8, arrow_center[1]+4), (arrow_center[0]+8, arrow_center[1]+4), (arrow_center[0], arrow_center[1]-4)])
        else:
            pygame.draw.polygon(self.screen, arrow_color, [(arrow_center[0]-8, arrow_center[1]-4), (arrow_center[0]+8, arrow_center[1]-4), (arrow_center[0], arrow_center[1]+4)])

        # 4. 绘制展开列表
        if self.is_open:
            num_show = min(len(self.options), self.max_visible)
            list_rect = pygame.Rect(self.rect.x, self.rect.bottom + 5, self.size[0], num_show * self.item_height)
            
            # 绘制背景 (象牙白)
            pygame.draw.rect(self.screen, (255, 250, 240), list_rect, border_radius=4)
            pygame.draw.rect(self.screen, (40, 40, 40), list_rect, border_radius=4, width=1)
            
            # 裁剪区域 (重要：防止滚动时文字出界)
            old_clip = self.screen.get_clip()
            self.screen.set_clip(list_rect)
            
            mouse_pos = pygame.mouse.get_pos()
            for i in range(len(self.options)):
                y_pos = self.rect.bottom + 5 + i * self.item_height - self.scroll_offset
                opt_rect = pygame.Rect(self.rect.x, y_pos, self.size[0], self.item_height)
                
                # 只绘制在可见区域内的
                if opt_rect.bottom < list_rect.top or opt_rect.top > list_rect.bottom:
                    continue

                if opt_rect.collidepoint(mouse_pos):
                    # 悬停使用浅灰黄色
                    pygame.draw.rect(self.screen, (220, 220, 200), opt_rect.inflate(-4, -4), border_radius=4)
                
                self._draw_text_with_scale(self.options[i][0], opt_rect, align_left=True, color=(40, 40, 40))
            
            self.screen.set_clip(old_clip)

            # 绘制滚动条指示器
            if len(self.options) > self.max_visible:
                bar_h = list_rect.height * (self.max_visible / len(self.options))
                bar_y = list_rect.top + (self.scroll_offset / (len(self.options) * self.item_height)) * list_rect.height
                pygame.draw.rect(self.screen, (160, 160, 160), (list_rect.right - 6, bar_y, 4, bar_h), border_radius=2)

    def _draw_text_with_scale(self, text, rect, align_left=False, color=(255, 255, 255)):
        font_size = int(self.size[1] * 0.45)
        max_w = rect.width - (40 if not align_left else 20)
        
        # 安全获取字体逻辑
        font_name = None
        for fn in ['stkaiti', 'kaiti', 'simsun', 'arial']:
            font_path = pygame.font.match_font(fn)
            if font_path:
                font_name = font_path
                break
        
        while font_size > 10:
            font = pygame.font.Font(font_name, font_size)
            surf = font.render(text, True, color)
            if surf.get_width() <= max_w:
                break
            font_size -= 2
            
        text_pos = (rect.x + 15, rect.y + (rect.height - surf.get_height()) // 2)
        self.screen.blit(surf, text_pos)

    def handle_input(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.is_open:
                num_show = min(len(self.options), self.max_visible)
                list_rect = pygame.Rect(self.rect.x, self.rect.bottom + 5, self.size[0], num_show * self.item_height)
                if list_rect.collidepoint(event.pos):
                    # 计算当前悬停的是哪个索引
                    new_hover = int((event.pos[1] - (self.rect.bottom + 5) + self.scroll_offset) // self.item_height)
                    if 0 <= new_hover < len(self.options) and new_hover != self.hovered_index:
                        self.hovered_index = new_hover
                        if Dropdown.scroll_sound:
                            Dropdown.scroll_sound.play()
                else:
                    self.hovered_index = -1
            return False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: # 左键点击
                if self.rect.collidepoint(event.pos):
                    if Dropdown.select_sound:
                        Dropdown.select_sound.play()
                    self.is_open = not self.is_open
                    return True
                
                if self.is_open:
                    num_show = min(len(self.options), self.max_visible)
                    list_rect = pygame.Rect(self.rect.x, self.rect.bottom + 5, self.size[0], num_show * self.item_height)
                    if list_rect.collidepoint(event.pos):
                        # 计算点击的是哪个索引
                        clicked_idx = int((event.pos[1] - (self.rect.bottom + 5) + self.scroll_offset) // self.item_height)
                        if 0 <= clicked_idx < len(self.options):
                            if Dropdown.select_sound:
                                Dropdown.select_sound.play()
                            self.selected_index = clicked_idx
                            self.is_open = False
                            return True
                    self.is_open = False
            
            elif self.is_open: # 滚轮事件
                max_scroll = max(0, len(self.options) * self.item_height - (min(len(self.options), self.max_visible) * self.item_height))
                if event.button == 4: # 向上滚
                    if self.scroll_offset > 0:
                        self.scroll_offset = max(0, self.scroll_offset - self.item_height)
                        if Dropdown.scroll_sound:
                            Dropdown.scroll_sound.play()
                        return True
                elif event.button == 5: # 向下滚
                    if self.scroll_offset < max_scroll:
                        self.scroll_offset = min(max_scroll, self.scroll_offset + self.item_height)
                        if Dropdown.scroll_sound:
                            Dropdown.scroll_sound.play()
                        return True
                    
        return False

    @property
    def selected_value(self):
        if self.selected_index == -1: return None
        return self.options[self.selected_index][1]
