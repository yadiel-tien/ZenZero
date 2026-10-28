"""训练控制台：进程管理（runner）+ 制品管理（artifacts）+ Flask 路由（routes）。

挂在 5000 端口的棋类服务上，页面在 /admin，API 在 /api/admin/*。
"""
from .routes import bp

__all__ = ['bp']
