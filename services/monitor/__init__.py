"""训练监控：只读采集 + HTTP 面板（见 services/monitor/app.py 的 serve()）。"""
from .app import make_server, serve

__all__ = ['make_server', 'serve']
