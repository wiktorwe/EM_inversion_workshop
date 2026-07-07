"""Shared Jupyter Server settings for Voila launchers."""

# Voila/Tornado warn when ping_timeout > ping_interval; keep timeout <= interval.
c.ServerApp.websocket_ping_interval = 30000
c.ServerApp.websocket_ping_timeout = 25000
