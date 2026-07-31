# FR-004 Handoff: Built-in Web Admin Dashboard & Configuration API

## Objectives
Implement GitHub Issue #4: Web Admin Dashboard (`http://localhost:5050/admin`), dynamic SettingsManager (`config/manager.py`), Admin REST API (`proxy/api/admin_routes.py`), rotating logger (`logs/spm_proxy.log`), and CLI live monitor (`scripts/monitor_spm.py`).

### Requirements
1. **Settings Manager (`config/manager.py`)**:
   - `SettingsManager` reading/writing `config/config.json`.
   - Keys: `BACKEND_LLM_URL`, `BACKEND_API_KEY`, `SPM_PROXY_PORT`.
2. **Admin REST API (`proxy/api/admin_routes.py`)**:
   - `GET /admin/api/v1/stats`: Returns JSON stats (requests, latencies, active sessions, gating breakdown, DB size).
   - `GET /admin/api/v1/config` & `POST /admin/api/v1/config`: Read/write settings.
   - `DELETE /admin/api/v1/sessions/{session_id}`: Delete specific session data.
   - `DELETE /admin/api/v1/factory_reset`: Truncate all tables and reset state.
3. **Web Admin Dashboard (`proxy/ui/index.html`)**:
   - Single-page Vanilla HTML/CSS UI with Dashboard, Config, and Data Management tabs.
   - Serve under `http://localhost:5050/admin` in `proxy/main.py`.
4. **Rotating Logger (`proxy/core/logger.py`)**:
   - Save structured logs to `logs/spm_proxy.log` (10 MB rotation).
5. **CLI Live Monitor (`scripts/monitor_spm.py`)**:
   - Stream live metrics and logs to terminal.
6. **Unit Tests (`tests/test_fr004_observability.py`)**:
   - Test config manager, admin API, wipeout safety, and UI serving.
   - Run `pytest tests/ -v` (100% passing).

### Commands
```bash
source /home/osmon/Desktop/Experiments/SillyTavern/spm-demo-mvp/venv/bin/activate
python -m pytest tests/ -v
```

### Git Target
- Branch: `main`
