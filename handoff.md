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

---

# 🔐 Security Audit Report & Secrets Remediation Log (Hermes Pass)

## 📋 Executive Summary
- **Auditor:** Hermes (Zero-Context Security Auditor)
- **Scope:** Full repository sweep for hardcoded secrets, API keys, and credentials.
- **Status:** **COMPLETED**
- **Findings Summary:** 14 hardcoded credentials / default passcodes identified across configuration, script, UI, and test suites.

---

## 📊 Discovered Secrets & Hardcoded Credentials

| # | File Path | Line Number | Current Hardcoded Secret | Recommended `.env` Variable / Placeholder | Severity |
|---|-----------|-------------|--------------------------|------------------------------------------|----------|
| 1 | `config/config.json` | L3 | `"BACKEND_API_KEY": "spm-secret-key"` | `os.getenv("SPM_SECRET_KEY", "")` | 🔴 Critical |
| 2 | `config/manager.py` | L23 | `"BACKEND_API_KEY": "sk-dummy-key"` | `os.getenv("SPM_SECRET_KEY", "sk-dummy-key")` | 🟠 High |
| 3 | `config/docker-compose.yml` | L11 | `POSTGRES_PASSWORD: spm_secure_password` | `${POSTGRES_PASSWORD:-spm_secure_password}` | 🔴 Critical |
| 4 | `.env.example` | L25 | `POSTGRES_PASSWORD=spm_secure_password` | `POSTGRES_PASSWORD=<CHANGE_THIS_SECRET>` | 🟢 Info |
| 5 | `scripts/sleep_cycle.py` | L147 | `os.getenv("POSTGRES_PASSWORD", "spm_secure_password")` | `os.getenv("POSTGRES_PASSWORD")` | 🟡 Medium |
| 6 | `proxy/ui/index.html` | L122 | `value="sk-dummy-key"` | `value=""` (Placeholder) | 🟡 Medium |
| 7 | `tests/conftest.py` | L20 | `password="spm_secure_password"` | `os.getenv("POSTGRES_PASSWORD", "spm_secure_password")` | 🟢 Low (Test) |
| 8 | `tests/test_sensory_filter.py` | L14 | `password="spm_secure_password"` | `os.getenv("POSTGRES_PASSWORD", "spm_secure_password")` | 🟢 Low (Test) |
| 9 | `tests/test_sleep_cycle.py` | L20 | `password="spm_secure_password"` | `os.getenv("POSTGRES_PASSWORD", "spm_secure_password")` | 🟢 Low (Test) |
| 10 | `tests/test_fr001_session_isolation.py` | L29 | `password="spm_secure_password"` | `os.getenv("POSTGRES_PASSWORD", "spm_secure_password")` | 🟢 Low (Test) |
| 11 | `tests/test_retriever.py` | L13 | `password="spm_secure_password"` | `os.getenv("POSTGRES_PASSWORD", "spm_secure_password")` | 🟢 Low (Test) |
| 12 | `tests/test_e2e_integration.py` | L39 | `password="spm_secure_password"` | `os.getenv("POSTGRES_PASSWORD", "spm_secure_password")` | 🟢 Low (Test) |
| 13 | `tests/test_fr002_bulk_import.py` | L37 | `password="spm_secure_password"` | `os.getenv("POSTGRES_PASSWORD", "spm_secure_password")` | 🟢 Low (Test) |
| 14 | `tests/test_fr003_tiered_lifecycle.py` | L31 | `password="spm_secure_password"` | `os.getenv("POSTGRES_PASSWORD", "spm_secure_password")` | 🟢 Low (Test) |

---

## 🛠 Recommended Git History Scrub (Post-Hermes Clean)

Once values are moved to `.env` (which is already listed in `.gitignore`), execute the following commands to permanently remove old hardcoded strings from git history:

```bash
# 1. Install git-filter-repo
pip install git-filter-repo

# 2. Scrub literal secrets across all commits and branches
git filter-repo --replace-text <(echo "spm_secure_password==>REDACTED_PASSWORD"; echo "spm-secret-key==>REDACTED_KEY")

# 3. Force push scrubbed history to remote
git push origin --force --all
git push origin --force --tags
```
