# 🚦 UrbanPulse — AI-Based Urban Traffic Command & Incident Response System

[![Tests](https://img.shields.io/badge/tests-24%2F24%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66%2B-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Domain:** Smart Cities / Urban Mobility  
**Program:** IBM SkillsBuild Gartner Program 2026 — Agentic AI Capstone  
**Version:** 1.0.0

---

## 🎯 Problem Statement

City traffic authorities need an intelligent operations system capable of monitoring traffic conditions, detecting disruptions, identifying likely causes, and recommending coordinated interventions. Existing manual monitoring systems cannot process the volume of concurrent events that occur during peak hours or emergencies — incident detection takes **8–15 minutes** manually, while UrbanPulse targets **< 2 minutes**.

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/vignesh2396/urban-traffic-command_IBMBob.git
cd urban-traffic-command_IBMBob

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — add your GEMINI_API_KEY (required)
# Optionally add OPENWEATHER_API_KEY

# 4. Initialise database and RAG knowledge base
python -m data.seed_data
python -m rag.embeddings

# 5. Launch the Streamlit app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Docker (alternative)

```bash
cp .env.example .env
# Edit .env with your API keys
docker-compose up --build
```

---

## 🏗️ Architecture

```
User / TMC Operator
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│              SUPERVISOR AGENT (LangGraph)               │
│              StateGraph Controller + HITL Gate          │
└────────┬───────────────────────────────────┬────────────┘
         │                                   │
    [if anomaly]                       [if severity ≥ 3]
         │                                   │
         ▼                                   ▼
┌──────────────────┐           ┌─────────────────────────┐
│ Traffic Monitor  │           │  Incident Investigation  │
│ Agent            │           │  Agent (LLM + RAG + WX)  │
└──────────────────┘           └────────────┬────────────┘
         │                                  │
         ▼                        ┌─────────┴─────────┐
┌──────────────────┐              ▼                   ▼
│ Congestion       │  ┌──────────────────┐ ┌──────────────────┐
│ Detection Agent  │  │ Route Diversion  │ │ Signal           │
└──────────────────┘  │ Agent (LLM)      │ │ Optimization     │
                      └──────────────────┘ │ Agent (LLM)      │
                                           └──────────────────┘
                                                    │
                                                    ▼
                                   ┌──────────────────────────┐
                                   │ Traffic Response          │
                                   │ Coordination Agent (LLM)  │
                                   └────────────┬─────────────┘
                                                │
                                   ┌────────────▼─────────────┐
                                   │     HITL GATE            │
                                   │  severity ≥ 9 → Approve? │
                                   └────────────┬─────────────┘
                                                │
                                        Execute / Log
```

---

## 🤖 Agent Descriptions

| Agent | Responsibility |
|-------|---------------|
| **Traffic Monitoring Agent** | Ingests or simulates sensor data (density, speed, vehicle count) |
| **Congestion Detection Agent** | Rule-based threshold analysis; classifies anomaly type |
| **Incident Investigation Agent** | LLM + Weather API + RAG to classify incident, assign severity, generate reasoning |
| **Route Diversion Agent** | LLM-generated alternative route recommendations (≥2 options) |
| **Signal Optimization Agent** | LLM-recommended adaptive signal timing for affected intersections |
| **Traffic Response Coordination Agent** | Assembles full response plan, generates commuter & authority alerts |
| **Supervisor Agent** | LangGraph StateGraph controller; routes between agents; HITL gate |

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| LLM | Google Gemini 1.5 Flash |
| Agentic Framework | LangGraph (StateGraph) |
| UI | Streamlit |
| RAG | ChromaDB + Gemini Embeddings |
| Database | SQLite |
| Weather | OpenWeatherMap API (free tier) |
| Language | Python 3.11+ |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Linting | Ruff |
| Security | pip-audit, dotenv secrets management |

---

## 📁 Project Structure

```
urban-traffic-command/
├── app.py                          # Streamlit UI entry point
├── create_video.py                 # moviepy+PIL slideshow video generator
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── UrbanPulse_Presentation.pptx   # Project presentation (13 slides)
├── UrbanPulse_Demo.mp4            # Video demo (generated by create_video.py)
├── agents/                         # 7 specialized AI agents
│   ├── supervisor_agent.py
│   ├── traffic_monitoring_agent.py
│   ├── congestion_detection_agent.py
│   ├── incident_investigation_agent.py
│   ├── route_diversion_agent.py
│   ├── signal_optimization_agent.py
│   └── traffic_response_coordination_agent.py
├── tools/                          # Sensor, Weather, RAG, Alert tools
│   ├── sensor_tools.py
│   ├── weather_tools.py
│   └── rag_tools.py
├── rag/                            # ChromaDB + embeddings + documents
│   ├── embeddings.py
│   └── documents/
├── data/                           # SQLite + seed script
│   └── seed_data.py
├── utils/                          # Config, DB, audit logger, state models
│   ├── config.py
│   ├── db.py
│   ├── audit_logger.py
│   ├── llm_helpers.py
│   └── state_models.py
├── tests/                          # Unit tests + LLM tests + scenario docs
│   ├── test_llm_agent.py           # 15 comprehensive LLM evaluation tests
│   ├── test_congestion_detection.py
│   ├── test_sensor_tools.py
│   └── test_cases.md               # 8 system-level test scenarios
├── assets/                         # Architecture diagram + slide previews
└── logs/                           # Application logs
```

---

## ✅ Test Results

Run the full test suite:

```bash
pytest tests/ -v
```

**Results: 24/24 tests passing**

| Test Category | Count | Status |
|---------------|-------|--------|
| LLM Agent Tests (correctness, faithfulness, hallucination, robustness, safety, HITL) | 15 | ✅ All Pass |
| Congestion Detection Unit Tests | 6 | ✅ All Pass |
| Sensor Tools Unit Tests | 3 | ✅ All Pass |
| **Total** | **24** | **✅ 100%** |

### LLM Test Coverage

| Test ID | Category | Description |
|---------|----------|-------------|
| TC-LLM-01 | Correctness | Congestion classification accuracy |
| TC-LLM-02 | Correctness | CRITICAL accident classification |
| TC-LLM-03 | Faithfulness | RAG sources cited in output |
| TC-LLM-04 | Hallucination | Schema compliance (valid enum values) |
| TC-LLM-05 | Robustness | Malformed JSON triggers rule fallback |
| TC-LLM-06 | Robustness | API exception triggers fallback |
| TC-LLM-07 | Safety | Prompt injection resistance |
| TC-LLM-08 | RAG | Empty RAG → LLM general knowledge fallback |
| TC-LLM-09 | Robustness | Severity score out-of-range clamped to 1–10 |
| TC-LLM-10 | Explainability | Reasoning field always populated |
| TC-LLM-11 | Traceability | Agent trace entry appended after call |
| TC-LLM-12 | Performance | Latency < 2 seconds (mocked) |
| TC-LLM-13 | HITL | Severity ≥ 9 triggers approval gate |
| TC-LLM-14 | HITL | Severity < 9 skips gate |
| TC-LLM-15 | Schema | All required incident fields present |

---

## 🎬 Generate Video Demo

```bash
python create_video.py
```

Generates:
- `assets/slide_previews/slide_NN.png` — PNG previews of all 10 slides
- `UrbanPulse_Demo.mp4` — Full 80-second MP4 video (1920×1080, 24fps)

---

## 📋 Agentic AI Requirements (Guideline §4)

| Requirement | Implementation | Status |
|-------------|---------------|--------|
| Autonomous AI Agent | Supervisor Agent makes routing decisions | ✅ |
| Multiple Agent Roles (≥3) | 7 specialized agents | ✅ |
| Tool Calling (≥2 tools) | Sensor, Weather, RAG, Audit tools | ✅ |
| RAG | ChromaDB + Gemini Embeddings | ✅ |
| Agent Memory/State | LangGraph MemorySaver + SQLite | ✅ |
| Multi-Step Reasoning | 7-node pipeline with conditional edges | ✅ |
| Human-in-the-Loop | HITL approval for CRITICAL severity | ✅ |
| Explainability | Reasoning panel + agent trace in UI | ✅ |

---

## 🛡️ Responsible AI

- All **CRITICAL** decisions require human operator approval
- AI provides **recommendations only** — never autonomous enforcement
- **No PII** collected or stored
- All recommendations include **cited evidence and reasoning**
- Limitations clearly stated in UI banner
- **Hallucination mitigated** by RAG grounding and structured output parsing
- **Immutable audit trail** for every agent action and human decision

---

## 🌍 Business & Social Impact

| Metric | Target |
|--------|--------|
| Incident detection time | < 2 minutes (vs 8–15 min manual) |
| Incidents with AI response plan | ≥ 80% |
| CRITICAL incidents with human approval | 100% |
| Audit trail completeness | 100% of actions logged |

---

## ⚠️ Known Limitations

- Traffic sensor data is simulated; real CCTV/hardware integration not implemented
- Weather coordinates are approximate; not geocoded per real zone boundary
- LLM responses depend on API availability and model quality
- SQLite is suitable for prototype; production requires PostgreSQL or equivalent
- No real-time WebSocket streaming; page refresh required for live updates

---

## 🔮 Future Improvements

- Live sensor feed integration (SCATS, SCOOT, or custom IoT)
- WebSocket/SSE for real-time dashboard updates
- Multi-city/multi-tenant architecture
- SMS/push notification delivery for commuter alerts
- Historical ML model for congestion prediction
- Production Kubernetes deployment with Helm charts
- Mobile app for field operators

---

## 🔧 Environment Variables

Copy `.env.example` to `.env` and configure:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here   # Required
OPENWEATHER_API_KEY=your_openweather_key_here    # Optional — falls back to mock data
```

Get your Gemini API key free at: https://aistudio.google.com/app/apikey

---

## 📖 Target Users

| User | Role |
|------|------|
| TMC Operators | Primary — real-time monitoring, incident response approval |
| Traffic Commissioner | Executive — KPI review, audit oversight |
| Emergency Services | External — incident alerts, route coordination |
| Municipal IT Admins | System health, configuration, log access |

---

*UrbanPulse — IBM SkillsBuild Gartner Program 2026 Capstone | Agentic AI*  
*GitHub: https://github.com/vignesh2396/urban-traffic-command_IBMBob*
