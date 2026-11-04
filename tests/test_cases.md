# UrbanPulse — Test Cases
# AI-Based Urban Traffic Command & Incident Response System
# Guideline Requirement: §8 — At least 5 realistic test scenarios

## Test Scenario TS-01: Normal Traffic — No Alert
**Type:** Normal Use Case
**Precondition:** Zone Z-01 has density=0.30, speed=55 km/h, clear weather

**Steps:**
1. Navigate to Live Zone Monitor
2. Select Zone Z-01, Scenario = "normal"
3. Click "Trigger AI Investigation"

**Expected Result:**
- Agent classifies as NORMAL
- No incident card in Incident Queue
- Audit log shows: TrafficMonitoringAgent → SENSOR_READ, CongestionDetectionAgent → ANOMALY_NONE
- Pipeline terminates at congestion node (no anomaly edge taken)

**Pass Criteria:**
- `anomaly.detected == False`
- No incident record created in DB
- No alert messages generated

---

## Test Scenario TS-02: Moderate Congestion — Auto-Approved Response
**Type:** Alternative User Scenario
**Precondition:** Zone Z-07 (Old Town Market) — density=0.82, speed=16 km/h, no adverse weather

**Steps:**
1. Select Zone Z-07, Scenario = "congestion"
2. Click "Trigger AI Investigation"
3. Navigate to Incident Queue

**Expected Result:**
- Incident classified as CONGESTION or INCIDENT
- Severity score 5–8 (below CRITICAL threshold of 9)
- 2 diversion routes displayed
- Signal change recommendations shown
- `requires_approval = False` → Auto-approved
- Commuter alert generated

**Pass Criteria:**
- `incident.severity_score` between 4 and 8
- `response_plan.requires_approval == False`
- `len(diversion_routes) >= 2`
- Incident record saved in DB with `status = 'CLOSED'`

---

## Test Scenario TS-03: Critical Accident — HITL Gate Triggered
**Type:** High-Risk / Escalation Scenario
**Precondition:** Zone Z-12 — density=0.97, speed=2 km/h, heavy rain scenario

**Steps:**
1. Select Zone Z-07 (closest available), Scenario = "accident"
2. Click "Trigger AI Investigation"
3. Navigate to Incident Queue
4. Observe HITL approval gate
5. Click "APPROVE"

**Expected Result:**
- Incident classified as ACCIDENT or INCIDENT with CRITICAL severity
- `response_plan.requires_approval == True`
- Red "CRITICAL: Human Approval Required" banner shown
- Approve/Reject buttons rendered
- On APPROVE: incident status → CLOSED, hitl_decisions record inserted
- Audit log shows: HITL → PLAN_APPROVED with operator_id

**Pass Criteria:**
- `incident.severity_score >= 9`
- `state.awaiting_human == True` before approval
- `hitl_decisions.decision == 'APPROVED'` after click
- `incidents.status == 'CLOSED'` after approval

---

## Test Scenario TS-04: Missing Sensor Data — Graceful Degradation
**Type:** Missing / Invalid Input
**Precondition:** Zone Z-99 (non-existent zone ID)

**Steps:**
1. Directly invoke `get_latest_sensor_reading("Z-99")` in test
2. Verify fallback behaviour

**Expected Result:**
- `get_latest_sensor_reading` returns `None`
- Audit log entry: SensorTool → SENSOR_DATA_MISSING
- `simulate_sensor_reading` called as fallback
- Pipeline continues without crash
- Anomaly assessed on simulated data

**Pass Criteria:**
- No unhandled exception raised
- Audit log contains SENSOR_DATA_MISSING entry
- Final state has `sensor_reading != None` (simulation fallback kicked in)

---

## Test Scenario TS-05: Concurrent Multi-Zone Incidents
**Type:** System Stress / Prioritisation
**Precondition:** Trigger two zones simultaneously in the same Streamlit session

**Steps:**
1. Trigger Z-03 with scenario="congestion"
2. Trigger Z-07 with scenario="accident"
3. Navigate to Incident Queue

**Expected Result:**
- Two incident entries in `active_incidents` session state
- Incident Queue sorted by severity (accident first)
- Both response plans generated independently
- Z-07 accident shows higher severity than Z-03 congestion

**Pass Criteria:**
- `len(active_incidents) == 2`
- Z-07 incident appears before Z-03 in sorted queue
- Each incident has its own `incident_id` (UUIDs differ)
- Both incidents saved in `incidents` table

---

## Test Scenario TS-06: RAG No Result Found
**Type:** Tool Failure / Empty Result
**Precondition:** ChromaDB collection is empty OR query has no close matches

**Steps:**
1. Call `retrieve_regulations("xyzzy unknown term 99999")` directly
2. Observe agent behaviour when rag_docs = []

**Expected Result:**
- `retrieve_regulations` returns `[]`
- Audit log: RAGTool → RAG_COLLECTION_EMPTY or 0 hits
- Incident Investigation Agent proceeds with LLM general knowledge
- Response includes `rag_sources = []`
- Reasoning notes that no regulation was found

**Pass Criteria:**
- No exception raised when RAG returns empty
- `incident.rag_sources == []`
- Incident still classified (LLM fallback used)
- Audit log shows RAG query with `hits: 0`

---

## Test Scenario TS-07: Operator Rejects Response Plan
**Type:** HITL Rejection Flow
**Precondition:** A CRITICAL incident has been detected and is awaiting approval

**Steps:**
1. Trigger Zone with scenario="accident"
2. Navigate to Incident Queue
3. Click "REJECT"

**Expected Result:**
- `hitl_decisions.decision == 'REJECTED'`
- Incident flagged as ESCALATED
- Audit log shows: HITL → PLAN_REJECTED with operator_id
- UI shows rejection confirmation message

**Pass Criteria:**
- `hitl_decisions.decision == 'REJECTED'`
- State records rejection without crashing
- Audit log has PLAN_REJECTED entry with human_actor

---

## Test Scenario TS-08: LLM API Failure — Graceful Fallback
**Type:** External API Failure
**Precondition:** Gemini API key is invalid or API is unreachable

**Steps:**
1. Set GEMINI_API_KEY to an invalid value in .env
2. Trigger an investigation
3. Observe error handling

**Expected Result:**
- `IncidentInvestigationAgent` catches the API exception
- Audit log: LLM_FALLBACK_USED
- Rule-based fallback classification used
- No unhandled exception or Streamlit crash
- UI displays warning (if applicable)
- Pipeline completes with degraded but valid output

**Pass Criteria:**
- No unhandled exception
- `incident.classification` is populated (from fallback)
- Audit log contains LLM_FALLBACK_USED entry
- `incident.reasoning` mentions "fallback"
