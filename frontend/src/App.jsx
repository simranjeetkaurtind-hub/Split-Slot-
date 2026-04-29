import React, { useEffect, useMemo, useState } from "react";

function resolveApiBase() {
  const env = import.meta.env.VITE_API_BASE;
  if (typeof env === "string" && env.trim() !== "") {
    return env.trim().replace(/\/$/, "");
  }
  if (typeof window !== "undefined") {
    const h = window.location.hostname;
    if (h === "localhost" || h === "127.0.0.1") {
      return "/api";
    }
  }
  return "http://localhost:8000";
}

const API_BASE = resolveApiBase();

const defaultConfig = {
  priority_weight: 0.35,
  demand_weight: 0.3,
  urgency_weight: 0.25,
  affinity_weight: 0.1,
  delta_threshold: 0.025,
};

const viewHints = {
  leadership: [
    "High demand roles were prioritized early.",
    "Lower priority roles received allocation as saturation increased.",
    "No slot wastage observed.",
    "Bench (ELTP) utilized only after real demand was fulfilled.",
  ],
  engineering: [],
};

const projectCycle = ["Alpha", "Beta", "Gamma", "Delta", "Bench"];

const buildSlots = (count) =>
  Array.from({ length: count }, (_, idx) => {
    const slotIdx = idx + 1;
    return {
      slot_id: `S${String(slotIdx).padStart(3, "0")}`,
      panel_id: `P${String(slotIdx).padStart(3, "0")}`,
      project: projectCycle[(slotIdx - 1) % projectCycle.length],
      tech_stack: "*",
      level: "*",
      batch_id: "1",
    };
  });

export default function App() {
  const [config, setConfig] = useState(defaultConfig);
  const [savedConfig, setSavedConfig] = useState(defaultConfig);
  const [scenarios, setScenarios] = useState([]);
  const [scenarioId, setScenarioId] = useState("");
  const [batchSizes, setBatchSizes] = useState([4, 10, 2, 15, 40, 8, 6, 5, 10]);
  const [useCustomScenario, setUseCustomScenario] = useState(false);
  const [customJosText, setCustomJosText] = useState("");
  const [customSlotCount, setCustomSlotCount] = useState(50);
  const [result, setResult] = useState(null);
  const [previousResult, setPreviousResult] = useState(null);
  const [compare, setCompare] = useState(false);
  const [viewMode, setViewMode] = useState("leadership");
  const [batchIndex, setBatchIndex] = useState(0);
  const [status, setStatus] = useState({ type: "idle", message: "" });

  useEffect(() => {
    fetch(`${API_BASE}/config`)
      .then((res) => res.json())
      .then((data) => {
        setConfig(data);
        setSavedConfig(data);
      })
      .catch((err) => {
        setStatus({ type: "error", message: `Config load failed: ${err}` });
      });

    fetch(`${API_BASE}/scenarios`)
      .then((res) => res.json().then((data) => ({ res, data })))
      .then(({ res, data }) => {
        if (!res.ok || !Array.isArray(data)) {
          setStatus({
            type: "error",
            message: `Scenario load failed: HTTP ${res.status} ${typeof data === "object" ? JSON.stringify(data) : String(data)}`,
          });
          return;
        }
        setScenarios(data);
        if (data.length > 0) setScenarioId(data[0].id);
      })
      .catch((err) => {
        setStatus({ type: "error", message: `Scenario load failed: ${err}` });
      });
  }, []);

  const scenario = useMemo(
    () => scenarios.find((s) => s.id === scenarioId),
    [scenarios, scenarioId]
  );

  useEffect(() => {
    if (useCustomScenario || !scenarioId || scenarios.length === 0) return;
    const s = scenarios.find((x) => x.id === scenarioId);
    if (!s?.jos) return;
    setCustomJosText(JSON.stringify(s.jos, null, 2));
    setCustomSlotCount(s.slots?.length || 50);
    const raw = s.batch_sizes;
    if (Array.isArray(raw) && raw.length > 0) {
      const parsed = raw
        .map((v) => Number(v))
        .filter((n) => Number.isFinite(n) && n > 0);
      if (parsed.length > 0) setBatchSizes(parsed);
    }
  }, [scenarioId, scenarios, useCustomScenario]);

  const weightSum = useMemo(
    () =>
      config.priority_weight +
      config.demand_weight +
      config.urgency_weight +
      config.affinity_weight,
    [config]
  );

  const runSimulation = async () => {
    if (!scenarioId && !useCustomScenario) return;
    setStatus({ type: "loading", message: "Running simulation..." });
    setPreviousResult(compare ? result : null);
    const payload = useCustomScenario
      ? {
          jos: JSON.parse(customJosText || "[]"),
          slots: buildSlots(customSlotCount),
          batch_sizes: batchSizes,
        }
      : {
          scenario_id: scenarioId,
          batch_sizes: batchSizes,
        };
    try {
      const res = await fetch(`${API_BASE}/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        throw new Error(`Simulation failed (${res.status})`);
      }
      const data = await res.json();
      setResult(data);
      setBatchIndex(0);
      setStatus({ type: "success", message: "Simulation complete." });
    } catch (err) {
      setStatus({ type: "error", message: `Simulation failed: ${err}` });
    }
  };

  const saveConfig = async () => {
    const res = await fetch(`${API_BASE}/config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    const data = await res.json();
    setSavedConfig(data);
    setConfig(data);
  };

  const resetConfig = () => setConfig(savedConfig);

  /** Stable JO order for tables: scenario roster first, then any extra ids from results (custom JOs). */
  const joDisplayOrder = useMemo(() => {
    const snap = result?.jo_snapshots_end;
    const fromScenario = (scenario?.jos || []).map((j) => j.id);
    if (!snap) return fromScenario;
    const extra = Object.keys(snap).filter((id) => !fromScenario.includes(id));
    return [...fromScenario, ...extra.sort()];
  }, [result, scenario]);

  const batchAllocations = useMemo(() => {
    if (!result?.assignments) return [];
    const bySlot = new Map();
    result.assignments.forEach((a) => bySlot.set(a.slot_id, a.jo_id));
    const keys =
      joDisplayOrder.length > 0
        ? joDisplayOrder
        : Object.keys(result.jo_snapshots_end || {}).sort();
    return (result.batch_debug || []).map((batch) => {
      const counts = Object.fromEntries(keys.map((id) => [id, 0]));
      let unassigned = 0;
      batch.slot_ids.forEach((slotId) => {
        const joId = bySlot.get(slotId) || "-";
        if (joId === "-") unassigned += 1;
        else if (Object.prototype.hasOwnProperty.call(counts, joId)) counts[joId] += 1;
        else counts[joId] = (counts[joId] || 0) + 1;
      });
      if (unassigned > 0) counts["(unassigned)"] = unassigned;
      return { batchIndex: batch.batch_index, counts };
    });
  }, [result, joDisplayOrder]);
  const batches = result?.batch_debug ?? [];
  const activeBatch = batches[batchIndex];

  const downloadBatchLog = () => {
    if (!batches.length) return;
    const content = batches
      .map((batch) => {
        const header = `Batch ${batch.batch_index + 1}: ${batch.slot_ids.join(", ")}`;
        return [header, ...batch.debug_log_lines, ""].join("\n");
      })
      .join("\n");
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "batch_output_full.txt";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="container">
      <div className="panel">
        <h2>Simulation Control Panel</h2>
        {status.message && (
          <p className={status.type === "error" ? "warning" : ""}>{status.message}</p>
        )}
        <div className="grid grid-2">
          <div>
            <label>Scenario</label>
            <select value={scenarioId} onChange={(e) => setScenarioId(e.target.value)}>
              {scenarios.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
            <p>{scenario?.description}</p>
            {scenario && (
              <div className="panel">
                <h3>Scenario Inputs</h3>
                <div className="row">
                  <span className="pill">Slots: {scenario.slots?.length || 0}</span>
                  <span className="pill">
                    Batch Sizes: {(scenario.batch_sizes || []).join(", ")}
                  </span>
                </div>
                <table className="table">
                  <thead>
                    <tr>
                      <th>JO</th>
                      <th>Priority</th>
                      <th>Total Demand</th>
                      <th>Days Remaining</th>
                      <th>Project</th>
                      <th>Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scenario.jos.map((jo) => (
                      <tr key={jo.id}>
                        <td>{jo.id}</td>
                        <td>{jo.priority}</td>
                        <td>{jo.initial_demand}</td>
                        <td>{jo.days_remaining}</td>
                        <td>{jo.project}</td>
                        <td>{jo.type}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <label className="row">
              <input
                type="checkbox"
                checked={useCustomScenario}
                onChange={() => setUseCustomScenario((prev) => !prev)}
              />
              Use custom scenario
            </label>
            {useCustomScenario && (
              <div className="grid">
                <label>Custom JOs (JSON array)</label>
                <textarea
                  rows="10"
                  value={customJosText}
                  onChange={(e) => setCustomJosText(e.target.value)}
                />
                <label>Slots to simulate</label>
                <input
                  type="number"
                  min="1"
                  value={customSlotCount}
                  onChange={(e) => setCustomSlotCount(Number.parseInt(e.target.value, 10) || 0)}
                />
              </div>
            )}
          </div>
          <div>
            <label>Batch sizes</label>
            <input
              value={batchSizes.join(", ")}
              onChange={(e) =>
                setBatchSizes(
                  e.target.value
                    .split(",")
                    .map((v) => parseInt(v.trim(), 10))
                    .filter((v) => !Number.isNaN(v) && v > 0)
                )
              }
            />
            <div className="row">
              <button onClick={runSimulation}>Run Simulation</button>
              <label>
                <input
                  type="checkbox"
                  checked={compare}
                  onChange={() => setCompare((prev) => !prev)}
                />
                Compare with Previous Run
              </label>
            </div>
          </div>
        </div>
      </div>

      <div className="panel">
        <h2>Config Console</h2>
        <div className="grid grid-2">
          {[
            ["Priority Weight", "priority_weight"],
            ["Demand Weight", "demand_weight"],
            ["Urgency Weight", "urgency_weight"],
            ["Affinity Weight", "affinity_weight"],
          ].map(([label, key]) => (
            <div key={key}>
              <label>{label}</label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={config[key]}
                onChange={(e) =>
                  setConfig((prev) => ({
                    ...prev,
                    [key]: Number.parseFloat(e.target.value),
                  }))
                }
              />
            </div>
          ))}
          <div>
            <label>Delta Threshold</label>
            <input
              type="number"
              step="0.001"
              value={config.delta_threshold}
              onChange={(e) =>
                setConfig((prev) => ({
                  ...prev,
                  delta_threshold: Number.parseFloat(e.target.value),
                }))
              }
            />
          </div>
        </div>
        {Math.abs(weightSum - 1) > 0.001 && (
          <p className="warning">Weight sum is {weightSum.toFixed(2)} (should be 1.00).</p>
        )}
        <div className="row">
          <button onClick={saveConfig}>Save Config</button>
          <button className="secondary" onClick={resetConfig}>
            Reset to Default
          </button>
        </div>
      </div>

      <div className="panel">
        <h2>Dashboard</h2>
        <div className="row">
          <button onClick={() => setViewMode("leadership")}>Leadership View</button>
          <button className="secondary" onClick={() => setViewMode("engineering")}>
            Engineering View
          </button>
        </div>
        {viewMode === "leadership" && (
          <ul>
            {viewHints.leadership.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        )}
        {viewMode === "engineering" && (
          <p>Scores, delta comparisons, saturation bands, and split logic are shown in batch details.</p>
        )}
      </div>

      <div className="panel">
        <h2>Batch Simulator</h2>
        {result ? (
          <>
            <div className="row">
              <button
                className="secondary"
                onClick={() => setBatchIndex((idx) => Math.max(0, idx - 1))}
              >
                Previous
              </button>
              <button
                className="secondary"
                onClick={() =>
                  setBatchIndex((idx) => Math.min(batches.length - 1, idx + 1))
                }
              >
                Next
              </button>
              <span className="pill">
                Batch {batchIndex + 1} / {batches.length}
              </span>
              <button className="secondary" onClick={downloadBatchLog}>
                Download Full Batch Log
              </button>
            </div>
            {activeBatch && (
              <div>
                <h3>Batch {activeBatch.batch_index + 1}</h3>
                <p>Slots: {activeBatch.slot_ids.join(", ")}</p>
                {batchAllocations[batchIndex] && (
                  <table className="table">
                    <thead>
                      <tr>
                        <th>JO</th>
                        <th>Slots Assigned</th>
                      </tr>
                    </thead>
                    <tbody>
                      {joDisplayOrder.map((joId) => {
                        const count = batchAllocations[batchIndex].counts[joId] ?? 0;
                        return (
                          <tr key={joId}>
                            <td>{joId}</td>
                            <td>{count}</td>
                          </tr>
                        );
                      })}
                      {(batchAllocations[batchIndex].counts["(unassigned)"] ?? 0) > 0 && (
                        <tr key="unassigned">
                          <td>(unassigned)</td>
                          <td>{batchAllocations[batchIndex].counts["(unassigned)"]}</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                )}
                <pre style={{ whiteSpace: "pre-wrap" }}>
                  {activeBatch.debug_log_lines.join("\n")}
                </pre>
              </div>
            )}
          </>
        ) : (
          <p>Run the simulation to view batch-by-batch allocation.</p>
        )}
      </div>

      <div className="panel">
        <h2>Explainability</h2>
        {result ? (
          <ul>
            {result.explanations?.slice(0, 15).map((exp) => (
              <li key={`${exp.slot_id}-${exp.jo_id}`}>{exp.reason}</li>
            ))}
          </ul>
        ) : (
          <p>No explanations yet.</p>
        )}
      </div>

      <div className="panel">
        <h2>Inputs & Results Summary</h2>
        {scenario && (
          <>
            <h3>Scenario Inputs</h3>
            <div className="row">
              <span className="pill">Slots processed: {scenario.slots?.length || 0}</span>
              <span className="pill">
                Batch Sizes: {(scenario.batch_sizes || batchSizes).join(", ")}
              </span>
            </div>
            <table className="table">
              <thead>
                <tr>
                  <th>JO</th>
                  <th>Priority</th>
                  <th>Total Demand</th>
                  <th>Remaining (after run)</th>
                  {result ? <th>Slots allocated</th> : null}
                  <th>Days Remaining</th>
                  <th>Project</th>
                  <th>Type</th>
                </tr>
              </thead>
              <tbody>
                {scenario.jos.map((jo) => {
                  const end = result?.jo_snapshots_end?.[jo.id];
                  return (
                    <tr key={jo.id}>
                      <td>{jo.id}</td>
                      <td>{jo.priority}</td>
                      <td>{jo.initial_demand}</td>
                      <td>{end != null ? end.active_demand : jo.active_demand}</td>
                      {result ? <td>{end != null ? end.slots_allocated : "—"}</td> : null}
                      <td>{jo.days_remaining}</td>
                      <td>{jo.project}</td>
                      <td>{jo.type}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </>
        )}
        {result ? (
          <>
            <h3>Results (same order as scenario)</h3>
            <p className="pill">
              JOs with 0 slots were still evaluated; caps/score may have blocked assignment.
            </p>
            <table className="table">
              <thead>
                <tr>
                  <th>JO</th>
                  <th>Total Slots</th>
                  <th>Saturation</th>
                  <th>% Share</th>
                </tr>
              </thead>
              <tbody>
                {joDisplayOrder.map((id) => {
                  const jo = result.jo_snapshots_end?.[id];
                  if (!jo) return null;
                  return (
                    <tr key={id}>
                      <td>{jo.id}</td>
                      <td>{jo.slots_allocated}</td>
                      <td>
                        {jo.initial_demand
                          ? ((jo.slots_allocated / jo.initial_demand) * 100).toFixed(1)
                          : "0.0"}
                        %
                      </td>
                      <td>
                        {result.assignments?.length
                          ? ((jo.slots_allocated / result.assignments.length) * 100).toFixed(1)
                          : "0.0"}
                        %
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </>
        ) : (
          <p>No results yet.</p>
        )}
        {compare && previousResult && (
          <p className="pill">Previous run available for comparison.</p>
        )}
      </div>
    </div>
  );
}
