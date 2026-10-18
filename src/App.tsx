import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  Activity,
  AlertCircle,
  ArrowDownToLine,
  Check,
  ChevronRight,
  CircleHelp,
  Code2,
  Copy,
  Database,
  FileJson2,
  Gauge,
  LayoutDashboard,
  LoaderCircle,
  RotateCcw,
  ScanSearch,
  Settings2,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
  Zap,
} from "lucide-react";

type BugAnalysis = {
  category: string;
  severity: string;
  priority: string;
  environment: string;
  reproducible: boolean | "unknown";
  probable_cause: string;
  suggested_action: string;
};
type AnalyzeResponse = {
  analysis: BugAnalysis;
  mode: ModelMode;
  model_name: string;
  elapsed_ms: number;
};

type ModelMode = "base" | "fine_tuned";
type ModelStatus = {
  status: "ok";
  base_model: string;
  adapter_available: boolean;
  adapter_path: string;
};
type Page = "playground" | "benchmarks" | "dataset" | "pipeline" | "schema";

const EXAMPLE_REPORT =
  "Whenever I submit the login form with a valid email but the wrong password, the page keeps spinning indefinitely. It happens on Chrome, and refreshing the page fixes it temporarily.";
const OUTPUT_SCHEMA = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  title: "Bug2JSON analysis",
  type: "object",
  additionalProperties: false,
  required: [
    "category",
    "severity",
    "priority",
    "environment",
    "reproducible",
    "probable_cause",
    "suggested_action",
  ],
  properties: {
    category: {
      type: "string",
      enum: [
        "authentication", "ui", "api", "database", "performance", "validation",
        "integration", "security", "deployment", "compatibility", "other", "unknown",
      ],
    },
    severity: { type: "string", enum: ["critical", "high", "medium", "low", "unknown"] },
    priority: { type: "string", enum: ["P1", "P2", "P3", "P4", "unknown"] },
    environment: { type: "string", minLength: 1, maxLength: 500 },
    reproducible: { type: ["boolean", "string"], enum: [true, false, "unknown"] },
    probable_cause: { type: "string", minLength: 1, maxLength: 500 },
    suggested_action: { type: "string", minLength: 1, maxLength: 500 },
  },
} as const;

const EXAMPLES = [
  { label: "Login spinner bug", report: EXAMPLE_REPORT },
  {
    label: "Slow report query",
    report: "The dashboard takes about 20 seconds to load when I open reports.",
  },
  {
    label: "Empty form crash",
    report: "The app crashes after clicking Save on an empty form.",
  },
  {
    label: "Mobile checkout",
    report: "On my iPhone, checkout returns to the cart after I tap Place order.",
  },
];

const SECTIONS: { id: Page; label: string; icon: typeof LayoutDashboard }[] = [
  { id: "playground", label: "Playground", icon: LayoutDashboard },
  { id: "benchmarks", label: "Benchmarks", icon: Activity },
  { id: "dataset", label: "Dataset", icon: Database },
];
const DIAGNOSTICS: { id: Page; label: string; icon: typeof Activity }[] = [
  { id: "pipeline", label: "Pipeline Monitor", icon: Gauge },
  { id: "schema", label: "Schema Inspector", icon: ScanSearch },
];

function titleCase(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function truncateReport(value: string): string {
  return Array.from(value).slice(0, 4000).join("");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isModelStatus(value: unknown): value is ModelStatus {
  return (
    isRecord(value) &&
    value.status === "ok" &&
    typeof value.base_model === "string" &&
    typeof value.adapter_available === "boolean" &&
    typeof value.adapter_path === "string"
  );
}

function isBugAnalysis(value: unknown): value is BugAnalysis {
  return (
    isRecord(value) &&
    [
      "authentication", "ui", "api", "database", "performance", "validation",
      "integration", "security", "deployment", "compatibility", "other", "unknown",
    ].includes(String(value.category)) &&
    ["critical", "high", "medium", "low", "unknown"].includes(String(value.severity)) &&
    ["P1", "P2", "P3", "P4", "unknown"].includes(String(value.priority)) &&
    typeof value.environment === "string" &&
    value.environment.trim().length > 0 &&
    value.environment.length <= 500 &&
    (typeof value.reproducible === "boolean" || value.reproducible === "unknown") &&
    typeof value.probable_cause === "string" &&
    value.probable_cause.length > 0 &&
    value.probable_cause.length <= 500 &&
    typeof value.suggested_action === "string" &&
    value.suggested_action.length > 0 &&
    value.suggested_action.length <= 500
  );
}

function isAnalyzeResponse(value: unknown): value is AnalyzeResponse {
  return (
    isRecord(value) &&
    isBugAnalysis(value.analysis) &&
    (value.mode === "base" || value.mode === "fine_tuned") &&
    typeof value.model_name === "string" &&
    typeof value.elapsed_ms === "number" &&
    Number.isFinite(value.elapsed_ms) &&
    value.elapsed_ms >= 0
  );
}

function App() {
  const [page, setPage] = useState<Page>("playground");
  const [report, setReport] = useState("");
  const [mode, setMode] = useState<ModelMode>("base");
  const [analysis, setAnalysis] = useState<BugAnalysis | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [serviceOnline, setServiceOnline] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState<"overview" | "json">("overview");
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [showExamples, setShowExamples] = useState(false);
  const [copyError, setCopyError] = useState("");
  const [exportFeedback, setExportFeedback] = useState("");
  const requestController = useRef<AbortController | null>(null);

  const json = useMemo(
    () => (analysis ? JSON.stringify(analysis, null, 2) : ""),
    [analysis],
  );
  const reportLength = Array.from(report).length;
  const modelReady = mode === "base" || Boolean(modelStatus?.adapter_available);

  useEffect(() => {
    let active = true;
    async function refreshStatus() {
      try {
        const response = await fetch("/api/status");
        if (!response.ok) throw new Error(`Status request failed (${response.status}).`);
        const data: unknown = await response.json();
        if (!isModelStatus(data)) throw new Error("The API returned an invalid status response.");
        if (active) {
          setModelStatus(data);
          setServiceOnline(true);
        }
      } catch {
        if (active) {
          setServiceOnline(false);
          setModelStatus(null);
        }
      }
    }
    void refreshStatus();
    const timer = window.setInterval(() => void refreshStatus(), 10_000);
    return () => {
      active = false;
      window.clearInterval(timer);
      requestController.current?.abort();
      requestController.current = null;
    };
  }, []);

  function resetAnalysis() {
    requestController.current?.abort();
    requestController.current = null;
    setLoading(false);
    setAnalysis(null);
    setError("");
    setElapsedMs(null);
    setCopyError("");
    setExportFeedback("");
  }

  useEffect(() => {
    if (!modelStatus?.adapter_available && mode === "fine_tuned") {
      setMode("base");
      resetAnalysis();
    }
  }, [mode, modelStatus?.adapter_available]);

  async function analyze() {
    const trimmedReport = report.trim();
    if (trimmedReport.length < 10) return;
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    setLoading(true);
    setError("");
    setAnalysis(null);
    setElapsedMs(null);
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ report: trimmedReport, mode }),
        signal: controller.signal,
      });
      let body: unknown;
      try {
        body = await response.json();
      } catch {
        body = null;
      }
      if (!response.ok) {
        const detail = isRecord(body) && typeof body.detail === "string"
          ? body.detail
          : `The API request failed (${response.status}).`;
        throw new Error(detail);
      }
      if (!isAnalyzeResponse(body)) {
        throw new Error("The API returned an invalid analysis response.");
      }
      setAnalysis(body.analysis);
      setElapsedMs(body.elapsed_ms);
      setActiveTab("overview");
    } catch (reason) {
      if (controller.signal.aborted) return;
      setError(
        reason instanceof Error
          ? reason.message
          : "Could not reach the Bug2JSON API. Check that the backend is running.",
      );
    } finally {
      if (requestController.current === controller) {
        requestController.current = null;
        setLoading(false);
      }
    }
  }

  async function copyJson() {
    if (!json) return;
    setCopyError("");
    setExportFeedback("");
    try {
      await navigator.clipboard.writeText(json);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopyError("Clipboard access is unavailable. Copy the JSON from the output panel.");
    }
  }

  function downloadJson() {
    if (!json) return;
    setCopyError("");
    setExportFeedback("");
    let objectUrl: string | null = null;
    const link = document.createElement("a");
    try {
      objectUrl = URL.createObjectURL(new Blob([json], { type: "application/json" }));
      link.href = objectUrl;
      link.download = "bug-report.json";
      link.style.display = "none";
      document.body.appendChild(link);
      link.click();
      setExportFeedback("Download requested: bug-report.json");
      window.setTimeout(() => {
        if (objectUrl) URL.revokeObjectURL(objectUrl);
      }, 1000);
    } catch (reason) {
      setCopyError(
        reason instanceof Error
          ? `Export failed: ${reason.message}`
          : "Export failed. Copy the JSON from the output panel instead.",
      );
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    } finally {
      link.remove();
    }
  }

  function chooseExample(exampleReport: string) {
    setReport(exampleReport);
    resetAnalysis();
    setShowExamples(false);
  }

  function clearReport() {
    setReport("");
    resetAnalysis();
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="Bug2JSON home">
          <span>BUG2JSON <small>PRO</small></span>
        </a>

        <div className="engine-card">
          <span className="engine-icon"><FileJson2 size={16} /></span>
          <span className="engine-copy">
            <strong>Bug2JSON Engine</strong>
            <small>{modelStatus?.base_model ?? "v0.1.0 · Local runtime"}</small>
          </span>
          <span className={`engine-indicator ${serviceOnline ? "connected" : ""}`} />
        </div>

        <nav className="side-nav" aria-label="Workspace">
          <span className="nav-caption">WORKSPACES</span>
          {SECTIONS.map(({ id, label, icon: Icon }) => (
            <button
              className={`side-link ${page === id ? "active" : ""}`}
              key={id}
              onClick={() => setPage(id)}
            >
              <Icon size={15} />
              <span>{label}</span>
            </button>
          ))}
          <span className="nav-caption diagnostics-caption">DIAGNOSTICS</span>
          {DIAGNOSTICS.map(({ id, label, icon: Icon }) => (
            <button
              className={`side-link ${page === id ? "active" : ""}`}
              key={id}
              onClick={() => setPage(id)}
            >
              <Icon size={15} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="runtime-label">
            <span>LOCAL API</span>
            <span className={serviceOnline ? "online-text" : "offline-text"}>
              {serviceOnline ? "Connected" : "Offline"}
            </span>
          </div>
          <div className="runtime-track"><span className={serviceOnline ? "online-track" : ""} /></div>
          <button className="preferences-link" onClick={() => setPage("pipeline")}>
            <span>Runtime status</span><Activity size={14} />
          </button>
        </div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <nav className="top-tabs" aria-label="Main sections">
            {SECTIONS.map(({ id, label }) => (
              <button
                className={page === id ? "selected" : ""}
                key={id}
                onClick={() => setPage(id)}
              >
                {label}
              </button>
            ))}
          </nav>
          <div className="topbar-right">
            <div className={`model-pill ${serviceOnline && modelReady ? "ready" : ""}`}>
              <span className="status-dot" />
              <span className="model-pill-name">
                {modelStatus?.base_model ?? "Model not connected"}
                {mode === "fine_tuned" && modelStatus?.adapter_available ? " · LoRA" : ""}
              </span>
              <span className="model-pill-state">
                {serviceOnline
                  ? mode === "fine_tuned"
                    ? modelReady ? "Adapter found" : "No adapter"
                    : "Configured"
                  : "Offline"}
              </span>
            </div>
            <button className="top-icon" aria-label="Pipeline monitor" onClick={() => setPage("pipeline")}>
              <Settings2 size={16} />
            </button>
          </div>
        </header>

        {page === "playground" && (
          <main className="page-content">
            <div className="page-heading">
              <div>
                <div className="eyebrow"><span className="blue-dot" /> ADAPTIVE EXTRACTION PIPELINE</div>
                <h1>Structured Bug Inference</h1>
              </div>
              <div className="example-picker">
                <button
                  className="example-toggle"
                  onClick={() => setShowExamples(!showExamples)}
                  aria-expanded={showExamples}
                >
                  Load example <ChevronRight size={13} />
                </button>
                {showExamples && (
                  <div className="example-menu">
                    {EXAMPLES.map((example) => (
                      <button key={example.label} onClick={() => chooseExample(example.report)}>
                        {example.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="analysis-columns">
              <section className="panel input-panel">
                <div className="panel-header">
                  <div className="panel-title"><FileJson2 size={15} /> Raw bug report</div>
                  <div className="panel-tools">
                    <span className="neutral-badge">Natural text</span>
                    <label className="mode-control">
                      <span>Model</span>
                      <select
                        value={mode}
                        onChange={(event) => {
                          setMode(event.target.value as ModelMode);
                          resetAnalysis();
                        }}
                        aria-label="Inference model"
                      >
                        <option value="base">Base</option>
                        <option value="fine_tuned" disabled={!modelStatus?.adapter_available}>LoRA</option>
                      </select>
                    </label>
                    <button className="bare-icon" onClick={clearReport} aria-label="Clear report">
                      <RotateCcw size={14} />
                    </button>
                  </div>
                </div>
                <textarea
                  className="report-editor"
                  value={report}
                  onChange={(event) => {
                    setReport(truncateReport(event.target.value));
                    resetAnalysis();
                  }}
                  placeholder="Describe what happened, what you expected, and any details that could help reproduce the issue..."
                  aria-label="Bug report"
                  onKeyDown={(event) => {
                    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                      event.preventDefault();
                      void analyze();
                    }
                  }}
                />
                <div className="input-bottom">
                  <div className="report-count">
                    <span>{reportLength}/4000 characters</span>
                    <i />
                    <span>~{Math.ceil(reportLength / 4)} tokens</span>
                    {report.trim().length > 0 && report.trim().length < 10 && (
                      <span className="minimum-length">Enter at least 10 characters</span>
                    )}
                  </div>
                  <button
                    className="extract-button"
                    onClick={analyze}
                    disabled={report.trim().length < 10 || loading}
                  >
                    {loading ? <LoaderCircle className="spin" size={15} /> : <Sparkles size={15} />}
                    {loading ? "Extracting..." : "Extract schema"}
                    <kbd>Ctrl/⌘ Enter</kbd>
                  </button>
                </div>
              </section>

              <section className="panel output-panel">
                <div className="panel-header">
                  <div className="panel-title output-title">
                    <span className={`output-state-icon ${analysis ? "success-icon" : error ? "error-icon" : "pending-icon"}`}>
                      {analysis ? <Check size={13} /> : error ? <AlertCircle size={13} /> : <Code2 size={13} />}
                    </span>
                    Extracted intelligence
                    {analysis && <span className="valid-badge"><ShieldCheck size={11} /> Valid schema</span>}
                  </div>
                  <div className="segmented-tabs">
                    <button
                      className={activeTab === "overview" ? "active" : ""}
                      onClick={() => setActiveTab("overview")}
                      disabled={!analysis}
                      aria-pressed={activeTab === "overview"}
                      title={!analysis ? "Generate an output to view it here." : undefined}
                    >
                      Overview
                    </button>
                    <button
                      className={activeTab === "json" ? "active" : ""}
                      onClick={() => setActiveTab("json")}
                      disabled={!analysis}
                      aria-pressed={activeTab === "json"}
                      title={!analysis ? "Generate an output to view it here." : undefined}
                    >
                      JSON
                    </button>
                  </div>
                </div>

                {error ? (
                  <div className="output-message error-message">
                    <AlertCircle size={19} />
                    <strong>Analysis unavailable</strong>
                    <span>{error}</span>
                    {!serviceOnline && <span>Start the API from the Bug2JSON folder, then retry.</span>}
                  </div>
                ) : !analysis ? (
                  <div className="output-message">
                    {loading ? <LoaderCircle className="spin" size={20} /> : <Code2 size={20} />}
                    <strong>{loading ? "Reading the report..." : "Your structured output will appear here"}</strong>
                    <span>
                      {loading
                        ? "The selected local model is processing the report."
                        : "Choose an example above, enter a report, then extract the schema."}
                    </span>
                    {!modelReady && serviceOnline && (
                      <span className="adapter-hint">No fine-tuned adapter yet. Switch to the base model for a baseline run.</span>
                    )}
                  </div>
                ) : activeTab === "json" ? (
                  <div className="json-view">
                    <div className="json-actions">
                      <span>RFC 8259 · 7 fields</span>
                      <div>
                        <button onClick={copyJson}>{copied ? <Check size={13} /> : <Copy size={13} />}{copied ? "Copied" : "Copy"}</button>
                        <button onClick={downloadJson}><ArrowDownToLine size={13} /> Export</button>
                      </div>
                    </div>
                    {copyError && <div className="action-feedback" role="status">{copyError}</div>}
                    {exportFeedback && <div className="action-feedback" role="status">{exportFeedback}</div>}
                    <pre>{json}</pre>
                  </div>
                ) : (
                  <>
                    <div className="metadata-grid">
                      <Meta label="Category" value={titleCase(analysis.category)} tone="blue" />
                      <Meta label="Severity" value={titleCase(analysis.severity)} tone="amber" />
                      <Meta label="Priority" value={`${analysis.priority}${analysis.priority === "unknown" ? "" : ` · ${priorityLabel(analysis.priority)}`}`} tone="violet" />
                      <Meta label="Client" value={analysis.environment} tone="neutral" />
                    </div>
                    <div className="insight-card">
                      <div className="insight-heading">
                        <span>PROBABLE CAUSE · UNVERIFIED</span>
                        <span className="prediction-note"><CircleHelp size={11} /> Model hypothesis</span>
                      </div>
                      <p>{analysis.probable_cause}</p>
                      <small><CircleHelp size={11} /> Based only on the details in the submitted report.</small>
                    </div>
                    <div className="insight-card action-card">
                      <div className="insight-heading">
                        <span>SUGGESTED INVESTIGATION</span>
                        <button className="copy-snippet" onClick={copyJson}>
                          {copied ? <Check size={12} /> : <Copy size={12} />}
                          {copied ? "Copied" : "Copy JSON"}
                        </button>
                      </div>
                      <p>{analysis.suggested_action}</p>
                    </div>
                    {copyError && <div className="action-feedback" role="status">{copyError}</div>}
                    <div className="result-bottom">
                      <span><span className="green-dot" /> Schema validated · evidence checks applied · review predictions</span>
                      <span>Latency: {elapsedMs === null ? "—" : `${elapsedMs} ms`}</span>
                    </div>
                  </>
                )}
              </section>
            </div>

            <div className="metric-strip">
              <Metric icon={<Code2 size={17} />} label="JSON schema" value="7 required fields" detail="Output contract" color="blue" />
              <Metric icon={<Zap size={17} />} label="Inference latency" value={elapsedMs === null ? "—" : `${elapsedMs} ms`} detail="End-to-end request" color="cyan" />
              <Metric icon={<ShieldCheck size={17} />} label="Generation mode" value="Deterministic" detail="Sampling disabled" color="green" />
            </div>
          </main>
        )}

        {page === "benchmarks" && <BenchmarksPage modelStatus={modelStatus} />}
        {page === "dataset" && <DatasetPage />}
        {page === "pipeline" && (
          <DiagnosticsPage
            title="Pipeline Monitor"
            eyebrow="DIAGNOSTICS / REQUEST FLOW"
            icon={<Activity size={18} />}
            online={serviceOnline}
            modelStatus={modelStatus}
          />
        )}
        {page === "schema" && <SchemaPage />}

        <footer className="main-footer">
          <span>BUG2JSON ENGINE <i>·</i> Local inference</span>
          <span>Predictions are reviewable, not verified facts</span>
        </footer>
      </div>
    </div>
  );
}

function priorityLabel(priority: string): string {
  return ({ P1: "Critical", P2: "High", P3: "Standard", P4: "Low" } as Record<string, string>)[priority] ?? "";
}

function Meta({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className={`meta-card ${tone}`}>
      <span className="meta-label">{label}</span>
      <span className="meta-value"><i />{value}</span>
    </div>
  );
}

function Metric({
  icon, label, value, detail, color,
}: { icon: ReactNode; label: string; value: string; detail: string; color: string }) {
  return (
    <div className="metric-card">
      <span className={`metric-icon ${color}`}>{icon}</span>
      <span className="metric-copy"><small>{label}</small><strong>{value}</strong><em>{detail}</em></span>
    </div>
  );
}

function BenchmarksPage({ modelStatus }: { modelStatus: ModelStatus | null }) {
  return (
    <main className="page-content secondary-page">
      <div className="eyebrow"><span className="green-dot" /> EVALUATION REPORT <span className="build-tag">HELD-OUT TEST SET</span></div>
      <div className="secondary-title-row">
        <div><h1>Model performance &amp; rigor</h1><p>Compare the base model and fine-tuned adapter on reports held out from training.</p></div>
        <span className="model-pill compact"><span className="status-dot" />{modelStatus?.base_model ?? "Model not connected"}</span>
      </div>
      <div className="benchmark-empty">
        <div className="empty-icon"><Activity size={22} /></div>
        <h2>Evaluation results will appear here</h2>
        <p>Run the comparison on a reviewed, held-out test split. We don’t show placeholder scores as model results.</p>
        <div className="metric-placeholder-grid">
          {["JSON validity", "Category macro-F1", "Severity accuracy", "Priority accuracy", "Field exact match", "Unsupported claims"].map((label) => (
            <div className="placeholder-metric" key={label}><span>{label}</span><strong>Not measured</strong></div>
          ))}
        </div>
        <div className="command-block"><span>RUN FROM THE PROJECT FOLDER</span><code>python training\evaluate.py --test data\splits\test.jsonl</code></div>
        {!modelStatus?.adapter_available && <p className="notice-line"><TriangleAlert size={14} /> Train a LoRA adapter before comparing base and fine-tuned results.</p>}
      </div>
    </main>
  );
}

function DatasetPage() {
  return (
    <main className="page-content secondary-page">
      <div className="eyebrow"><span className="blue-dot" /> DATASET WORKSPACE <span className="build-tag">LABEL POLICY</span></div>
      <div className="secondary-title-row">
        <div><h1>Dataset &amp; labeling</h1><p>Use consistent, human-reviewed labels. Starter examples are illustrative and excluded from training.</p></div>
        <span className="dataset-status"><span className="amber-dot" /> Starter examples · unreviewed</span>
      </div>
      <div className="dataset-layout">
        <section className="data-card">
          <div className="data-card-heading"><div><Database size={17} /><strong>Seven-field output contract</strong></div><span>STRICT JSON</span></div>
          <div className="field-list">
            {[
              ["category", "Subsystem affected"],
              ["severity", "Impact described by the reporter"],
              ["priority", "Urgency based on documented rules"],
              ["environment", "Browser, OS, device, or runtime"],
              ["reproducible", "true · false · unknown"],
              ["probable_cause", "A cautious, unverified hypothesis"],
              ["suggested_action", "One investigation step"],
            ].map(([key, meaning]) => <div className="field-row" key={key}><code>{key}</code><span>{meaning}</span></div>)}
          </div>
        </section>
        <section className="data-card guidance-card">
          <div className="data-card-heading"><div><CircleHelp size={17} /><strong>Labeling principles</strong></div></div>
          <ul>
            <li>Use <code>unknown</code> rather than inferring missing facts.</li>
            <li>Severity describes reported impact; priority reflects urgency and scope.</li>
            <li>Keep paraphrases from the same issue together using a stable group ID.</li>
            <li>Mark a row reviewed only after a person checks all seven labels.</li>
            <li>Keep an untouched test split for final comparisons.</li>
          </ul>
          <div className="command-block"><span>PREPARE GROUP-SAFE SPLITS</span><code>python training\prepare_data.py --input data\reviewed_reports.jsonl</code></div>
        </section>
      </div>
      <div className="notice-line"><TriangleAlert size={14} /> The bundled examples are synthetic format samples, not verified labels or a training dataset.</div>
    </main>
  );
}

function DiagnosticsPage({
  title, eyebrow, icon, online, modelStatus,
}: { title: string; eyebrow: string; icon: ReactNode; online: boolean; modelStatus: ModelStatus | null }) {
  return (
    <main className="page-content secondary-page">
      <div className="eyebrow"><span className={online ? "green-dot" : "amber-dot"} /> {eyebrow}</div>
      <div className="secondary-title-row"><div><h1>{title}</h1><p>Current runtime status and request configuration.</p></div></div>
      <div className="diagnostic-grid">
        <div className="diagnostic-card"><span className="diagnostic-icon">{icon}</span><small>API SERVICE</small><strong>{online ? "Connected" : "Not reachable"}</strong><p>{online ? "The local API is responding." : "Start the FastAPI server to enable analysis."}</p></div>
        <div className="diagnostic-card"><span className="diagnostic-icon"><FileJson2 size={18} /></span><small>OUTPUT VALIDATION</small><strong>Strict schema</strong><p>Responses are checked against all seven required fields.</p></div>
        <div className="diagnostic-card"><span className="diagnostic-icon"><Database size={18} /></span><small>BASE MODEL</small><strong>{modelStatus?.base_model ?? "Not configured"}</strong><p>Loaded on the first analysis request.</p></div>
        <div className="diagnostic-card"><span className="diagnostic-icon"><Sparkles size={18} /></span><small>LORA ADAPTER</small><strong>{modelStatus?.adapter_available ? "Available" : "Not trained"}</strong><p>{modelStatus?.adapter_path ?? "Train an adapter to enable fine-tuned mode."}</p></div>
      </div>
    </main>
  );
}

function SchemaPage() {
  return (
    <main className="page-content secondary-page">
      <div className="eyebrow"><span className="blue-dot" /> DIAGNOSTICS / OUTPUT CONTRACT</div>
      <div className="secondary-title-row"><div><h1>Schema inspector</h1><p>Fields accepted by the API and shown in the JSON response.</p></div><span className="valid-badge"><ShieldCheck size={12} /> 7 fields · strict validation</span></div>
      <div className="schema-code"><pre>{JSON.stringify(OUTPUT_SCHEMA, null, 2)}</pre></div>
      <div className="schema-note"><CircleHelp size={16} /><span><strong>Predictions are not incident facts.</strong> Severity, priority, cause, and action should be reviewed. The model must use <code>"unknown"</code> when evidence is missing.</span></div>
    </main>
  );
}

export default App;
