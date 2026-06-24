import React, { useEffect, useMemo, useState } from "react";
import {
  Activity,
  Download,
  FileText,
  Loader2,
  Shield,
  ShieldCheck,
  TrendingDown,
  TrendingUp,
  X,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Footer from "../components/Footer.jsx";
import { exportAnalytics, fetchAnalytics, streamLogsUrl } from "../services/api.js";


const pct = (value) => `${Number(value || 0).toFixed(1)}%`;

export default function AnalyticsPage({ documentsCount }) {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [dateRange, setDateRange] = useState("today");
  const [customRange, setCustomRange] = useState({ start: "", end: "" });
  const [exportOpen, setExportOpen] = useState(false);
  const [liveOpen, setLiveOpen] = useState(false);
  const [retryOpen, setRetryOpen] = useState(false);
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      setLoading(true);
      setError("");
      setAnalytics(null);
      try {
        let range = dateRange;
        let start = null;
        let end = null;
        if (dateRange === "custom") {
          if (!customRange.start || !customRange.end) {
            if (mounted) setLoading(false);
            return;
          }
          start = customRange.start;
          end = customRange.end;
        }
        const data = await fetchAnalytics(range, start, end);
        if (mounted) setAnalytics(data);
      } catch (apiError) {
        if (mounted) setError(apiError.response?.data?.detail || apiError.message || "Unable to load analytics.");
      } finally {
        if (mounted) setLoading(false);
      }
    };
    load();
    return () => {
      mounted = false;
    };
  }, [dateRange, customRange.start, customRange.end]);

  useEffect(() => {
    if (!liveOpen) return undefined;
    const source = new EventSource(streamLogsUrl);
    source.onmessage = (event) => {
      try {
        const nextLog = JSON.parse(event.data);
        setLogs((current) => [nextLog, ...current].slice(0, 30));
      } catch {
        setLogs((current) => [{ timestamp: "--:--:--", step: event.data, status: "ok" }, ...current].slice(0, 30));
      }
    };
    source.onerror = () => {
      source.close();
    };
    return () => source.close();
  }, [liveOpen]);

  const metrics = analytics || {};
  const totalQueries = Number(metrics.total_queries || metrics.questions_asked || 0);
  const avgConfidence = Number(metrics.average_confidence || 0);
  const avgFaithfulness = Number(metrics.average_faithfulness || 0);
  const avgRelevance = Number(metrics.average_relevance || 0);
  const avgPrecision = Number(metrics.average_precision || 0);
  const avgRecall = Number(metrics.average_recall || 0);
  const hallucinationRate = Number(metrics.hallucination_rate || 0);
  const retryRate = Number(metrics.retry_rate || 0);
  const retryCount = Number(metrics.retry_count || 0);
  const hallucinationCount = Number(metrics.hallucination_count || 0);
  const reliableCount = Number(metrics.reliable_count || 0);
  const indexedDocuments = Number(metrics.documents_indexed ?? documentsCount ?? 0);
  const history = metrics.history ?? [];

  const overviewChart = useMemo(() => ([
    { name: "Total Queries", value: totalQueries },
    { name: "Retried Queries", value: retryCount },
    { name: "Reliable Answers", value: reliableCount },
    { name: "Hallucinations", value: hallucinationCount },
  ]), [totalQueries, retryCount, reliableCount, hallucinationCount]);

  const qualityCards = useMemo(() => ([
    { label: "Answer Relevance", value: avgRelevance, tone: "text-accent", dataKey: "average_relevance" },
    { label: "Context Precision", value: avgPrecision, tone: "text-[#6A2A05]", dataKey: "average_precision" },
    { label: "Context Recall", value: avgRecall, tone: "text-[#8C5A46]", dataKey: "average_recall" },
  ]), [avgRelevance, avgPrecision, avgRecall]);

  const pieData = [
    { name: "Reliable", value: reliableCount },
    { name: "Hallucination Risk", value: hallucinationCount },
  ].filter((item) => item.value > 0);

  return (
    <>
      <main className="px-5 py-9 sm:px-7">
        <div className="mx-auto max-w-7xl">
          <div className="flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
            <div>
              <div className="mb-3 flex flex-wrap gap-3">
                <span className="rounded-full bg-[#F5DFD2] px-3 py-1 text-xs font-black text-accent">Llama-3-70B Powered</span>
                <span className="rounded-full bg-[#F5DFD2] px-3 py-1 text-xs font-black text-accent">Groq Inference</span>
              </div>
              <h1 className="text-4xl font-black tracking-tight">Platform Analytics</h1>
              <p className="mt-3 max-w-2xl text-sm leading-6 text-[#6A4034]">
                Performance metrics for the self-correcting retrieval pipeline and hallucination prevention engine.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <div className="flex rounded-md border border-line bg-paper p-1">
                {["All", "Today", "7d", "30d", "Custom"].map((range) => (
                  <button
                    key={range}
                    type="button"
                    onClick={() => setDateRange(range.toLowerCase())}
                    className={`rounded px-3 py-2 text-xs font-black ${dateRange === range.toLowerCase() ? "bg-accent text-white" : "text-[#6A4034]"}`}
                  >
                    {range}
                  </button>
                ))}
              </div>
              {dateRange === "custom" && (
                <div className="flex items-center gap-2 rounded-md border border-line bg-paper px-3 py-2">
                  <input type="date" className="bg-transparent text-xs font-bold" onChange={(e) => setCustomRange((p) => ({ ...p, start: e.target.value }))} />
                  <span className="text-muted">-</span>
                  <input type="date" className="bg-transparent text-xs font-bold" onChange={(e) => setCustomRange((p) => ({ ...p, end: e.target.value }))} />
                </div>
              )}
              <div className="relative">
                <button type="button" onClick={() => setExportOpen(!exportOpen)} className="rounded-md bg-paper px-5 py-3 text-sm font-bold text-ink">
                  <Download className="mr-2 inline" size={15} />
                  Export Report
                </button>
                {exportOpen && (
                  <div className="absolute right-0 top-12 z-20 w-36 rounded-md border border-line bg-[#FFFEFC] p-1 text-sm shadow-soft">
                    {["CSV", "JSON", "PDF"].map((format) => (
                      <button
                        key={format}
                        type="button"
                        className="block w-full rounded px-3 py-2 text-left font-semibold hover:bg-paper"
                        onClick={async () => {
                           setExportOpen(false);
                           try {
                            const blob = await exportAnalytics(
                              format.toLowerCase(),
                              dateRange,
                              customRange.start,
                              customRange.end,
                            );
                             const url = window.URL.createObjectURL(blob);
                             const a = document.createElement('a');
                             a.href = url;
                             a.download = `analytics_report.${format.toLowerCase()}`;
                             document.body.appendChild(a);
                             a.click();
                             a.remove();
                             window.URL.revokeObjectURL(url);
                           } catch (e) {
                             console.error('Export failed:', e);
                             alert('Export failed. Please try again.');
                           }
                         }}
                      >
                        {format}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              <button type="button" onClick={() => setLiveOpen(true)} className="rounded-md bg-accent px-5 py-3 text-sm font-black text-white">Live Feed</button>
            </div>
          </div>

          {loading && (
            <div className="mt-8 flex items-center gap-2 rounded-lg border border-line bg-paper p-4 text-sm font-semibold text-accent">
              <Loader2 className="animate-spin" size={18} />
              Loading analytics...
            </div>
          )}
          {error && <p className="mt-8 rounded-lg bg-red-50 p-4 text-sm text-red-700">{error}</p>}

          <div className="mt-9 grid gap-6 lg:grid-cols-3">
            <MetricCard icon={Activity} label="Total Queries" value={totalQueries.toLocaleString()} note="Logged from real question history" history={history} dataKey="questions" />
            <MetricCard icon={ShieldCheck} label="Faithfulness" value={pct(avgFaithfulness)} note={`${pct(avgRelevance)} answer relevance`} history={history} dataKey="average_faithfulness" />
            <div className="rounded-lg bg-accent p-8 text-white shadow-soft">
              <Shield size={24} />
              <p className="mt-6 text-xs font-black uppercase tracking-wide text-[#FFE4D6]">Hallucination Rate</p>
              <p className="mt-2 text-4xl font-black">{pct(hallucinationRate)}</p>
              <div className="mt-8 border-t border-white/20 pt-5 text-sm font-black">Self-Correction Active</div>
            </div>
          </div>

          <div className="mt-6 grid gap-6 md:grid-cols-[1fr_1fr_2fr]">
            <SmallMetric label="Average Confidence" value={pct(avgConfidence)} helper="Grounding and answer quality score" icon={TrendingUp} history={history} dataKey="average_confidence" />
            <button type="button" onClick={() => setRetryOpen(true)} className="text-left">
              <SmallMetric label="Retry Rate" value={pct(retryRate)} helper="Queries that needed a rewrite-and-retry cycle" icon={TrendingDown} history={history} dataKey="retries" />
            </button>
            <div className="rounded-lg border border-line bg-paper p-6">
              <p className="text-sm font-semibold text-[#6A4034]">Documents Indexed</p>
              <div className="mt-2 flex items-center justify-between">
                <p className="text-3xl font-black">{indexedDocuments.toLocaleString()}</p>
                <FileText className="text-accent" size={26} />
              </div>
              <p className="mt-2 text-sm text-muted">{Number(metrics.vector_store_mb || 0).toFixed(2)} MB vector store</p>
            </div>
          </div>

          <div className="mt-6 grid gap-6 lg:grid-cols-3">
            {qualityCards.map((item) => (
              <div key={item.label} className="rounded-lg border border-line bg-[#FFFEFC] p-6">
                <p className="text-sm font-semibold text-[#6A4034]">{item.label}</p>
                <p className={`mt-2 text-3xl font-black ${item.tone}`}>{pct(item.value)}</p>
                <Sparkline data={history} dataKey={item.dataKey} compact />
              </div>
            ))}
          </div>

          <section className="mt-9 rounded-lg border border-line bg-paper p-6 shadow-soft">
            <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <h2 className="text-2xl font-black">Confidence Trends</h2>
                <p className="text-sm text-[#6A4034]">Daily confidence performance from real query evaluations.</p>
              </div>
              <div className="flex gap-4 text-xs font-semibold text-[#6A4034]">
                <span className="flex items-center gap-2"><span className="h-3 w-3 rounded-full bg-accent" /> Confidence</span>
                <span className="flex items-center gap-2"><span className="h-3 w-3 rounded-full bg-[#6A2A05]" /> Faithfulness</span>
              </div>
            </div>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history}>
                  <CartesianGrid stroke="#E2D8CF" vertical={false} />
                  <XAxis dataKey="date" tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} domain={[0, 100]} />
                  <Tooltip />
                  <Line type="monotone" dataKey="average_confidence" stroke="#C84B2F" strokeWidth={3} dot={{ r: 3 }} />
                  <Line type="monotone" dataKey="average_faithfulness" stroke="#6A2A05" strokeWidth={3} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="mt-6 grid gap-6 lg:grid-cols-2">
            <div className="rounded-lg border border-line bg-[#FFFEFC] p-6">
              <h2 className="text-xl font-black">Verification Split</h2>
              <div className="mt-4 h-64">
                {pieData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={pieData} dataKey="value" nameKey="name" innerRadius={62} outerRadius={96}>
                        {pieData.map((entry) => (
                          <Cell key={entry.name} fill={entry.name === "Reliable" ? "#C84B2F" : "#6A2A05"} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="grid h-full place-items-center text-sm text-muted">No question history yet.</div>
                )}
              </div>
            </div>
            <div className="rounded-lg border border-line bg-[#FFFEFC] p-6">
              <h2 className="text-xl font-black">Retry Statistics</h2>
              <div className="mt-4 h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={history}>
                    <CartesianGrid stroke="#E2D8CF" vertical={false} />
                    <XAxis dataKey="date" tickLine={false} axisLine={false} />
                    <YAxis tickLine={false} axisLine={false} allowDecimals={false} />
                    <Tooltip />
                    <Bar dataKey="retries" radius={[3, 3, 0, 0]}>
                      {history.map((entry) => (
                        <Cell key={entry.date} fill={entry.retries > 0 ? "#6A2A05" : "#C84B2F"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </section>

          <section className="mt-6 grid gap-6 lg:grid-cols-2">
            <div className="rounded-lg border border-line bg-paper p-6">
              <h2 className="text-xl font-black">Self-Healing Timeline</h2>
              <div className="mt-4 h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={overviewChart}>
                    <CartesianGrid stroke="#E2D8CF" vertical={false} />
                    <XAxis dataKey="name" tickLine={false} axisLine={false} />
                    <YAxis tickLine={false} axisLine={false} />
                    <Tooltip cursor={{ fill: "#F8F6F2" }} />
                    <Bar dataKey="value" radius={[3, 3, 0, 0]}>
                      {overviewChart.map((entry) => (
                        <Cell key={entry.name} fill={entry.name === "Hallucinations" ? "#6A2A05" : "#C84B2F"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="rounded-lg border border-line bg-[#FFFEFC] p-6">
              <h2 className="text-xl font-black">System Status</h2>
              <div className="mt-6 space-y-4">
                <StatusRow label="API Health" value={metrics.status || "unknown"} />
                <StatusRow label="Total Queries" value={totalQueries.toLocaleString()} />
                <StatusRow label="Average Precision" value={pct(avgPrecision)} />
                <StatusRow label="Average Recall" value={pct(avgRecall)} />
              </div>
            </div>
          </section>
        </div>
      </main>
      {retryOpen && (
        <Modal title="Retry Rate Drill-Down" onClose={() => setRetryOpen(false)}>
          {(metrics.failed_queries || []).length > 0 ? (
            <div className="space-y-3">
              {metrics.failed_queries.map((item, index) => (
                <div key={`${item.query}-${index}`} className="rounded-md border border-line bg-paper p-3 text-sm">
                  <p className="font-black">{item.query}</p>
                  <p className="mt-1 text-muted">{item.reason || "Faithfulness score fell below the grounded threshold."}</p>
                  <p className="mt-2 text-xs font-black text-accent">
                    Confidence: {pct(item.confidence)} | Faithfulness: {pct(item.faithfulness)}
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted">No failed query detail has been recorded yet.</p>
          )}
        </Modal>
      )}
      {liveOpen && (
        <Modal title="Live SSE Log Stream" onClose={() => setLiveOpen(false)}>
          <div className="max-h-80 space-y-2 overflow-y-auto">
            {logs.length === 0 && <p className="text-sm text-muted">Waiting for pipeline events...</p>}
            {logs.map((log, index) => (
              <div key={`${log.timestamp}-${log.step}-${index}`} className="flex items-center justify-between rounded-md border border-line bg-paper px-3 py-2 text-sm">
                <span className="font-black">{log.step}</span>
                <span className="text-xs text-muted">{log.timestamp} - {log.status}</span>
              </div>
            ))}
          </div>
        </Modal>
      )}
      <Footer />
    </>
  );
}

function MetricCard({ icon: Icon, label, value, note, history = [], dataKey }) {
  return (
    <div className="rounded-lg border border-line bg-paper p-8 shadow-soft">
      <Icon className="text-accent" size={22} />
      <p className="mt-6 text-xs font-black uppercase tracking-wide text-[#6F5A52]">{label}</p>
      <p className="mt-2 text-4xl font-black">{value}</p>
      <Sparkline data={history} dataKey={dataKey} />
      <div className="mt-8 border-t border-line pt-5 text-sm font-black text-accent">{note}</div>
    </div>
  );
}

function SmallMetric({ label, value, helper, icon: Icon, history = [], dataKey }) {
  return (
    <div className="rounded-lg border border-line bg-paper p-6">
      <p className="text-sm font-semibold text-[#6A4034]">{label}</p>
      <div className="mt-2 flex items-center gap-3">
        <p className="text-3xl font-black">{value}</p>
        <Icon className="text-accent" size={18} />
      </div>
      <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-[#E4DCD6]">
        <div className="h-full rounded-full bg-accent" style={{ width: value }} />
      </div>
      <Sparkline data={history} dataKey={dataKey} compact />
      <p className="mt-2 text-xs text-muted">{helper}</p>
    </div>
  );
}

function Sparkline({ data, dataKey, compact = false }) {
  return (
    <div className={compact ? "mt-3 h-10" : "mt-5 h-12"}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <Line type="monotone" dataKey={dataKey} stroke="#C84B2F" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function Modal({ title, children, onClose }) {
  return (
    <div className="fixed inset-0 z-40 grid place-items-center bg-black/25 p-4">
      <div className="w-full max-w-xl rounded-lg bg-[#FFFEFC] p-6 shadow-soft">
        <div className="mb-5 flex items-start justify-between gap-4">
          <h2 className="text-xl font-black">{title}</h2>
          <button type="button" onClick={onClose} className="icon-button" aria-label="Close modal">
            <X size={18} />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

function StatusRow({ label, value }) {
  return (
    <div className="flex items-center justify-between border-b border-line pb-3 text-sm">
      <span className="font-semibold text-muted">{label}</span>
      <span className="font-black capitalize">{value}</span>
    </div>
  );
}
