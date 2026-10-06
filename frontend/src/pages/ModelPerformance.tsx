import React, { useEffect, useState } from "react";
import {
  BarChart3,
  Cpu,
  Layers,
  Zap,
  Activity,
  CheckCircle2,
  AlertTriangle,
  FileCheck,
  RefreshCw,
  ShieldCheck
} from "lucide-react";
import { fetchModelMetrics, ApiServiceError } from "../api/client";
import type { ModelMetricsResponse } from "../types/inspection";

export const ModelPerformance: React.FC = () => {
  const [metrics, setMetrics] = useState<ModelMetricsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchModelMetrics();
      setMetrics(data);
    } catch (err) {
      if (err instanceof ApiServiceError) {
        setError(err.detail);
      } else {
        setError("Failed to load model performance metrics.");
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMetrics();
  }, []);

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 flex flex-col items-center justify-center space-y-4">
        <RefreshCw className="w-8 h-8 text-cyan-400 animate-spin" />
        <p className="text-slate-400 text-sm">Loading verified model benchmarks & evaluation results...</p>
      </div>
    );
  }

  if (error || !metrics) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-xl p-6 text-rose-400 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
          <div>
            <h3 className="font-semibold text-base">Benchmark Telemetry Unavailable</h3>
            <p className="text-xs text-rose-300/80 mt-1">{error || "Could not retrieve metric files."}</p>
            <button
              onClick={loadMetrics}
              className="mt-4 px-3 py-1.5 bg-rose-500/20 hover:bg-rose-500/30 rounded text-xs font-semibold text-rose-300 transition-colors"
            >
              Retry Connection
            </button>
          </div>
        </div>
      </div>
    );
  }

  const overall = metrics.test_evaluation?.overall;
  const benchmark = metrics.performance_benchmark;
  const perClass = metrics.test_evaluation?.per_class;
  const prodModel = metrics.production_model;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-[#0C1222] via-[#0E1528] to-[#0A0E1A] border border-slate-800/90 rounded-2xl p-6 shadow-2xl relative overflow-hidden">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-amber-500/15 text-amber-400 border border-amber-500/30 rounded-md">
                Verified Production Model Registry
              </span>
              <span className="text-xs text-slate-400 font-mono">
                {prodModel?.model_id || "vgg16_baseline_production"}
              </span>
            </div>
            <h1 className="text-2xl font-bold text-white tracking-tight mt-2 flex items-center gap-2">
              <BarChart3 className="w-6 h-6 text-amber-400" />
              {prodModel?.model_name || "SolarSentinel VGG16 Transfer Learning"}
            </h1>
            <p className="text-xs text-slate-400 mt-1 max-w-2xl">
              Strict evaluation against the 131-sample hash-stratified held-out test split
              (<code className="text-amber-300">ml/metadata/split_manifest_70_15_15.json</code>).
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-3 text-right shadow-lg">
              <div className="text-[11px] text-slate-400">Untouched Test Samples</div>
              <div className="text-xl font-bold font-mono text-amber-400">
                {metrics.test_evaluation?.total_test_samples ?? "Not available"}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Production Model Identity Card */}
      <div className="bg-[#0B101D] border border-slate-800/90 rounded-2xl p-5 shadow-xl">
        <div className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-amber-400" />
          Production Model Specifications
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-xs font-mono">
          <div className="bg-slate-900/80 p-3 rounded border border-slate-800">
            <span className="text-[11px] text-slate-500 block">Production Model</span>
            <span className="text-slate-200 mt-0.5 font-semibold block truncate">
              {prodModel?.model_name || "VGG16 Baseline"}
            </span>
          </div>
          <div className="bg-slate-900/80 p-3 rounded border border-slate-800">
            <span className="text-[11px] text-slate-500 block">Architecture</span>
            <span className="text-cyan-300 mt-0.5 font-semibold block">
              {prodModel?.architecture || metrics.architecture || "VGG16"}
            </span>
          </div>
          <div className="bg-slate-900/80 p-3 rounded border border-slate-800">
            <span className="text-[11px] text-slate-500 block">Input Resolution</span>
            <span className="text-slate-200 mt-0.5 block">
              {prodModel?.input_shape ? `${prodModel.input_shape[0]}x${prodModel.input_shape[1]}x${prodModel.input_shape[2]}` : "244x244x3"}
            </span>
          </div>
          <div className="bg-slate-900/80 p-3 rounded border border-slate-800">
            <span className="text-[11px] text-slate-500 block">Test Set Size</span>
            <span className="text-emerald-400 mt-0.5 font-semibold block">
              {metrics.test_evaluation?.total_test_samples ?? 131} samples
            </span>
          </div>
          <div className="bg-slate-900/80 p-3 rounded border border-slate-800">
            <span className="text-[11px] text-slate-500 block">Grad-CAM Target Layer</span>
            <span className="text-indigo-300 mt-0.5 font-semibold block">
              {prodModel?.gradcam_target_layer || "block5_conv3"}
            </span>
          </div>
        </div>
      </div>

      {/* Top Level Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 font-medium">Test Set Accuracy</span>
            <FileCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-emerald-400 mt-2">
            {typeof overall?.accuracy === "number" ? `${(overall.accuracy * 100).toFixed(2)}%` : "Not available"}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Exact match across held-out test split</p>
        </div>

        <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 font-medium">Macro F1 Score</span>
            <Layers className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-cyan-400 mt-2">
            {typeof overall?.f1_macro === "number" ? `${(overall.f1_macro * 100).toFixed(2)}%` : "Not available"}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Unweighted mean across all 6 classes</p>
        </div>

        <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 font-medium">Weighted F1 Score</span>
            <Activity className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-indigo-400 mt-2">
            {typeof overall?.f1_weighted === "number" ? `${(overall.f1_weighted * 100).toFixed(2)}%` : "Not available"}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Support-weighted harmonic mean</p>
        </div>

        <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 font-medium">Mean Pipeline Latency</span>
            <Zap className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-amber-400 mt-2">
            {typeof benchmark?.total_pipeline_latency_ms?.mean === "number"
              ? `${benchmark.total_pipeline_latency_ms.mean.toFixed(1)} ms`
              : "Not available"}
          </div>
          <p className="text-[11px] text-slate-500 mt-1">Inference + Grad-CAM Heatmap Gen</p>
        </div>
      </div>

      {/* Per-Class Evaluation Table */}
      <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-6 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-bold text-white tracking-tight">
              Class-by-Class Evaluation Breakdown
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Individual precision, recall, and F1 performance on unseen test split.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500 bg-slate-900 px-2.5 py-1 rounded border border-slate-800">
            Split: 70/15/15 Stratified
          </span>
        </div>

        {perClass && Object.keys(perClass).length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                  <th className="py-3 px-4">Class Name</th>
                  <th className="py-3 px-4">Precision</th>
                  <th className="py-3 px-4">Recall</th>
                  <th className="py-3 px-4">F1 Score</th>
                  <th className="py-3 px-4">Test Support</th>
                  <th className="py-3 px-4 text-right">Reliability Rating</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {Object.entries(perClass).map(([clsName, stats]) => {
                  const f1 = stats.f1_score;
                  return (
                    <tr key={clsName} className="hover:bg-slate-900/50 transition-colors">
                      <td className="py-3.5 px-4 font-sans font-semibold text-white">
                        {clsName}
                      </td>
                      <td className="py-3.5 px-4 text-slate-300">
                        {(stats.precision * 100).toFixed(1)}%
                      </td>
                      <td className="py-3.5 px-4 text-slate-300">
                        {(stats.recall * 100).toFixed(1)}%
                      </td>
                      <td className="py-3.5 px-4">
                        <span className={`font-bold ${
                          f1 >= 0.85 ? "text-emerald-400" : f1 >= 0.70 ? "text-amber-400" : "text-rose-400"
                        }`}>
                          {(f1 * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="py-3.5 px-4 text-slate-400">
                        {stats.support}
                      </td>
                      <td className="py-3.5 px-4 text-right font-sans">
                        {f1 >= 0.85 ? (
                          <span className="inline-flex items-center gap-1 text-[11px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20 font-medium">
                            <CheckCircle2 className="w-3 h-3" /> Excellent
                          </span>
                        ) : f1 >= 0.70 ? (
                          <span className="inline-flex items-center gap-1 text-[11px] text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20 font-medium">
                            <AlertTriangle className="w-3 h-3" /> Moderate
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-[11px] text-rose-400 bg-rose-500/10 px-2 py-0.5 rounded border border-rose-500/20 font-medium">
                            <AlertTriangle className="w-3 h-3" /> Challenging
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-4 bg-slate-900 rounded text-center text-xs text-slate-400">
            Per-class test metrics are not available.
          </div>
        )}
      </div>

      {/* Latency & Hardware Benchmark Telemetry */}
      {benchmark && benchmark.total_pipeline_latency_ms ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Latency Breakdown */}
          <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-6 shadow-xl space-y-4">
            <div className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-amber-400" />
              <h3 className="text-base font-bold text-white tracking-tight">
                Latency Profile ({benchmark.hardware?.iterations ?? 30} Iterations)
              </h3>
            </div>

            <div className="space-y-4 pt-2">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">Model Inference (Forward Pass)</span>
                  <span className="font-mono text-cyan-300">
                    Mean: {benchmark.inference_latency_ms?.mean?.toFixed(1) ?? "N/A"}ms | p95: {benchmark.inference_latency_ms?.p95?.toFixed(1) ?? "N/A"}ms
                  </span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-cyan-500 h-2 rounded-full"
                    style={{
                      width: `${
                        benchmark.inference_latency_ms && benchmark.total_pipeline_latency_ms
                          ? (benchmark.inference_latency_ms.mean / benchmark.total_pipeline_latency_ms.mean) * 100
                          : 30
                      }%`
                    }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">Grad-CAM Attribution (Gradient Backprop)</span>
                  <span className="font-mono text-indigo-300">
                    Mean: {benchmark.gradcam_latency_ms?.mean?.toFixed(1) ?? "N/A"}ms | p95: {benchmark.gradcam_latency_ms?.p95?.toFixed(1) ?? "N/A"}ms
                  </span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-indigo-500 h-2 rounded-full"
                    style={{
                      width: `${
                        benchmark.gradcam_latency_ms && benchmark.total_pipeline_latency_ms
                          ? (benchmark.gradcam_latency_ms.mean / benchmark.total_pipeline_latency_ms.mean) * 100
                          : 70
                      }%`
                    }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2 pt-3 border-t border-slate-800/80 text-center font-mono">
                <div className="bg-slate-900/80 p-2.5 rounded border border-slate-800">
                  <div className="text-[10px] text-slate-500">Min Latency</div>
                  <div className="text-xs font-bold text-slate-300 mt-1">
                    {benchmark.total_pipeline_latency_ms.min.toFixed(1)} ms
                  </div>
                </div>
                <div className="bg-slate-900/80 p-2.5 rounded border border-slate-800">
                  <div className="text-[10px] text-slate-500">Median (p50)</div>
                  <div className="text-xs font-bold text-slate-300 mt-1">
                    {benchmark.total_pipeline_latency_ms.p50.toFixed(1)} ms
                  </div>
                </div>
                <div className="bg-slate-900/80 p-2.5 rounded border border-slate-800">
                  <div className="text-[10px] text-slate-500">95th Percentile</div>
                  <div className="text-xs font-bold text-amber-400 mt-1">
                    {benchmark.total_pipeline_latency_ms.p95.toFixed(1)} ms
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Hardware & Scientific Transparency */}
          <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-6 shadow-xl space-y-4">
            <div className="flex items-center gap-2">
              <Cpu className="w-5 h-5 text-indigo-400" />
              <h3 className="text-base font-bold text-white tracking-tight">
                Hardware Environment & Audit Trail
              </h3>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs font-mono pt-1">
              <div className="bg-slate-900/80 p-3 rounded border border-slate-800">
                <span className="text-[11px] text-slate-500 block">Host Platform</span>
                <span className="text-slate-200 mt-0.5 block truncate" title={benchmark.hardware?.os || "Windows"}>
                  {benchmark.hardware?.os || "Windows"}
                </span>
              </div>
              <div className="bg-slate-900/80 p-3 rounded border border-slate-800">
                <span className="text-[11px] text-slate-500 block">Processor</span>
                <span className="text-slate-200 mt-0.5 block truncate" title={benchmark.hardware?.cpu || "Host CPU"}>
                  {benchmark.hardware?.cpu || "Host CPU"}
                </span>
              </div>
              <div className="bg-slate-900/80 p-3 rounded border border-slate-800">
                <span className="text-[11px] text-slate-500 block">Python Runtime</span>
                <span className="text-slate-200 mt-0.5 block">
                  Python {benchmark.hardware?.python_version || "3.10"}
                </span>
              </div>
              <div className="bg-slate-900/80 p-3 rounded border border-slate-800">
                <span className="text-[11px] text-slate-500 block">TensorFlow / Keras</span>
                <span className="text-slate-200 mt-0.5 block">
                  TF {benchmark.hardware?.tensorflow_version || "2.21"}
                </span>
              </div>
            </div>

            {/* Scientific Disclaimer Card */}
            <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-3 text-xs text-slate-400 flex items-start gap-2.5">
              <ShieldCheck className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
              <div className="text-[11px] leading-relaxed">
                <span className="font-semibold text-slate-300">Scientific Integrity Notice: </span>
                Metrics are loaded dynamically from <code className="text-cyan-300">GET /api/v1/model/metrics</code> backed by <code className="text-cyan-300">ml/metadata/production_model_metrics.json</code>. Latency measurements are hardware-dependent and executed on host CPU.
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};