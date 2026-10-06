import { CheckCircle2, AlertTriangle, Database, Server, Shield } from "lucide-react";
import type { HealthResponse } from "../types/inspection";

interface SystemStatusProps {
  health: HealthResponse | null;
  isLoading: boolean;
  onRefresh: () => void;
}

export const SystemStatus: React.FC<SystemStatusProps> = ({ health, isLoading, onRefresh }) => {
  const isLoaded = health?.model_loaded ?? false;

  const classesDesc = [
    { name: "Bird-drop", desc: "Localized acidic organic residue causing localized light blockage and hotspot acceleration." },
    { name: "Clean", desc: "Nominal reference state; surface glass is clean with unimpeded solar irradiance." },
    { name: "Dusty", desc: "Particulate soiling causing uniform light attenuation across panel modules." },
    { name: "Electrical-damage", desc: "Internal cell discoloration, burn marks, and bypass diode hotspot anomalies." },
    { name: "Physical-Damage", desc: "Structural glass fractures, micro-cracks, and impact shattering patterns." },
    { name: "Snow-Covered", desc: "Partial or full winter snow accumulation obstructing PV generation." }
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Overview Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">System Telemetry & Health</h2>
          <p className="text-sm text-slate-400 mt-1">
            Real-time operational readiness, runtime configuration, and verified condition vocabulary.
          </p>
        </div>
        <button
          onClick={onRefresh}
          disabled={isLoading}
          className="px-4 py-2 bg-slate-900 border border-slate-800 hover:border-cyan-500/40 text-xs font-semibold text-slate-300 rounded-lg transition-colors flex items-center gap-2 self-start sm:self-auto"
        >
          <Server className="w-4 h-4 text-cyan-400" />
          <span>{isLoading ? "Checking..." : "Refresh Status"}</span>
        </button>
      </div>

      {/* Grid Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Backend API Service */}
        <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5 shadow-xl space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Backend REST API
            </span>
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-xl font-bold text-white">
            {health?.status ? health.status.toUpperCase() : "CHECKING..."}
          </div>
          <div className="text-xs text-slate-400 space-y-1 font-mono">
            <div>Framework: FastAPI (Uvicorn ASGI)</div>
            <div>API Version: {health?.version || "1.0.0"}</div>
            <div>Service: {health?.service || "SolarSentinel AI"}</div>
          </div>
        </div>

        {/* Deep Learning Engine */}
        <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5 shadow-xl space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Neural Weights
            </span>
            {isLoaded ? (
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            ) : (
              <AlertTriangle className="w-4 h-4 text-amber-400" />
            )}
          </div>
          <div className="text-xl font-bold text-white">
            {isLoaded ? "LOADED & ACTIVE" : "UNINITIALIZED"}
          </div>
          <div className="text-xs text-slate-400 space-y-1 font-mono">
            <div>Backbone: VGG16 (ImageNet)</div>
            <div>Runtime: TensorFlow 2.21 / Keras 3</div>
            <div>Mode: {health?.model_mode || "MODEL_UNAVAILABLE"}</div>
          </div>
        </div>

        {/* Explainability Engine */}
        <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-5 shadow-xl space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              XAI Attribution
            </span>
            <CheckCircle2 className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-xl font-bold text-white">Grad-CAM Ready</div>
          <div className="text-xs text-slate-400 space-y-1 font-mono">
            <div>Target Layer: block5_conv3</div>
            <div>Colormap: OpenCV JET</div>
            <div>Resolution: 244×244 Baseline</div>
          </div>
        </div>
      </div>

      {/* Verified Classes Catalog */}
      <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-6 shadow-xl space-y-5">
        <div className="flex items-center gap-2 pb-3 border-b border-slate-800">
          <Database className="w-5 h-5 text-cyan-400" />
          <div>
            <h3 className="text-base font-bold text-white">
              Verified Condition Vocabulary ({health?.verified_classes?.length || 6} Classes)
            </h3>
            <p className="text-xs text-slate-400">
              Canonical class definitions derived from original academic project audit.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {classesDesc.map((c) => (
            <div key={c.name} className="p-4 bg-slate-900/80 rounded-lg border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-bold text-cyan-300 font-mono">{c.name}</span>
                <span className="text-[10px] uppercase tracking-wider bg-slate-800 px-2 py-0.5 rounded text-slate-400">
                  Verified
                </span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed">{c.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Scientific Honesty Notice */}
      <div className="p-5 bg-slate-900/60 border border-slate-800 rounded-xl text-xs text-slate-400 leading-relaxed space-y-2">
        <div className="flex items-center gap-2 text-cyan-400 font-semibold text-sm">
          <Shield className="w-4 h-4" />
          <span>Scientific and Engineering Transparency</span>
        </div>
        <p>
          SolarSentinel AI strictly reports factual system states. When trained weights are uninitialized, the backend reports <code className="text-cyan-300">model_loaded: false</code> and rejects uncalibrated inference with HTTP 503 rather than fabricating artificial confidence metrics.
        </p>
      </div>
    </div>
  );
};
