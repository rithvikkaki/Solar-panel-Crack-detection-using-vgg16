import { Activity, Cpu, Sun } from "lucide-react";
import type { HealthResponse } from "../../types/inspection";

interface HeaderProps {
  health: HealthResponse | null;
  activeTab: "workspace" | "status" | "performance" | "architecture";
  setActiveTab: (tab: "workspace" | "status" | "performance" | "architecture") => void;
  isDemoMode: boolean;
  setIsDemoMode: (val: boolean) => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  activeTab,
  setActiveTab,
  isDemoMode,
  setIsDemoMode
}) => {
  const isLoaded = health?.model_loaded ?? false;

  return (
    <header className="border-b border-slate-800 bg-[#090D14]/90 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-400 via-orange-500 to-amber-600 flex items-center justify-center text-slate-950 shadow-lg shadow-amber-500/25 ring-1 ring-amber-400/40">
            <Sun className="w-6 h-6 text-slate-950 stroke-[2.5]" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg tracking-tight text-white">SolarSentinel</span>
              <span className="px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-amber-500/15 text-amber-400 border border-amber-500/30 rounded-md">
                VGG16 AI
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Photovoltaic Health & Explainable Defect Attribution
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => setActiveTab("workspace")}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              activeTab === "workspace"
                ? "bg-gradient-to-r from-amber-500 to-amber-600 text-slate-950 shadow-md shadow-amber-500/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
            }`}
          >
            Inspection Workspace
          </button>
          <button
            onClick={() => setActiveTab("performance")}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              activeTab === "performance"
                ? "bg-gradient-to-r from-amber-500 to-amber-600 text-slate-950 shadow-md shadow-amber-500/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
            }`}
          >
            Model Performance
          </button>
          <button
            onClick={() => setActiveTab("status")}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              activeTab === "status"
                ? "bg-gradient-to-r from-amber-500 to-amber-600 text-slate-950 shadow-md shadow-amber-500/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
            }`}
          >
            System Health
          </button>
          <button
            onClick={() => setActiveTab("architecture")}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              activeTab === "architecture"
                ? "bg-gradient-to-r from-amber-500 to-amber-600 text-slate-950 shadow-md shadow-amber-500/20"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
            }`}
          >
            Architecture
          </button>
        </nav>

        {/* Runtime Status Pill */}
        <div className="flex items-center gap-3">
          {/* Mode Switcher */}
          <div className="hidden md:flex items-center gap-2 bg-slate-900 border border-slate-800 px-3 py-1 rounded-full text-xs">
            <span className="text-slate-400">Mode:</span>
            <button
              onClick={() => setIsDemoMode(!isDemoMode)}
              className={`font-semibold flex items-center gap-1.5 ${
                isDemoMode ? "text-amber-400" : "text-emerald-400"
              }`}
              title="Toggle between Real Inference and Demo Mode"
            >
              <span className={`w-2 h-2 rounded-full ${isDemoMode ? "bg-amber-400 animate-pulse" : "bg-emerald-400"}`} />
              {isDemoMode ? "DEMO VISUALIZATION" : "PRODUCTION AI"}
            </button>
          </div>

          {/* Model Status Indicator */}
          <div
            className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border ${
              isLoaded
                ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                : "bg-amber-500/10 text-amber-400 border-amber-500/30"
            }`}
          >
            {isLoaded ? (
              <>
                <Cpu className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Model:</span>
                <span className="font-semibold">ACTIVE</span>
              </>
            ) : (
              <>
                <Activity className="w-3.5 h-3.5 animate-pulse" />
                <span className="hidden sm:inline">Weights:</span>
                <span className="font-semibold">UNINITIALIZED</span>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
