import { AlertTriangle, Sparkles, Database } from "lucide-react";

interface ModelUnavailableNoticeProps {
  onEnableDemoMode: () => void;
}

export const ModelUnavailableNotice: React.FC<ModelUnavailableNoticeProps> = ({
  onEnableDemoMode
}) => {
  return (
    <div className="bg-gradient-to-br from-amber-500/10 via-[#0F172A] to-[#090D14] border border-amber-500/30 rounded-xl p-6 shadow-2xl space-y-5">
      <div className="flex items-start gap-4">
        <div className="w-10 h-10 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center shrink-0 border border-amber-500/30">
          <AlertTriangle className="w-5 h-5" />
        </div>
        <div className="space-y-1">
          <h3 className="text-base font-bold text-white tracking-tight">
            Model Weights Not Currently Available
          </h3>
          <p className="text-xs text-amber-200/80 leading-relaxed">
            Per strict scientific honesty guidelines, SolarSentinel AI will not simulate or fabricate neural network predictions when trained weights are uninitialized.
          </p>
        </div>
      </div>

      {/* Operational Setup Guide */}
      <div className="bg-black/50 border border-slate-800 rounded-lg p-4 space-y-2.5">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
          <Database className="w-4 h-4 text-cyan-400" />
          <span>How to activate production AI inference:</span>
        </div>
        <ol className="text-xs text-slate-400 space-y-1.5 pl-5 list-decimal font-mono">
          <li>Place class images in <span className="text-cyan-300">data/raw/</span> (e.g. Bird-drop, Clean, Dusty, Physical-Damage).</li>
          <li>Validate integrity: <span className="text-emerald-400">python ml/training/validate_dataset.py --data_dir data/raw</span></li>
          <li>Execute training: <span className="text-emerald-400">python ml/training/train.py --data_dir data/raw</span></li>
          <li>Restart backend to automatically load <span className="text-cyan-300">ml/models/solar_sentinel_vgg16.keras</span>.</li>
        </ol>
      </div>

      {/* Demo Visualization Option */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2 border-t border-slate-800/80">
        <p className="text-xs text-slate-400 text-center sm:text-left">
          Want to explore the UI workflow while awaiting training?
        </p>
        <button
          onClick={onEnableDemoMode}
          className="w-full sm:w-auto px-4 py-2 bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-colors"
        >
          <Sparkles className="w-4 h-4" />
          <span>Switch to Demo Visualization Mode</span>
        </button>
      </div>
    </div>
  );
};
