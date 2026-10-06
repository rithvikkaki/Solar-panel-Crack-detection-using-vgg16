import { useState } from "react";
import { Info, Layers } from "lucide-react";
import type { ExplainabilityPayload } from "../../types/inspection";

interface GradCAMViewerProps {
  explainability: ExplainabilityPayload;
}

export const GradCAMViewer: React.FC<GradCAMViewerProps> = ({ explainability }) => {
  const [activeTab, setActiveTab] = useState<"overlay" | "heatmap" | "original">("overlay");
  const [blendOpacity, setBlendOpacity] = useState<number>(0.5);

  const { images, target_class_name, target_layer_name, disclaimer, original_dimensions } = explainability;

  return (
    <div className="bg-[#0B101D] border border-slate-800/90 rounded-2xl p-6 shadow-2xl shadow-black/60 space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-amber-400" />
          <div>
            <h4 className="text-sm font-bold text-white">Explainable AI (Grad-CAM Attribution)</h4>
            <p className="text-[11px] text-slate-400">
              Visual feature attribution map for condition: <span className="text-amber-400 font-semibold">{target_class_name}</span>
            </p>
          </div>
        </div>

        {/* View Segmented Toggle */}
        <div className="flex items-center gap-1 bg-slate-900/90 p-1 rounded-xl border border-slate-800 text-xs">
          <button
            onClick={() => setActiveTab("overlay")}
            className={`px-3 py-1 rounded-lg transition-all ${
              activeTab === "overlay" ? "bg-gradient-to-r from-amber-500 to-amber-600 text-slate-950 font-bold shadow-sm shadow-amber-500/20" : "text-slate-400 hover:text-white"
            }`}
          >
            AI Focus Overlay
          </button>
          <button
            onClick={() => setActiveTab("heatmap")}
            className={`px-3 py-1 rounded-lg transition-all ${
              activeTab === "heatmap" ? "bg-gradient-to-r from-amber-500 to-amber-600 text-slate-950 font-bold shadow-sm shadow-amber-500/20" : "text-slate-400 hover:text-white"
            }`}
          >
            Heatmap
          </button>
          <button
            onClick={() => setActiveTab("original")}
            className={`px-3 py-1 rounded-lg transition-all ${
              activeTab === "original" ? "bg-gradient-to-r from-amber-500 to-amber-600 text-slate-950 font-bold shadow-sm shadow-amber-500/20" : "text-slate-400 hover:text-white"
            }`}
          >
            Original
          </button>
        </div>
      </div>

      {/* Visual Display Container */}
      <div className="relative bg-black/60 rounded-xl overflow-hidden flex items-center justify-center border border-slate-800/80 min-h-[320px] max-h-[480px]">
        {activeTab === "overlay" && (
          <div className="relative w-full h-full flex items-center justify-center">
            {/* Dynamic Overlay Stack */}
            <img
              src={images.original}
              alt="Original Solar Panel"
              className="max-h-[460px] w-auto object-contain rounded-lg"
            />
            <img
              src={images.heatmap}
              alt="Grad-CAM Heatmap"
              style={{ opacity: 1.0 - blendOpacity }}
              className="absolute inset-0 max-h-[460px] w-auto h-full m-auto object-contain mix-blend-screen pointer-events-none"
            />
          </div>
        )}

        {activeTab === "heatmap" && (
          <img
            src={images.heatmap}
            alt="Grad-CAM Activation Heatmap"
            className="max-h-[460px] w-auto object-contain rounded-lg"
          />
        )}

        {activeTab === "original" && (
          <img
            src={images.original}
            alt="Original Solar Panel"
            className="max-h-[460px] w-auto object-contain rounded-lg"
          />
        )}

        {/* Telemetry Tag */}
        <div className="absolute bottom-2 right-2 bg-slate-950/80 backdrop-blur-sm border border-slate-800 px-2.5 py-1 rounded-md text-[10px] font-mono text-amber-300/80">
          Resolution: {original_dimensions[1]}×{original_dimensions[0]} | Target Layer: {target_layer_name}
        </div>
      </div>

      {/* Blend Slider (Visible on Overlay Tab) */}
      {activeTab === "overlay" && (
        <div className="flex items-center justify-between gap-4 px-2 text-xs text-slate-400">
          <span>Heatmap Intensity:</span>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={1.0 - blendOpacity}
            onChange={(e) => setBlendOpacity(1.0 - parseFloat(e.target.value))}
            className="w-48 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
          />
          <span className="font-mono text-amber-400 font-bold w-12 text-right">
            {Math.round((1.0 - blendOpacity) * 100)}%
          </span>
        </div>
      )}

      {/* Interpretability & Scientific Mandate Note */}
      <div className="flex items-start gap-2.5 p-3.5 bg-amber-950/20 border border-amber-500/25 rounded-xl text-xs text-amber-200/90 leading-relaxed">
        <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-bold text-amber-300">Interpretability Note: </span>
          {disclaimer}
        </div>
      </div>
    </div>
  );
};
