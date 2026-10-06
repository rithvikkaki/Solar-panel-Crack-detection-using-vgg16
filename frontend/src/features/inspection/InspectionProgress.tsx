import React, { useEffect, useState } from "react";
import { CheckCircle2, Loader2 } from "lucide-react";

interface InspectionProgressProps {
  isProcessing: boolean;
}

const STAGES = [
  "Validating file format and pixel decompression buffer",
  "Executing canonical VGG16 BGR zero-centering normalization",
  "Forward pass through deep convolutional backbone",
  "Computing GradientTape spatial activations (Grad-CAM XAI)",
  "Synthesizing rule-based health telemetry & maintenance plan"
];

export const InspectionProgress: React.FC<InspectionProgressProps> = ({ isProcessing }) => {
  const [currentStage, setCurrentStage] = useState(0);

  useEffect(() => {
    if (!isProcessing) {
      setCurrentStage(0);
      return;
    }

    const interval = setInterval(() => {
      setCurrentStage((prev) => (prev < STAGES.length - 1 ? prev + 1 : prev));
    }, 450);

    return () => clearInterval(interval);
  }, [isProcessing]);

  if (!isProcessing) return null;

  return (
    <div className="bg-[#0F172A] border border-cyan-500/30 rounded-xl p-5 shadow-xl shadow-cyan-950/30 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" />
          <h4 className="text-sm font-semibold text-white">AI Diagnostic Pipeline In Progress</h4>
        </div>
        <span className="text-xs font-mono text-cyan-400">
          Stage {currentStage + 1} of {STAGES.length}
        </span>
      </div>

      <div className="space-y-2 pt-2">
        {STAGES.map((desc, idx) => {
          const isDone = idx < currentStage;
          const isCurrent = idx === currentStage;

          return (
            <div key={desc} className="flex items-center gap-2.5 text-xs">
              {isDone ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
              ) : isCurrent ? (
                <Loader2 className="w-4 h-4 text-cyan-400 animate-spin shrink-0" />
              ) : (
                <div className="w-4 h-4 rounded-full border border-slate-700 shrink-0" />
              )}
              <span
                className={`${
                  isDone
                    ? "text-slate-400 line-through decoration-slate-600"
                    : isCurrent
                    ? "text-cyan-300 font-medium"
                    : "text-slate-600"
                }`}
              >
                {desc}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
