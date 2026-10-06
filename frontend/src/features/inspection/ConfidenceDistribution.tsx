import React from "react";
import { BarChart3 } from "lucide-react";

interface ConfidenceDistributionProps {
  probabilities: Record<string, number>;
  predictedCondition: string;
}

export const ConfidenceDistribution: React.FC<ConfidenceDistributionProps> = ({
  probabilities,
  predictedCondition
}) => {
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

  return (
    <div className="bg-[#0B101D] border border-slate-800/90 rounded-2xl p-6 shadow-2xl shadow-black/60 space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-amber-400" />
          <h4 className="text-sm font-bold text-white">Full Probability Spectrum</h4>
        </div>
        <span className="text-xs text-slate-400">Softmax Distribution</span>
      </div>

      <div className="space-y-3.5">
        {entries.map(([className, prob]) => {
          const isWinner = className === predictedCondition;
          const percentage = (prob * 100).toFixed(1);

          return (
            <div key={className} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className={`font-medium ${isWinner ? "text-amber-300 font-bold flex items-center gap-1.5" : "text-slate-300"}`}>
                  {isWinner && <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />}
                  {className}
                </span>
                <span className={`font-mono ${isWinner ? "text-amber-300 font-bold" : "text-slate-400"}`}>{percentage}%</span>
              </div>
              <div className="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden border border-slate-800/80">
                <div
                  className={`h-2.5 rounded-full transition-all duration-500 ${
                    isWinner
                      ? "bg-gradient-to-r from-amber-500 to-orange-400 shadow-sm shadow-amber-500/50"
                      : "bg-slate-700/80"
                  }`}
                  style={{ width: `${Math.max(2, prob * 100)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <p className="text-[11px] text-slate-500 pt-2 border-t border-slate-800/60">
        Probabilities sum strictly to 100% across verified classification conditions.
      </p>
    </div>
  );
};
