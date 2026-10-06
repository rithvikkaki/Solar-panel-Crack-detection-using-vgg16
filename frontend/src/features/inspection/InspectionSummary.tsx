import { ClipboardList, Clock, Wrench } from "lucide-react";
import type { HealthAssessment } from "../../types/inspection";

interface InspectionSummaryProps {
  healthAssessment: HealthAssessment;
  inspectionId: string;
  timestamp: string;
}

export const InspectionSummary: React.FC<InspectionSummaryProps> = ({
  healthAssessment,
  inspectionId,
  timestamp
}) => {
  const getPriorityStyle = (priority: string) => {
    switch (priority) {
      case "IMMEDIATE":
      case "URGENT":
        return "bg-rose-500/10 text-rose-400 border-rose-500/30";
      case "SCHEDULED":
      case "ELEVATED":
        return "bg-amber-500/10 text-amber-400 border-amber-500/30";
      case "ROUTINE":
      case "NONE":
      default:
        return "bg-emerald-500/10 text-emerald-400 border-emerald-500/30";
    }
  };

  return (
    <div className="bg-[#0B101D] border border-slate-800/90 rounded-2xl p-6 shadow-2xl shadow-black/60 space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <Wrench className="w-4 h-4 text-amber-400" />
          <h4 className="text-sm font-bold text-white">Maintenance Intelligence & Action Plan</h4>
        </div>
        <span className={`px-2.5 py-0.5 rounded-lg text-xs font-bold border ${getPriorityStyle(healthAssessment.maintenance_priority)}`}>
          Priority: {healthAssessment.maintenance_priority}
        </span>
      </div>

      <div className="bg-slate-900/90 rounded-xl p-4 border border-slate-800/80">
        <p className="text-xs font-bold text-amber-400/90 uppercase tracking-wider mb-1">
          Prescriptive Recommendation
        </p>
        <p className="text-sm text-slate-200 leading-relaxed font-normal">
          {healthAssessment.recommendation}
        </p>
      </div>

      <div className="p-3 bg-slate-900/50 border border-slate-800/80 rounded-lg text-xs text-slate-400 leading-relaxed">
        <span className="font-semibold text-slate-300">Methodology Note: </span>
        {healthAssessment.methodology_note}
      </div>

      {/* Metadata Audit Footer */}
      <div className="pt-2 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 text-[11px] text-slate-500 border-t border-slate-800/60 font-mono">
        <span className="flex items-center gap-1.5">
          <ClipboardList className="w-3.5 h-3.5" />
          ID: {inspectionId}
        </span>
        <span className="flex items-center gap-1.5">
          <Clock className="w-3.5 h-3.5" />
          Timestamp: {new Date(timestamp).toLocaleString()}
        </span>
      </div>
    </div>
  );
};
