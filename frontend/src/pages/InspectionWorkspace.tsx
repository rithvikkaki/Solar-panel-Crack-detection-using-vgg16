import React, { useState } from "react";
import { Play, Sparkles, AlertCircle, RefreshCw } from "lucide-react";
import { submitDemoInspection, submitInspection, ApiServiceError } from "../api/client";
import type { HealthResponse, InspectionResponse } from "../types/inspection";
import { ImageUpload } from "../features/inspection/ImageUpload";
import { InspectionProgress } from "../features/inspection/InspectionProgress";
import { PredictionCard } from "../features/inspection/PredictionCard";
import { GradCAMViewer } from "../features/inspection/GradCAMViewer";
import { ConfidenceDistribution } from "../features/inspection/ConfidenceDistribution";
import { InspectionSummary } from "../features/inspection/InspectionSummary";
import { ModelUnavailableNotice } from "../features/inspection/ModelUnavailableNotice";

interface InspectionWorkspaceProps {
  health: HealthResponse | null;
  isDemoMode: boolean;
  setIsDemoMode: (val: boolean) => void;
}

export const InspectionWorkspace: React.FC<InspectionWorkspaceProps> = ({
  health,
  isDemoMode,
  setIsDemoMode
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [inspectionResult, setInspectionResult] = useState<InspectionResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const isModelLoaded = health?.model_loaded ?? false;

  const handleRunInspection = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setErrorMessage(null);

    try {
      let result: InspectionResponse;
      if (isDemoMode || !isModelLoaded) {
        // If demo mode active or model not loaded, run demo preview
        result = await submitDemoInspection(selectedFile);
      } else {
        // Production Real-model inference
        result = await submitInspection(selectedFile);
      }
      setInspectionResult(result);
    } catch (err) {
      if (err instanceof ApiServiceError) {
        if (err.statusCode === 503) {
          setErrorMessage("Model weights are uninitialized. Inference cannot be run without trained weights.");
        } else {
          setErrorMessage(err.detail);
        }
      } else {
        setErrorMessage("Inspection failed. Please ensure the backend service is running.");
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setInspectionResult(null);
    setErrorMessage(null);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* If model weights missing and NOT in demo mode, show prominent honesty notice */}
      {!isModelLoaded && !isDemoMode && (
        <ModelUnavailableNotice onEnableDemoMode={() => setIsDemoMode(true)} />
      )}

      {/* Main Dual-Pane Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Pane: Ingestion & Telemetry Controls */}
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-[#0B101D] border border-slate-800/90 rounded-2xl p-6 shadow-2xl shadow-black/60 space-y-5">
            <div>
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-bold text-white tracking-tight">
                  Panel Imagery Ingestion
                </h2>
                <span className="text-[10px] font-bold uppercase tracking-wider text-amber-400/90 bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded-full">
                  VGG16 Backbone
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Upload raw visual surface photography to trigger deep feature extraction and explainability.
              </p>
            </div>

            <ImageUpload
              onFileSelected={(file) => {
                setSelectedFile(file);
                setInspectionResult(null);
                setErrorMessage(null);
              }}
              selectedFile={selectedFile}
              disabled={isProcessing}
              onClear={handleReset}
            />

            {errorMessage && (
              <div className="flex items-start gap-2 p-3 bg-rose-500/10 border border-rose-500/30 rounded-xl text-xs text-rose-400">
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                <span>{errorMessage}</span>
              </div>
            )}

            {/* Action Buttons */}
            <div className="pt-2 flex flex-col gap-2.5">
              <button
                disabled={!selectedFile || isProcessing}
                onClick={handleRunInspection}
                className={`w-full py-3.5 px-4 rounded-xl font-bold text-xs uppercase tracking-wider flex items-center justify-center gap-2.5 shadow-xl transition-all ${
                  !selectedFile || isProcessing
                    ? "bg-slate-850 text-slate-600 border border-slate-800 cursor-not-allowed"
                    : isDemoMode || !isModelLoaded
                    ? "bg-gradient-to-r from-amber-600 via-orange-600 to-amber-700 hover:from-amber-500 hover:to-orange-500 text-white shadow-amber-950/50"
                    : "bg-gradient-to-r from-amber-400 via-orange-500 to-amber-500 hover:from-amber-300 hover:to-orange-400 text-slate-950 shadow-amber-500/25 ring-1 ring-amber-400/50 active:scale-[0.99]"
                }`}
              >
                {isProcessing ? (
                  <RefreshCw className="w-4 h-4 animate-spin text-slate-950" />
                ) : isDemoMode || !isModelLoaded ? (
                  <Sparkles className="w-4 h-4" />
                ) : (
                  <Play className="w-4 h-4 fill-slate-950" />
                )}
                <span>
                  {isProcessing
                    ? "Running VGG16 Forward Pass..."
                    : isDemoMode || !isModelLoaded
                    ? "RUN DEMO PREVIEW INSPECTION"
                    : "RUN PRODUCTION AI INSPECTION"}
                </span>
              </button>

              {inspectionResult && (
                <button
                  type="button"
                  onClick={handleReset}
                  className="w-full py-2 text-xs text-slate-400 hover:text-amber-400 transition-colors text-center font-medium"
                >
                  Clear & Inspect New Panel
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Right Pane: AI Diagnostic Analytics & Grad-CAM View */}
        <div className="lg:col-span-7 space-y-6">
          {/* Active Diagnostic Progress */}
          {isProcessing && <InspectionProgress isProcessing={isProcessing} />}

          {/* Inspection Results Dashboard */}
          {inspectionResult && (
            <div className="space-y-6 animate-in fade-in duration-300">
              <PredictionCard inspection={inspectionResult} />
              <GradCAMViewer explainability={inspectionResult.explainability} />
              <ConfidenceDistribution
                probabilities={inspectionResult.class_probabilities}
                predictedCondition={inspectionResult.predicted_condition}
              />
              <InspectionSummary
                healthAssessment={inspectionResult.health_assessment}
                inspectionId={inspectionResult.inspection_id}
                timestamp={inspectionResult.timestamp}
              />
            </div>
          )}

          {/* Empty Placeholder State */}
          {!inspectionResult && !isProcessing && (
            <div className="bg-[#0F172A]/50 border border-slate-800/80 rounded-xl p-12 text-center flex flex-col items-center justify-center space-y-4 min-h-[420px]">
              <div className="w-16 h-16 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-600">
                <Play className="w-8 h-8 ml-1" />
              </div>
              <div className="max-w-sm space-y-1">
                <h3 className="text-base font-semibold text-slate-300">
                  Ready for Inspection Analysis
                </h3>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Select an image or one of the quick test fixtures on the left to initiate multi-stage visual classification and Grad-CAM spatial attribution.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
