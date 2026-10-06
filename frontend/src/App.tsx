import React, { useEffect, useState } from "react";
import { fetchHealth } from "./api/client";
import { Header } from "./components/common/Header";
import { ArchitectureDocs } from "./pages/ArchitectureDocs";
import { InspectionWorkspace } from "./pages/InspectionWorkspace";
import { SystemStatus } from "./pages/SystemStatus";
import { ModelPerformance } from "./pages/ModelPerformance";
import type { HealthResponse } from "./types/inspection";

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isLoadingHealth, setIsLoadingHealth] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<"workspace" | "performance" | "status" | "architecture">("workspace");
  const [isDemoMode, setIsDemoMode] = useState<boolean>(false);

  const checkSystemHealth = async () => {
    setIsLoadingHealth(true);
    try {
      const data = await fetchHealth();
      setHealth(data);
      // If model is not loaded, auto-enable demo mode for frictionless UI exploration
      if (!data.model_loaded) {
        setIsDemoMode(true);
      }
    } catch {
      // Backend may be starting or offline
      setHealth(null);
      setIsDemoMode(true);
    } finally {
      setIsLoadingHealth(false);
    }
  };

  useEffect(() => {
    checkSystemHealth();
    // Poll health status periodically
    const interval = setInterval(checkSystemHealth, 20000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#070A11] bg-[radial-gradient(ellipse_80%_60%_at_50%_-15%,rgba(245,158,11,0.12),rgba(0,0,0,0))] text-slate-100 flex flex-col selection:bg-amber-500 selection:text-slate-950 font-sans">
      {/* Top Header */}
      <Header
        health={health}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isDemoMode={isDemoMode}
        setIsDemoMode={setIsDemoMode}
      />

      {/* Dynamic Main View */}
      <main className="flex-1">
        {activeTab === "workspace" && (
          <InspectionWorkspace
            health={health}
            isDemoMode={isDemoMode}
            setIsDemoMode={setIsDemoMode}
          />
        )}
        {activeTab === "performance" && <ModelPerformance />}
        {activeTab === "status" && (
          <SystemStatus
            health={health}
            isLoading={isLoadingHealth}
            onRefresh={checkSystemHealth}
          />
        )}
        {activeTab === "architecture" && <ArchitectureDocs />}
      </main>

      {/* Global Application Footer */}
      <footer className="border-t border-slate-900 bg-[#06090F] py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-400">SolarSentinel AI</span>
            <span>•</span>
            <span>Explainable AI-Powered Solar Panel Health & Fault Intelligence Platform</span>
          </div>

          <div className="text-center sm:text-right text-[11px] text-slate-600">
            Interpretability Notice: Grad-CAM visualizes image regions that contributed strongly to model classification.
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
