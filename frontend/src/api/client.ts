/**
 * SolarSentinel AI - Type-Safe API Client
 */

import type {
  DemoStatusResponse,
  HealthResponse,
  InspectionResponse
} from "../types/inspection";

// Configurable API base URL: defaults to relative /api/v1 for Vite proxy,
// or can be overridden via VITE_API_URL / VITE_API_BASE_URL (e.g. http://127.0.0.1:8000 or https://<space>.hf.space)
const rawEnvUrl = (
  (import.meta.env.VITE_API_URL as string | undefined) ||
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ||
  ""
).trim().replace(/\/$/, "");

export const API_BASE = rawEnvUrl
  ? (rawEnvUrl.endsWith("/api/v1") ? rawEnvUrl : `${rawEnvUrl}/api/v1`)
  : "/api/v1";

export class ApiServiceError extends Error {

  statusCode: number;
  detail: string;

  constructor(statusCode: number, detail: string) {
    super(detail);
    this.name = "ApiServiceError";
    this.statusCode = statusCode;
    this.detail = detail;
  }
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: "Health check failed" }));
    throw new ApiServiceError(res.status, errorBody.detail || "Health check failed");
  }
  return res.json();
}

export async function fetchDemoStatus(): Promise<DemoStatusResponse> {
  const res = await fetch(`${API_BASE}/demo/status`);
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: "Demo status fetch failed" }));
    throw new ApiServiceError(res.status, errorBody.detail || "Failed to query demo status");
  }
  return res.json();
}

export async function submitInspection(file: File): Promise<InspectionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/inspect`, {
    method: "POST",
    body: formData
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiServiceError(res.status, errorBody.detail || "Inspection request failed");
  }

  return res.json();
}

export async function submitDemoInspection(file: File): Promise<InspectionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/demo/inspect`, {
    method: "POST",
    body: formData
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiServiceError(res.status, errorBody.detail || "Demo inspection request failed");
  }

  return res.json();
}

export async function fetchModelMetrics(): Promise<any> {
  const res = await fetch(`${API_BASE}/model/metrics`);
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: "Failed to fetch model metrics" }));
    throw new ApiServiceError(res.status, errorBody.detail || "Metrics request failed");
  }
  return res.json();
}

export async function fetchSampleImages(): Promise<any> {
  const res = await fetch(`${API_BASE}/demo/sample-images`);
  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: "Failed to fetch sample images" }));
    throw new ApiServiceError(res.status, errorBody.detail || "Sample images request failed");
  }
  return res.json();
}

export async function fetchSampleImageBlob(sampleId: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/demo/sample-images/${sampleId}`);
  if (!res.ok) {
    throw new ApiServiceError(res.status, `Failed to download sample image: ${res.statusText}`);
  }
  return res.blob();
}

