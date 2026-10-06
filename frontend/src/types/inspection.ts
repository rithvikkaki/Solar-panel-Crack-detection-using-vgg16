/**
 * SolarSentinel AI - Frontend TypeScript Definitions
 * Strictly synchronized with backend API schemas (API_CONTRACT.md).
 */

export interface GradCAMImages {
  original: string; // Base64 PNG data URI
  heatmap: string;  // Base64 PNG data URI
  overlay: string;  // Base64 PNG data URI
}

export interface ExplainabilityPayload {
  target_class_index: number;
  target_class_name: string;
  target_layer_name: string;
  original_dimensions: [number, number];
  disclaimer: string;
  images: GradCAMImages;
}

export interface HealthAssessment {
  health_score: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL" | string;
  maintenance_priority: "NONE" | "ROUTINE" | "ELEVATED" | "SCHEDULED" | "URGENT" | "IMMEDIATE" | string;
  recommendation: string;
  methodology_note: string;
}

export interface ModelMetadata {
  architecture: string;
  input_dimensions: [number, number, number];
  is_demo_preview: boolean;
}

export interface InspectionResponse {
  inspection_id: string;
  timestamp: string;
  filename: string;
  predicted_condition: string;
  confidence: number;
  confidence_level?: "HIGH" | "MODERATE" | "LOW" | string;
  class_probabilities: Record<string, number>;
  all_classes: string[];
  health_assessment: HealthAssessment;
  explainability: ExplainabilityPayload;
  model_metadata: ModelMetadata;
}


export interface HealthResponse {
  service: string;
  status: string;
  model_loaded: boolean;
  model_mode: "REAL_MODEL" | "MODEL_UNAVAILABLE";
  version: string;
  verified_classes: string[];
  timestamp: string;
  message: string;
}

export interface DemoStatusResponse {
  mode: "REAL_MODEL" | "DEMO_ONLY";
  model_available: boolean;
  message: string;
  setup_instructions?: string[];
}

export interface ApiError {
  error_code: string;
  detail: string;
  timestamp: string;
}

export interface ClassMetric {
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

export interface ProductionModelIdentity {
  model_id: string;
  model_name: string;
  model_version?: string;
  model_path?: string;
  architecture: string;
  input_shape: [number, number, number];
  class_names?: string[];
  gradcam_target_layer: string;
  preprocessing_pipeline?: string;
  training_strategy?: string;
  held_out_test_samples?: number;
}

export interface ModelMetricsResponse {
  status: string;
  production_model?: ProductionModelIdentity;
  architecture: string;
  input_dimensions: [number, number, number];
  test_evaluation?: {
    total_test_samples?: number;
    overall?: {
      accuracy: number;
      precision_macro: number;
      recall_macro: number;
      f1_macro: number;
      precision_weighted: number;
      recall_weighted: number;
      f1_weighted: number;
    };
    per_class?: Record<string, ClassMetric>;
  };
  production_model_metrics?: any;
  performance_benchmark?: {
    hardware?: {
      cpu?: string;
      os?: string;
      python_version?: string;
      tensorflow_version?: string;
      iterations?: number;
      hardware_notice?: string;
    };
    model_loading_time_ms?: number;
    inference_latency_ms?: {
      mean: number;
      min: number;
      max: number;
      p50: number;
      p95: number;
    };
    gradcam_latency_ms?: {
      mean: number;
      min: number;
      max: number;
      p50: number;
      p95: number;
    };
    total_pipeline_latency_ms?: {
      mean: number;
      min: number;
      max: number;
      p50: number;
      p95: number;
    };
  };
  timestamp: string;
}

export interface SampleImageItem {
  id: string;
  class_name: string;
  filename: string;
  relative_path: string;
  description: string;
}

export interface SampleImagesResponse {
  count: number;
  samples: SampleImageItem[];
}

