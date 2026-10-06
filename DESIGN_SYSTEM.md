# SolarSentinel AI - Product Design System & UI/UX Specification

**Product Name:** SolarSentinel AI  
**Tagline:** Intelligent Visual Inspection & Condition Analysis for Solar Panels  
**Product Positioning:** Enterprise & Utility-Grade AI Platform for Photovoltaic (PV) Health Intelligence  
**Role:** Senior Product Designer (UI/UX Agent)  
**Status:** Approved Design System  
**Date:** September 11, 2026  

---

## 1. Visual Philosophy & Design Principles

1. **Industrial & Scientific Aesthetic:**
   * Styled like high-precision mission-critical monitoring infrastructure (inspired by aerospace telemetry and high-end semiconductor inspection tools).
   * No generic consumer gradients, no chaotic rainbow colors, no bloated toy cards, no fake terminal ASCII art.
2. **High Information Density & Purposeful Spacing:**
   * Tight, structured layouts with clear visual hierarchy, monospace data readouts for telemetry, and crisp border separators.
3. **Restrained Color Architecture:**
   * Base canvas: Deep obsidian and graphite dark neutrals (`#090D14`, `#0F172A`, `#1E293B`).
   * Primary Action / Brand Accent: Precision Cyan / Electric Blue (`#38BDF8`, `#0284C7`).
   * Semantic Status Colors (Used strictly for condition risk states):
     * **Optimal / Clean:** Emerald Green (`#10B981`)
     * **Warning / Dusty / Snow:** Warm Amber (`#F59E0B`)
     * **Elevated / Bird-Drop:** Tangerine Orange (`#FB923C`)
     * **Critical / Damage / Cracks:** Coral Crimson (`#EF4444`)
4. **Honest Explainability Presentation:**
   * Grad-CAM visualizations must always be paired with the original reference image and clear tab controls (Original Image, Activation Heatmap, AI Focus Overlay).
   * Prominently display interpretability disclaimers.

---

## 2. Color Palette & Design Tokens

### Surface & Background Tokens
```css
--bg-canvas: #090D14;       /* Deepest obsidian background */
--bg-surface: #0F172A;      /* Primary container card background */
--bg-surface-elevated: #1E293B; /* Modals, dropdowns, active tabs */
--bg-surface-subtle: #141E33;   /* Secondary input fields & preview wells */
--border-subtle: rgba(255, 255, 255, 0.08); /* Minimal 1px card borders */
--border-highlight: rgba(56, 189, 248, 0.3); /* Focused state cyan border */
```

### Typography Tokens
```css
--text-primary: #F8FAFC;    /* High-contrast crisp headlines & values */
--text-secondary: #94A3B8;  /* Subtitles, secondary descriptions, units */
--text-muted: #64748B;      /* Disclaimers, footer notes, disabled states */
--text-accent: #38BDF8;     /* Active cyan telemetry values */
```

### Condition & Risk Semantic Tokens
| Condition / Level | Badge Background | Text Color | Border Color |
|---|---|---|---|
| `Clean` / `LOW` | `rgba(16, 185, 129, 0.12)` | `#34D399` | `rgba(16, 185, 129, 0.3)` |
| `Dusty` / `LOW` | `rgba(245, 158, 11, 0.12)` | `#FBBF24` | `rgba(245, 158, 11, 0.3)` |
| `Bird-drop` / `MEDIUM` | `rgba(251, 146, 60, 0.12)` | `#FB923C` | `rgba(251, 146, 60, 0.3)` |
| `Snow-Covered` / `MEDIUM` | `rgba(56, 189, 248, 0.12)` | `#38BDF8` | `rgba(56, 189, 248, 0.3)` |
| `Electrical-damage` / `HIGH` | `rgba(244, 63, 94, 0.12)` | `#FB7185` | `rgba(244, 63, 94, 0.3)` |
| `Physical-Damage` / `CRITICAL` | `rgba(239, 68, 68, 0.15)` | `#F87171` | `rgba(239, 68, 68, 0.4)` |

---

## 3. Core Page Specifications

### 1. Landing Page (`Home.tsx`)
* **Hero Section:**
  * Clean, technical product title: `SolarSentinel AI`.
  * Subtitle: *"Explainable AI-Powered Solar Panel Health & Fault Intelligence Platform"*.
  * System Badge: Live status indicator connected to `/api/v1/health` showing `ONLINE (REAL_MODEL)` or `ONLINE (MODEL_UNAVAILABLE)`.
  * Dual Action CTAs: `OPEN INSPECTION WORKSPACE` (primary) and `SYSTEM ARCHITECTURE & STATUS` (secondary).
* **Core Capabilities Grid:**
  * VGG16 Deep Transfer Learning (6 Verified PV Degradation Classes)
  * Gradient-Weighted Class Activation Mapping (Grad-CAM XAI)
  * Rule-Based Maintenance Health Assessment Engine
  * Full-Stack Production Telemetry & Diagnostics

### 2. Inspection Workspace (`InspectionWorkspace.tsx`)
* **Primary Layout:** Dual-pane layout on desktop (Upload & Controls on Left; Live AI Analytics & Grad-CAM Viewer on Right).
* **Step 1: Drag-and-Drop Ingestion:**
  * Accepts `.jpg`, `.jpeg`, `.png`, `.webp` up to 10MB.
  * Instant local client preview before submission.
  * Quick sample loader: provides 1-click test images for sample conditions.
* **Step 2: Processing State:**
  * Animated multi-stage diagnostic progress:
    1. Validating file format & decompressed pixel buffer
    2. Zero-centering BGR channels (VGG16 normalization)
    3. Forward pass through neural network backbone
    4. Gradient backpropagation & spatial activation pooling
    5. Rule-based maintenance prioritization
* **Step 3: Prediction & Health Header:**
  * Condition badge with semantic risk styling.
  * Numerical confidence meter with progress bar.
  * Health Score Gauge (e.g. `82 / 100`) accompanied by rule-based separation notice.
* **Step 4: Interactive Grad-CAM Attribution Viewer:**
  * Segmented 3-way toggle:
    * `AI Focus Overlay` (Default view)
    * `Activation Heatmap` (Pure JET colormap)
    * `Original Image` (Unmodified baseline)
  * Opacity slider ($0\% - 100\%$) for dynamic overlay blend adjustment.
  * Prominent scientific attribution disclaimer.
* **Step 5: Probability Spectrum:**
  * Horizontal bar chart displaying predicted probabilities across all 6 verified classes.
* **Step 6: Maintenance Action Plan:**
  * Prescriptive maintenance recommendation card with priority flag (`ROUTINE`, `ELEVATED`, `URGENT`, `IMMEDIATE`).

### 3. System Status & Architecture Page (`SystemStatus.tsx`)
* **Hardware & Runtime Telemetry:**
  * TensorFlow runtime version, Keras backend, Python environment.
  * Model weight status (`Loaded` vs `Not Loaded`).
  * Input dimensions: $244 \times 244 \times 3$.
* **Verified Class Reference:**
  * Catalog of all 6 verified dataset conditions and operational descriptions.
* **Model Training Guide:**
  * Step-by-step instructions for placing datasets in `data/raw/` and executing two-stage transfer learning.
