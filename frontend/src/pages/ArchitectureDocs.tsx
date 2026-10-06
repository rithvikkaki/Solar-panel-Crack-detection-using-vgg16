import { BookOpen, Layers, AlertCircle } from "lucide-react";

export const ArchitectureDocs: React.FC = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white tracking-tight">
          System Architecture & Technical Foundations
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Detailed breakdown of transfer learning pipelines, explainability methods, and architectural remedies.
        </p>
      </div>

      {/* Critical Flaws Fixed Comparison */}
      <div className="bg-[#0B101D] border border-slate-800/90 rounded-2xl p-6 shadow-2xl shadow-black/60 space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <AlertCircle className="w-5 h-5 text-amber-400" />
          <span>Academic Notebook Defects Fixed for Production</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
          {/* Defect 1 */}
          <div className="p-4 bg-slate-900/90 rounded-xl border border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-xs font-semibold">
              <span className="text-rose-400">Original Flaw: Dense(90)</span>
              <span className="text-emerald-400">Fixed: Dynamic Dense(6, Softmax)</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              The original notebook declared <code className="text-rose-300">Dense(90)</code> without activation for a 6-class dataset. 84 classes were untrained logits risking runtime indexing crashes. Replaced with dynamic sizing matching detected dataset classes.
            </p>
          </div>

          {/* Defect 2 */}
          <div className="p-4 bg-slate-900/90 rounded-xl border border-slate-800 space-y-2">
            <div className="flex items-center justify-between text-xs font-semibold">
              <span className="text-rose-400">Original Flaw: Monolithic Notebook</span>
              <span className="text-emerald-400">Fixed: Modular Decoupled Pipeline</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              All Kaggle hardcoded paths were refactored into reproducible standalone training (<code className="text-amber-300">train.py</code>), evaluation (<code className="text-amber-300">evaluate.py</code>), and pure NumPy preprocessing.
            </p>
          </div>
        </div>
      </div>

      {/* Transfer Learning Pipeline */}
      <div className="bg-[#0B101D] border border-slate-800/90 rounded-2xl p-6 shadow-2xl shadow-black/60 space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <Layers className="w-5 h-5 text-amber-400" />
          <span>Two-Stage Transfer Learning Strategy</span>
        </h3>

        <div className="space-y-4 text-xs text-slate-300 leading-relaxed">
          <div className="p-4 bg-slate-900/90 rounded-xl border border-slate-800">
            <h4 className="font-semibold text-amber-300 text-sm mb-1">Stage 1: Frozen Feature Extraction</h4>
            <p className="text-slate-400">
              Backbone VGG16 weights pre-trained on ImageNet are completely frozen. The classification head (Global Average Pooling, Dropout 0.3, Dense Softmax) is trained with Adam (<code className="text-amber-400">lr=0.001</code>) and EarlyStopping.
            </p>
          </div>

          <div className="p-4 bg-slate-900/90 rounded-lg border border-slate-800">
            <h4 className="font-semibold text-cyan-300 text-sm mb-1">Stage 2: Fine-Tuning Block 5</h4>
            <p className="text-slate-400">
              Layers 0 through 13 remain frozen to protect low-level edge and texture primitives. Convolutional Block 5 (<code className="text-cyan-400">block5_conv1</code>, <code className="text-cyan-400">block5_conv2</code>, <code className="text-cyan-400">block5_conv3</code>) is unfrozen and trained with reduced learning rate (<code className="text-cyan-400">lr=0.0001</code>), ReduceLROnPlateau, and ModelCheckpoint saving best weights in modern <code className="text-cyan-400">.keras</code> format.
            </p>
          </div>
        </div>
      </div>

      {/* Explainable AI Details */}
      <div className="bg-[#0F172A] border border-slate-800 rounded-xl p-6 shadow-xl space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-cyan-400" />
          <span>Grad-CAM Mathematical Attribution</span>
        </h3>

        <p className="text-xs text-slate-300 leading-relaxed">
          Gradient-Weighted Class Activation Mapping computes the gradient of the predicted class score $y^c$ with respect to the feature map activations $A^k$ of the deepest convolutional layer (<code className="text-cyan-300">block5_conv3</code>):
        </p>

        <div className="p-4 bg-black/60 rounded-lg border border-slate-800 font-mono text-xs text-cyan-300 space-y-1">
          <div>alpha_k^c = (1 / Z) * sum_i sum_j (d y^c / d A_i,j^k)</div>
          <div>L_GradCAM^c = ReLU( sum_k alpha_k^c * A^k )</div>
        </div>

        <p className="text-xs text-slate-400 leading-relaxed">
          The resulting 2D activation map is min-max normalized, upscaled with bilinear interpolation to match the original resolution, and colorized with OpenCV's JET colormap for overlay rendering.
        </p>
      </div>
    </div>
  );
};
