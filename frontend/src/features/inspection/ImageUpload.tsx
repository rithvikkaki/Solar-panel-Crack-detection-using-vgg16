import React, { useRef, useState, useEffect } from "react";
import { UploadCloud, Image as ImageIcon, AlertCircle, FileCheck } from "lucide-react";
import { fetchSampleImages, fetchSampleImageBlob } from "../../api/client";
import type { SampleImageItem } from "../../types/inspection";

interface ImageUploadProps {
  onFileSelected: (file: File) => void;
  selectedFile: File | null;
  disabled: boolean;
  onClear: () => void;
}

export const ImageUpload: React.FC<ImageUploadProps> = ({
  onFileSelected,
  selectedFile,
  disabled,
  onClear
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleImageItem[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchSampleImages()
      .then((data) => {
        if (data && data.samples) {
          setSamples(data.samples);
        }
      })
      .catch(() => {
        // Fallback handled by synthetic fixtures
      });
  }, []);

  const validateAndSelect = (file: File) => {
    setErrorMsg(null);
    const validExtensions = ["jpg", "jpeg", "png", "webp"];
    const ext = file.name.split(".").pop()?.toLowerCase() || "";

    if (!validExtensions.includes(ext)) {
      setErrorMsg(`Unsupported file type (.${ext}). Please upload JPG, PNG, or WEBP.`);
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setErrorMsg(`File size (${(file.size / 1024 / 1024).toFixed(1)} MB) exceeds the 10 MB limit.`);
      return;
    }

    onFileSelected(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSelect(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleLoadRealSample = async (sample: SampleImageItem) => {
    try {
      const blob = await fetchSampleImageBlob(sample.id);
      const file = new File([blob], sample.filename, { type: "image/jpeg" });
      validateAndSelect(file);
    } catch {
      handleSyntheticFallback(sample.class_name);
    }
  };

  const handleSyntheticFallback = (className: string) => {
    const canvas = document.createElement("canvas");
    canvas.width = 244;
    canvas.height = 244;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#1E293B";
    ctx.fillRect(0, 0, 244, 244);
    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 1;
    for (let i = 0; i < 244; i += 40) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, 244);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(244, i);
      ctx.stroke();
    }

    if (className.includes("Physical")) {
      ctx.strokeStyle = "#EF4444";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(40, 60);
      ctx.lineTo(120, 110);
      ctx.lineTo(100, 180);
      ctx.lineTo(190, 210);
      ctx.stroke();
    } else if (className.includes("Dusty")) {
      ctx.fillStyle = "rgba(217, 119, 6, 0.4)";
      ctx.fillRect(20, 20, 204, 204);
    } else if (className.includes("Bird")) {
      ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
      ctx.beginPath();
      ctx.arc(120, 110, 30, 0, 2 * Math.PI);
      ctx.fill();
    } else if (className.includes("Snow")) {
      ctx.fillStyle = "rgba(241, 245, 249, 0.7)";
      ctx.fillRect(10, 10, 224, 224);
    } else if (className.includes("Electrical")) {
      ctx.fillStyle = "rgba(239, 68, 68, 0.6)";
      ctx.beginPath();
      ctx.arc(122, 122, 40, 0, 2 * Math.PI);
      ctx.fill();
    } else {
      ctx.fillStyle = "rgba(56, 189, 248, 0.15)";
      ctx.fillRect(0, 0, 244, 244);
    }

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `${className.toLowerCase().replace(/\s+/g, "_")}_sample.jpg`, {
          type: "image/jpeg"
        });
        validateAndSelect(file);
      }
    }, "image/jpeg", 0.9);
  };

  const defaultClasses = [
    { id: "clean", class_name: "Clean", filename: "clean_sample.jpg", relative_path: "", description: "Normal" },
    { id: "dusty", class_name: "Dusty", filename: "dusty_sample.jpg", relative_path: "", description: "Soiled" },
    { id: "bird-drop", class_name: "Bird-drop", filename: "bird_sample.jpg", relative_path: "", description: "Fouling" },
    { id: "electrical-damage", class_name: "Electrical-damage", filename: "elec_sample.jpg", relative_path: "", description: "Hotspot" },
    { id: "physical-damage", class_name: "Physical-Damage", filename: "phys_sample.jpg", relative_path: "", description: "Crack" },
    { id: "snow-covered", class_name: "Snow-Covered", filename: "snow_sample.jpg", relative_path: "", description: "Snow" }
  ];

  const displayList = samples.length > 0 ? samples : defaultClasses;

  return (
    <div className="space-y-4">
      {/* Upload Well */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => !selectedFile && fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all cursor-pointer ${
          isDragging
            ? "border-amber-400 bg-amber-500/10 scale-[1.01]"
            : selectedFile
            ? "border-emerald-500/50 bg-emerald-500/10 cursor-default"
            : "border-slate-800 hover:border-amber-500/50 bg-[#0C1220]/90 hover:bg-[#11192C]"
        } ${disabled ? "opacity-50 pointer-events-none" : ""}`}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={(e) => e.target.files?.[0] && validateAndSelect(e.target.files[0])}
          accept=".jpg,.jpeg,.png,.webp"
          className="hidden"
        />

        {selectedFile ? (
          <div className="flex flex-col items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center ring-1 ring-emerald-500/40">
              <FileCheck className="w-6 h-6" />
            </div>
            <div>
              <p className="text-sm font-semibold text-white">{selectedFile.name}</p>
              <p className="text-xs text-slate-400">
                {(selectedFile.size / 1024).toFixed(1)} KB • Image Validated & Ready
              </p>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onClear();
              }}
              className="mt-2 text-xs text-rose-400 hover:text-rose-300 underline font-medium"
            >
              Replace Image
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-amber-500/15 text-amber-400 flex items-center justify-center ring-1 ring-amber-500/30">
              <UploadCloud className="w-6 h-6" />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-200">
                Drag and drop solar panel image, or{" "}
                <span className="text-amber-400 underline font-semibold">browse files</span>
              </p>
              <p className="text-xs text-slate-500 mt-1">
                Supported formats: JPG, PNG, WEBP (Max 10 MB)
              </p>
            </div>
          </div>
        )}
      </div>

      {errorMsg && (
        <div className="flex items-center gap-2 p-3 text-xs bg-rose-500/10 text-rose-400 border border-rose-500/30 rounded-lg">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Quick Diagnostic Fixtures - All 6 Classes */}
      <div className="pt-2">
        <p className="text-xs font-semibold text-slate-400 mb-2 flex items-center gap-1.5">
          <ImageIcon className="w-3.5 h-3.5 text-amber-400" />
          Quick Test Sample (All 6 Defect Classes):
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {displayList.map((sample) => {
            const isClean = sample.class_name.toLowerCase().includes("clean");
            const isCrack = sample.class_name.toLowerCase().includes("physical");
            const isDust = sample.class_name.toLowerCase().includes("dusty");
            const isElec = sample.class_name.toLowerCase().includes("electrical");
            const isSnow = sample.class_name.toLowerCase().includes("snow");
            const isBird = sample.class_name.toLowerCase().includes("bird");

            let badgeStyle = "border-slate-800 bg-slate-900/80 text-slate-300 hover:border-amber-500/50";
            if (isClean) badgeStyle = "border-emerald-500/30 bg-emerald-950/30 text-emerald-300 hover:border-emerald-400/60 hover:bg-emerald-900/40";
            else if (isCrack) badgeStyle = "border-rose-500/30 bg-rose-950/30 text-rose-300 hover:border-rose-400/60 hover:bg-rose-900/40";
            else if (isDust) badgeStyle = "border-amber-500/30 bg-amber-950/30 text-amber-300 hover:border-amber-400/60 hover:bg-amber-900/40";
            else if (isElec) badgeStyle = "border-purple-500/30 bg-purple-950/30 text-purple-300 hover:border-purple-400/60 hover:bg-purple-900/40";
            else if (isSnow) badgeStyle = "border-cyan-500/30 bg-cyan-950/30 text-cyan-300 hover:border-cyan-400/60 hover:bg-cyan-900/40";
            else if (isBird) badgeStyle = "border-sky-500/30 bg-sky-950/30 text-sky-300 hover:border-sky-400/60 hover:bg-sky-900/40";

            return (
              <button
                key={sample.id}
                type="button"
                disabled={disabled}
                onClick={() => handleLoadRealSample(sample)}
                className={`px-3 py-2 text-xs font-medium border rounded-lg text-center transition-all truncate shadow-sm flex items-center justify-center gap-1.5 ${badgeStyle}`}
                title={`Load sample image for ${sample.class_name}`}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-current opacity-70" />
                <span className="truncate">{sample.class_name}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
};
