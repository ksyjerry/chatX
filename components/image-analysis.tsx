"use client";

import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Image, Upload, Loader2, X } from "lucide-react";

const API_URL = "http://localhost:8000/api";

export function ImageAnalysis() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [prompt, setPrompt] = useState<string>(
    "이 이미지에 대해 자세히 설명해주세요."
  );

  const fileInputRef = useRef<HTMLInputElement>(null);

  // 파일 선택 핸들러
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;

    if (file) {
      setSelectedFile(file);

      // 이미지 미리보기 생성
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result as string);
      };
      reader.readAsDataURL(file);

      // 결과 초기화
      setResult(null);
      setError(null);
    }
  };

  // 파일 선택 버튼 클릭 핸들러
  const handleSelectClick = () => {
    fileInputRef.current?.click();
  };

  // 파일 제거 핸들러
  const handleRemoveFile = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // 이미지 분석 핸들러
  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError("이미지를 먼저 선택해주세요.");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // FormData 생성
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("prompt", prompt);
      formData.append("model", "gpt-4-vision-preview");
      formData.append("max_tokens", "1000");

      // API 요청
      const response = await fetch(`${API_URL}/upload-image`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("이미지 분석 중 오류가 발생했습니다.");
      }

      const data = await response.json();
      setResult(data.response);
    } catch (err) {
      console.error("이미지 분석 에러:", err);
      setError(
        err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다."
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto p-4 space-y-4">
      <div className="flex flex-col space-y-2">
        <h2 className="text-xl font-bold">이미지 분석</h2>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          이미지를 업로드하면 AI가 이미지를 분석하고 설명합니다.
        </p>
      </div>

      {/* 이미지 업로드 영역 */}
      <div className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg p-4 flex flex-col items-center justify-center">
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="image/*"
          className="hidden"
        />

        {previewUrl ? (
          <div className="relative w-full">
            <img
              src={previewUrl}
              alt="Preview"
              className="max-h-80 mx-auto rounded-lg object-contain shadow-md"
            />
            <button
              onClick={handleRemoveFile}
              className="absolute top-2 right-2 p-1 rounded-full bg-red-500 text-white hover:bg-red-700"
              aria-label="Remove image"
            >
              <X size={16} />
            </button>
          </div>
        ) : (
          <div
            onClick={handleSelectClick}
            className="flex flex-col items-center justify-center py-8 cursor-pointer w-full"
          >
            <Upload className="h-12 w-12 text-gray-400 mb-2" />
            <p className="text-sm text-gray-500 dark:text-gray-400">
              클릭하여 이미지 업로드
            </p>
            <p className="text-xs text-gray-400 mt-1">JPG, PNG, WEBP 지원</p>
          </div>
        )}
      </div>

      {/* 프롬프트 입력 필드 */}
      <div className="flex flex-col space-y-2">
        <label htmlFor="prompt" className="text-sm font-medium">
          프롬프트 (질문)
        </label>
        <Textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="이미지에 대해 물어볼 질문을 입력하세요"
          className="h-24 focus-visible:ring-orange-500 focus-visible:border-orange-500"
        />
      </div>

      {/* 분석 버튼 */}
      <Button
        onClick={handleAnalyze}
        disabled={!selectedFile || isLoading}
        className="w-full bg-orange-500 hover:bg-orange-600 text-white"
      >
        {isLoading ? (
          <>
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            분석 중...
          </>
        ) : (
          <>
            <Image className="h-4 w-4 mr-2" />
            이미지 분석하기
          </>
        )}
      </Button>

      {/* 에러 메시지 */}
      {error && (
        <div className="p-4 bg-red-50 dark:bg-red-900/20 text-red-500 dark:text-red-400 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* 결과 표시 */}
      {result && (
        <div className="p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
          <h3 className="font-semibold mb-2">분석 결과:</h3>
          <div className="text-gray-700 dark:text-gray-300 whitespace-pre-line">
            {result}
          </div>
        </div>
      )}
    </div>
  );
}
