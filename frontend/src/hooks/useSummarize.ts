import { useState } from "react";
import type { SummaryPayload, SummaryResult } from "../types";

interface UseSummarizeResult {
  result: SummaryResult | null;
  error: string | null;
  isLoading: boolean;
  submit: (payload: SummaryPayload) => Promise<void>;
  reset: () => void;
}

export function useSummarize(endpoint = "/summarize"): UseSummarizeResult {
  const [result, setResult] = useState<SummaryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const submit = async (payload: SummaryPayload) => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(
          text || `摘要 API 回傳錯誤，HTTP ${response.status}`,
        );
      }

      const data = (await response.json()) as SummaryResult;
      setResult(data);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("未知錯誤，請稍後再試。");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setResult(null);
    setError(null);
    setIsLoading(false);
  };

  return { result, error, isLoading, submit, reset };
}
