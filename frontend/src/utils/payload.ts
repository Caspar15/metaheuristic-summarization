import type { FormState, SummaryPayload } from "../types";

export function normalizeDocuments(raw: string): string[] {
  return raw
    .split(/\n-{3,}\n/)
    .map((segment) => segment.trim())
    .filter(Boolean);
}

export function buildPayload(form: FormState): SummaryPayload {
  const documents = normalizeDocuments(form.documentsRaw);

  const baseLength =
    form.lengthUnit === "sentences"
      ? {
          unit: "sentences" as const,
          max_sentences: Math.max(1, Math.floor(form.maxSentences)),
        }
      : {
          unit: "tokens" as const,
          max_tokens: Math.max(10, Math.floor(form.maxTokens)),
        };

  const payload: SummaryPayload = {
    documents,
    stage1: {
      algorithms: [form.stage1Base],
      llms: form.stage1LLM ? [form.stage1LLM] : [],
      candidate_k: Math.max(1, Math.floor(form.candidateK)),
    },
    stage2: {
      method: form.stage2,
      union_cap: Math.max(1, Math.floor(form.unionCap)),
    },
    length_control: baseLength,
  };

  if (form.includeReference && form.referenceRaw.trim()) {
    payload.reference = form.referenceRaw.trim();
  }

  return payload;
}
