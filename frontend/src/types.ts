export type LengthUnit = "sentences" | "tokens";

export type Stage1BaseAlgorithm = "greedy" | "grasp" | "nsga2";
export type Stage1LLM = "bert" | "roberta" | "xlnet";
export type Stage2Method = "fast" | "fast_grasp" | "fast_nsga2" | "greedy";

export interface PresetConfig {
  id: string;
  label: string;
  description: string;
  lengthUnit: LengthUnit;
  maxSentences?: number;
  maxTokens?: number;
  candidateK: number;
  unionCap: number;
}

export interface FormState {
  documentsRaw: string;
  preset: string;
  stage1Base: Stage1BaseAlgorithm;
  stage1LLM: Stage1LLM | null;
  stage2: Stage2Method;
  lengthUnit: LengthUnit;
  maxSentences: number;
  maxTokens: number;
  candidateK: number;
  unionCap: number;
  includeReference: boolean;
  referenceRaw: string;
}

export interface SummaryPayload {
  documents: string[];
  stage1: {
    algorithms: Stage1BaseAlgorithm[];
    llms: Stage1LLM[];
    candidate_k: number;
  };
  stage2: {
    method: Stage2Method;
    union_cap: number;
  };
  length_control:
    | { unit: "sentences"; max_sentences: number }
    | { unit: "tokens"; max_tokens: number };
  reference?: string;
}

export interface SummaryResult {
  summary?: string;
  selected_indices?: number[];
  sentences?: string[];
  metrics?: Record<string, number | string>;
  timing?: Record<string, number>;
  [key: string]: unknown;
}
