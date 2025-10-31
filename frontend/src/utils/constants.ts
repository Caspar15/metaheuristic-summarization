import type {
  LengthUnit,
  PresetConfig,
  Stage1BaseAlgorithm,
  Stage1LLM,
  Stage2Method,
} from "../types";

export const STAGE1_BASE_OPTIONS: Array<{
  id: Stage1BaseAlgorithm;
  label: string;
  description: string;
  estTime: string;
}> = [
  {
    id: "greedy",
    label: "Greedy",
    description: "MMR 權衡重要度與冗餘，效能最佳。",
    estTime: "≈ 0.002 s/文",
  },
  {
    id: "grasp",
    label: "GRASP",
    description: "隨機化建構 + 局部搜尋，品質與速度取得平衡。",
    estTime: "≈ 0.006 s/文",
  },
  {
    id: "nsga2",
    label: "NSGA-II",
    description: "多目標演化演算法，涵蓋度最佳但較耗時。",
    estTime: "≈ 0.17 s/文",
  },
];

export const STAGE1_LLM_OPTIONS: Array<{
  id: Stage1LLM;
  label: string;
  description: string;
  estTime: string;
}> = [
  {
    id: "bert",
    label: "BERT",
    description: "BERT-base-uncased 句向量排序。",
    estTime: "≈ 0.95 s/文",
  },
  {
    id: "roberta",
    label: "RoBERTa",
    description: "RoBERTa-base 句向量排序。",
    estTime: "≈ 0.63 s/文",
  },
  {
    id: "xlnet",
    label: "XLNet",
    description: "XLNet-base-cased 句向量排序。",
    estTime: "≈ 0.38 s/文",
  },
];

export const STAGE2_OPTIONS: Array<{
  id: Stage2Method;
  label: string;
  description: string;
}> = [
  {
    id: "fast",
    label: "Fast (TF-IDF fusion + MMR)",
    description: "預設建議：融合特徵與 TF-IDF 語意分數後再執行 MMR。",
  },
  {
    id: "fast_grasp",
    label: "Fast GRASP",
    description: "在 TF-IDF 融合分數上執行 GRASP 搜尋。",
  },
  {
    id: "fast_nsga2",
    label: "Fast NSGA-II",
    description: "以 TF-IDF 融合分數作為多目標演化的權重來源。",
  },
  {
    id: "greedy",
    label: "Greedy (MMR)",
    description: "只使用 TF-IDF 語意分數進行 MMR，適合資源有限環境。",
  },
];

export const PRESET_OPTIONS: PresetConfig[] = [
  {
    id: "news-3sent",
    label: "新聞三句摘要",
    description: "單篇新聞取三句摘要，對應 CNN/DailyMail 任務。",
    lengthUnit: "sentences",
    maxSentences: 3,
    candidateK: 25,
    unionCap: 15,
  },
  {
    id: "duc-400",
    label: "DUC 400 字摘要",
    description: "DUC2002 多文件 400 字任務，建議增加候選與聯集上限。",
    lengthUnit: "tokens",
    maxTokens: 400,
    candidateK: 50,
    unionCap: 40,
  },
];

export const DEFAULT_STAGE1_BASE: Stage1BaseAlgorithm = "greedy";
export const DEFAULT_STAGE1_LLM: Stage1LLM | null = null;
export const DEFAULT_STAGE2: Stage2Method = "fast";

export const LENGTH_UNITS: Array<{ id: LengthUnit; label: string }> = [
  { id: "sentences", label: "句數限制" },
  { id: "tokens", label: "字數限制" },
];

