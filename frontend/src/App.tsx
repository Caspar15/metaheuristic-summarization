import { useState } from "react";
import type { FormEvent, ReactElement } from "react";
import "./App.css";
import { DocumentInput } from "./components/DocumentInput";
import { Stage1Settings } from "./components/Stage1Settings";
import { Stage2Settings } from "./components/Stage2Settings";
import { LengthControl } from "./components/LengthControl";
import { SummaryResult } from "./components/SummaryResult";
import { PresetSelector } from "./components/PresetSelector";
import {
  DEFAULT_STAGE1_BASE,
  DEFAULT_STAGE1_LLM,
  DEFAULT_STAGE2,
  PRESET_OPTIONS,
} from "./utils/constants";
import { buildPayload, normalizeDocuments } from "./utils/payload";
import { useSummarize } from "./hooks/useSummarize";
import type {
  FormState,
  LengthUnit,
  Stage1BaseAlgorithm,
  Stage1LLM,
} from "./types";

const INITIAL_FORM: FormState = {
  documentsRaw: "",
  preset: PRESET_OPTIONS[0]?.id ?? "news-3sent",
  stage1Base: DEFAULT_STAGE1_BASE,
  stage1LLM: DEFAULT_STAGE1_LLM,
  stage2: DEFAULT_STAGE2,
  lengthUnit: PRESET_OPTIONS[0]?.lengthUnit ?? "sentences",
  maxSentences: PRESET_OPTIONS[0]?.maxSentences ?? 3,
  maxTokens: PRESET_OPTIONS[0]?.maxTokens ?? 400,
  candidateK: PRESET_OPTIONS[0]?.candidateK ?? 25,
  unionCap: PRESET_OPTIONS[0]?.unionCap ?? 15,
  includeReference: false,
  referenceRaw: "",
};

function App(): ReactElement {
  const [form, setForm] = useState<FormState>(INITIAL_FORM);
  const [formError, setFormError] = useState<string | null>(null);
  const apiEndpoint = import.meta.env.VITE_API_ENDPOINT ?? "/summarize";
  const { result, error, isLoading, submit, reset } = useSummarize(apiEndpoint);

  const updateForm = <K extends keyof FormState>(key: K, value: FormState[K]) =>
    setForm((prev) => ({ ...prev, [key]: value }));

  const handlePresetChange = (presetId: string) => {
    const preset = PRESET_OPTIONS.find((item) => item.id === presetId);
    if (!preset) {
      updateForm("preset", presetId);
      return;
    }
    setForm((prev) => ({
      ...prev,
      preset: presetId,
      lengthUnit: preset.lengthUnit,
      maxSentences:
        preset.lengthUnit === "sentences"
          ? preset.maxSentences ?? prev.maxSentences
          : prev.maxSentences,
      maxTokens:
        preset.lengthUnit === "tokens"
          ? preset.maxTokens ?? prev.maxTokens
          : prev.maxTokens,
      candidateK: preset.candidateK,
      unionCap: preset.unionCap,
    }));
  };

  const handleStage1BaseChange = (value: Stage1BaseAlgorithm) => {
    updateForm("stage1Base", value);
  };

  const handleStage1LLMChange = (value: Stage1LLM | null) => {
    updateForm("stage1LLM", value);
  };

  const handleUnitChange = (unit: LengthUnit) => {
    updateForm("lengthUnit", unit);
  };

  const handleSubmit = async (evt: FormEvent<HTMLFormElement>) => {
    evt.preventDefault();
    setFormError(null);

    const docs = normalizeDocuments(form.documentsRaw);
    if (!docs.length) {
      setFormError("Please provide at least one article. Use --- to split multiple articles.");
      return;
    }
    const payload = buildPayload(form);
    await submit(payload);
  };

  return (
    <div className="app-shell">
      <header className="app-shell__header">
        <h1>Metaheuristic Summarization Dashboard</h1>
        <p>
          Configure Stage1 algorithms/LLMs and Stage2 refinement. Supports both short (3-sentence) and 400-token summaries. Back-end `/summarize` API is required.
        </p>
      </header>

      <div className="app-shell__content">
        <form className="app-shell__form" onSubmit={handleSubmit}>
          <PresetSelector value={form.preset} onChange={handlePresetChange} />
          <DocumentInput
            documentsRaw={form.documentsRaw}
            onDocumentsChange={(value) => updateForm("documentsRaw", value)}
            includeReference={form.includeReference}
            onToggleReference={(checked) =>
              updateForm("includeReference", checked)
            }
            referenceRaw={form.referenceRaw}
            onReferenceChange={(value) => updateForm("referenceRaw", value)}
          />
          <Stage1Settings
            selectedBase={form.stage1Base}
            onChangeBase={handleStage1BaseChange}
            selectedLLM={form.stage1LLM}
            onChangeLLM={handleStage1LLMChange}
          />
          <Stage2Settings
            method={form.stage2}
            onMethodChange={(value) => updateForm("stage2", value)}
            candidateK={form.candidateK}
            onCandidateKChange={(value) => updateForm("candidateK", value)}
            unionCap={form.unionCap}
            onUnionCapChange={(value) => updateForm("unionCap", value)}
          />
          <LengthControl
            unit={form.lengthUnit}
            maxSentences={form.maxSentences}
            maxTokens={form.maxTokens}
            onUnitChange={handleUnitChange}
            onMaxSentencesChange={(value) => updateForm("maxSentences", value)}
            onMaxTokensChange={(value) => updateForm("maxTokens", value)}
          />

          <section className="panel panel--padded">
            <div className="status-bar">
              <button className="button" type="submit" disabled={isLoading}>
                {isLoading ? "Generating..." : "Generate Summary"}
              </button>
              {formError && <span className="error">{formError}</span>}
            </div>
          </section>
        </form>

        <aside className="app-shell__sidebar">
          <SummaryResult
            result={result}
            error={error}
            isLoading={isLoading}
            onReset={reset}
          />
        </aside>
      </div>
    </div>
  );
}

export default App;
