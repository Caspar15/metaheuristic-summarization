import "./Panel.css";
import { STAGE2_OPTIONS } from "../utils/constants";
import type { Stage2Method } from "../types";

interface Stage2SettingsProps {
  method: Stage2Method;
  onMethodChange: (next: Stage2Method) => void;
  candidateK: number;
  onCandidateKChange: (value: number) => void;
  unionCap: number;
  onUnionCapChange: (value: number) => void;
}

export function Stage2Settings({
  method,
  onMethodChange,
  candidateK,
  onCandidateKChange,
  unionCap,
  onUnionCapChange,
}: Stage2SettingsProps) {
  return (
    <section className="panel">
      <header>
        <h2>Stage 2 · 精煉摘要</h2>
        <p className="panel__subtitle">
          Stage2 將 Stage1 產生的候選句聯集後，再以 TF-IDF 語意與 MMR
          重新挑出最終摘要。
        </p>
      </header>

      <fieldset>
        <legend>精煉方法</legend>
        <div className="chips">
          {STAGE2_OPTIONS.map((option) => (
            <label
              key={option.id}
              className={`badge ${method === option.id ? "badge--active" : ""}`}
            >
              <input
                type="radio"
                name="stage2-method"
                value={option.id}
                checked={method === option.id}
                onChange={() => onMethodChange(option.id)}
              />
              <span className="badge__text">
                <span className="badge__label">{option.label}</span>
                <span className="badge__hint">{option.description}</span>
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      <div className="numbers">
        <label className="number-input">
          Stage1 候選 K
          <input
            type="number"
            min={5}
            max={200}
            value={candidateK}
            onChange={(evt) => onCandidateKChange(Number(evt.target.value))}
          />
          <span className="helper">
            每個 Stage1 選項會取前 K 句進入聯集。DUC 任務建議 40~60。
          </span>
        </label>
        <label className="number-input">
          聯集 cap
          <input
            type="number"
            min={10}
            max={250}
            value={unionCap}
            onChange={(evt) => onUnionCapChange(Number(evt.target.value))}
          />
          <span className="helper">
            聯集最多保留的候選句數；適度限制可避免 Stage2 過慢。
          </span>
        </label>
      </div>
    </section>
  );
}
