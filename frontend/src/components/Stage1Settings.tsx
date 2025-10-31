import "./Panel.css";
import {
  STAGE1_BASE_OPTIONS,
  STAGE1_LLM_OPTIONS,
} from "../utils/constants";
import type { Stage1BaseAlgorithm, Stage1LLM } from "../types";

interface Stage1SettingsProps {
  selectedBase: Stage1BaseAlgorithm;
  onChangeBase: (next: Stage1BaseAlgorithm) => void;
  selectedLLM: Stage1LLM | null;
  onChangeLLM: (next: Stage1LLM | null) => void;
}

export function Stage1Settings({
  selectedBase,
  onChangeBase,
  selectedLLM,
  onChangeLLM,
}: Stage1SettingsProps) {
  return (
    <section className="panel">
      <header>
        <h2>Stage 1 · 候選生成</h2>
        <p className="panel__subtitle">
          請選擇一種特徵式演算法（必選）與一種 LLM Encoder（可選）。
        </p>
      </header>

      <fieldset>
        <legend>演算法類（必選一種）</legend>
        <div className="chips">
          {STAGE1_BASE_OPTIONS.map((item) => {
            const active = selectedBase === item.id;
            return (
              <label
                key={item.id}
                className={`badge ${active ? "badge--active" : ""}`}
              >
                <input
                  type="radio"
                  name="stage1-base"
                  value={item.id}
                  checked={active}
                  onChange={() => onChangeBase(item.id)}
                />
                <span className="badge__text">
                  <span className="badge__label">{item.label}</span>
                  <span className="badge__meta">{item.estTime}</span>
                  <span className="badge__hint">{item.description}</span>
                </span>
              </label>
            );
          })}
        </div>
        <p className="helper">
          建議依據效能與品質需求選擇：Greedy（最快）、GRASP（折衷）、NSGA-II（最佳
          ROUGE）。
        </p>
      </fieldset>

      <fieldset>
        <legend>LLM Encoder（可選）</legend>
        <div className="chips">
          <label
            className={`badge ${selectedLLM === null ? "badge--active" : ""}`}
          >
            <input
              type="radio"
              name="stage1-llm"
              value="none"
              checked={selectedLLM === null}
              onChange={() => onChangeLLM(null)}
            />
            <span className="badge__text">
              <span className="badge__label">不使用</span>
              <span className="badge__meta">0.00 s/文</span>
              <span className="badge__hint">僅使用特徵式候選。</span>
            </span>
          </label>
          {STAGE1_LLM_OPTIONS.map((item) => {
            const active = selectedLLM === item.id;
            return (
              <label
                key={item.id}
                className={`badge ${active ? "badge--active" : ""}`}
              >
                <input
                  type="radio"
                  name="stage1-llm"
                  value={item.id}
                  checked={active}
                  onChange={() => onChangeLLM(item.id)}
                />
                <span className="badge__text">
                  <span className="badge__label">{item.label}</span>
                  <span className="badge__meta">{item.estTime}</span>
                  <span className="badge__hint">{item.description}</span>
                </span>
              </label>
            );
          })}
        </div>
        <p className="helper">
          LLM 需先下載對應模型。若環境未安裝，維持「不使用」即可。
        </p>
      </fieldset>
    </section>
  );
}
