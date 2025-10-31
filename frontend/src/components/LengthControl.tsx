import "./Panel.css";
import type { LengthUnit } from "../types";

interface LengthControlProps {
  unit: LengthUnit;
  maxSentences: number;
  maxTokens: number;
  onUnitChange: (unit: LengthUnit) => void;
  onMaxSentencesChange: (value: number) => void;
  onMaxTokensChange: (value: number) => void;
}

export function LengthControl({
  unit,
  maxSentences,
  maxTokens,
  onUnitChange,
  onMaxSentencesChange,
  onMaxTokensChange,
}: LengthControlProps) {
  return (
    <section className="panel">
      <header>
        <h2>摘要長度</h2>
        <p className="panel__subtitle">
          句數模式適合短摘要，字數模式可支援 DUC 400 字任務。
        </p>
      </header>
      <div className="chips">
        <label className={`badge ${unit === "sentences" ? "badge--active" : ""}`}>
          <input
            type="radio"
            name="length-unit"
            value="sentences"
            checked={unit === "sentences"}
            onChange={() => onUnitChange("sentences")}
          />
          句數限制
        </label>
        <label className={`badge ${unit === "tokens" ? "badge--active" : ""}`}>
          <input
            type="radio"
            name="length-unit"
            value="tokens"
            checked={unit === "tokens"}
            onChange={() => onUnitChange("tokens")}
          />
          字數限制
        </label>
      </div>
      {unit === "sentences" ? (
        <label className="number-input">
          最多句數
          <input
            type="number"
            min={1}
            max={50}
            value={maxSentences}
            onChange={(evt) => onMaxSentencesChange(Number(evt.target.value))}
          />
          <span className="helper">預設三句，可依資料集調整。</span>
        </label>
      ) : (
        <label className="number-input">
          最多字數
          <input
            type="number"
            min={50}
            max={1000}
            value={maxTokens}
            onChange={(evt) => onMaxTokensChange(Number(evt.target.value))}
          />
          <span className="helper">DUC 400 字任務建議設定為 400。</span>
        </label>
      )}
    </section>
  );
}
