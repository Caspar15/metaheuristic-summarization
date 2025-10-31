import "./Panel.css";
import { PRESET_OPTIONS } from "../utils/constants";

interface PresetSelectorProps {
  value: string;
  onChange: (presetId: string) => void;
}

export function PresetSelector({ value, onChange }: PresetSelectorProps) {
  return (
    <section className="panel">
      <header>
        <h2>任務模板</h2>
        <p className="panel__subtitle">
          依據常見任務載入預設長度與候選參數。切換後仍可在下方微調。
        </p>
      </header>
      <div className="chips">
        {PRESET_OPTIONS.map((preset) => {
          const active = value === preset.id;
          return (
            <label
              key={preset.id}
              className={`badge ${active ? "badge--active" : ""}`}
              title={preset.description}
            >
              <input
                type="radio"
                name="preset"
                value={preset.id}
                checked={active}
                onChange={() => onChange(preset.id)}
              />
              {preset.label}
            </label>
          );
        })}
      </div>
    </section>
  );
}

