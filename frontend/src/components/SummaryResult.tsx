import "./Panel.css";
import type { SummaryResult } from "../types";

interface SummaryResultProps {
  result: SummaryResult | null;
  error: string | null;
  isLoading: boolean;
  onReset: () => void;
}

export function SummaryResult({
  result,
  error,
  isLoading,
  onReset,
}: SummaryResultProps) {
  if (isLoading) {
    return (
      <section className="panel panel--sticky">
        <div className="status-bar">
          <span className="status-bar__spinner" />
          <span>正在執行 Stage1 / Stage2 流程，請稍候...</span>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="panel panel--sticky">
        <header>
          <h2>摘要失敗</h2>
          <p className="panel__subtitle">
            發生錯誤時可以清除狀態後再重新送出摘要任務。
          </p>
        </header>
        <div className="result-card result-card--error">
          <h3 className="result-card__title">錯誤訊息</h3>
          <p className="result-card__content">{error}</p>
        </div>
        <button className="button" type="button" onClick={onReset}>
          清除狀態
        </button>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="panel panel--sticky">
        <header>
          <h2>摘要預覽</h2>
          <p className="panel__subtitle">
            完成設定並送出後，摘要結果會顯示在這裡。
          </p>
        </header>
        <div className="result-card result-card--placeholder">
          <h3 className="result-card__title">尚未產生摘要</h3>
          <p className="result-card__content">
            建議至少提供一篇文章內容，並確認 Stage1/Stage2 組合。送出請求後，此處將顯示摘要文字、選句索引與計時資訊。
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="panel panel--sticky">
      <header>
        <h2>摘要結果</h2>
        <p className="panel__subtitle">
          下方包含摘要內容、選句資訊以及 API 回傳的統計數值。
        </p>
      </header>

      <article className="result-card">
        <h3 className="result-card__title">摘要內容</h3>
        <div className="result-card__content">
          {result.summary || "(摘要內容為空)"}
        </div>
      </article>

      {Array.isArray(result.sentences) && result.sentences.length > 0 && (
        <article className="result-card">
          <h3 className="result-card__title">摘要句子</h3>
          <ol className="result-card__list">
            {result.sentences.map((sentence, idx) => (
              <li key={idx}>{sentence}</li>
            ))}
          </ol>
        </article>
      )}

      {Array.isArray(result.selected_indices) && (
        <article className="result-card">
          <h3 className="result-card__title">選句索引</h3>
          <p className="result-card__content">
            {result.selected_indices.join(", ")}
          </p>
        </article>
      )}

      {(result.metrics || result.timing) && (
        <div className="stats-grid">
          {result.metrics &&
            Object.entries(result.metrics).map(([key, value]) => (
              <div className="stat-card" key={`metric-${key}`}>
                <h4>{key}</h4>
                <span>
                  {typeof value === "number" ? value.toFixed(4) : String(value)}
                </span>
              </div>
            ))}
          {result.timing &&
            Object.entries(result.timing).map(([key, value]) => (
              <div className="stat-card" key={`timing-${key}`}>
                <h4>{key}</h4>
                <span>{Number(value).toFixed(2)} s</span>
              </div>
            ))}
        </div>
      )}

      <details>
        <summary>檢視 API 原始回應</summary>
        <pre className="code-block">{JSON.stringify(result, null, 2)}</pre>
      </details>
      <button className="button button--ghost" type="button" onClick={onReset}>
        清除結果
      </button>
    </section>
  );
}

