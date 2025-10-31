import "./Panel.css";

interface DocumentInputProps {
  documentsRaw: string;
  onDocumentsChange: (value: string) => void;
  includeReference: boolean;
  onToggleReference: (value: boolean) => void;
  referenceRaw: string;
  onReferenceChange: (value: string) => void;
}

export function DocumentInput({
  documentsRaw,
  onDocumentsChange,
  includeReference,
  onToggleReference,
  referenceRaw,
  onReferenceChange,
}: DocumentInputProps) {
  return (
    <section className="panel">
      <header>
        <h2>輸入文章</h2>
        <p className="panel__subtitle">
          可貼上多篇文章，使用 <code>---</code> 分隔；或上傳預先切句的 JSON。
        </p>
      </header>
      <textarea
        className="textarea"
        placeholder={`文章 A 內容...\n\n---\n\n文章 B 內容...`}
        value={documentsRaw}
        onChange={(evt) => onDocumentsChange(evt.target.value)}
      />
      <div className="panel__divider" />
      <label className="checkbox">
        <input
          type="checkbox"
          checked={includeReference}
          onChange={(evt) => onToggleReference(evt.target.checked)}
        />
        提供參考摘要以計算 ROUGE
      </label>
      {includeReference && (
        <textarea
          className="textarea textarea--muted"
          placeholder="輸入人工參考摘要..."
          value={referenceRaw}
          onChange={(evt) => onReferenceChange(evt.target.value)}
        />
      )}
    </section>
  );
}

