# Metaheuristic Summarization – Frontend

React + Vite dashboard for orchestrating Stage1/Stage2 summarization runs.

## 開發環境

```bash
cd frontend
npm install --include=dev
```

### 環境變數

建立 `frontend/.env.local` 指定後端 API 端點（範例使用 Flask 版）：

```
VITE_API_ENDPOINT=http://127.0.0.1:5000/summarize
```

若改用 FastAPI（`uvicorn backend.main:app --reload --port 8000`），改成：

```
VITE_API_ENDPOINT=http://127.0.0.1:8000/summarize
```

### 開發模式

```bash
npm run dev
```

預設在 `http://127.0.0.1:5173`，檢視摘要設定表單與結果即時回饋。

### 生產建置

```bash
npm run build
npm run preview
```

## 主要功能

- 上傳/貼上多篇文章（以 `---` 分隔）。
- Stage1 候選生成：演算法單選（Greedy/GRASP/NSGA-II） + LLM 單選或不使用。
- Stage2 精煉：fast / fast_grasp / fast_nsga2 / greedy。
- 長度控制：三句摘要或 400 token 模式。
- 顯示摘要、選句索引、耗時；支援 reference 以計算 ROUGE。
- 全介面已針對桌機與窄螢幕響應式調整。

