AI Assistant · Lab
一個以 Web 為主、支援語音輸入與任務導向操作 的 AI 助理原型，
目前聚焦在 實際可完成的使用情境（例如：產生請假信並快速用 Outlook 寄出）。
本專案採 前後端分離，前端負責互動與系統整合（語音、Outlook），
後端專注於 AI 推理、內容生成與狀態記憶。

✨ 目前可以做什麼（Current Features）
🗨️ AI 對話
使用 Gemini 模型（gemini-2.5-flash）
支援一般聊天與任務導向對話
保留最近對話記憶（in-memory）
📎 多檔案上傳
支援圖片 / PDF（可多檔）
自動摘要檔案內容
可針對上傳檔案進行提問
🎤 語音輸入（Browser-based）

使用瀏覽器 SpeechRecognition
不需安裝任何本機語音服務
語音與文字共用同一套後端邏輯
📧 請假信產生 + Outlook 寄送
AI 產生完整請假信內容
前端提供「📧 用 Outlook 寄出」按鈕
透過 mailto: 自動開啟使用者的 Outlook（或預設郵件程式）
實際寄送行為由使用者在 Outlook 中完成（安全、可控）

🧱 系統架構概覽
[ Browser ]
   ├─ UI / Chat
   ├─ SpeechRecognition
   ├─ Outlook (mailto)
   ↓
[ FastAPI Backend ]
   ├─ /chat
   ├─ Gemini API
   ├─ In-memory chat memory
   └─ 任務邏輯（請假信）

📁 專案結構說明
.
├─ index.html              # 前端主頁（UI / 語音 / Outlook）
├─ backend/
│  ├─ main.py              # FastAPI 主入口（目前所有核心邏輯）
│  ├─ __init__.py
│  ├─ action_router.py     # （未啟用）未來行為路由規劃
│  ├─ intent_parser.py     # （未啟用）意圖解析模組
│  └─ voice_api.py         # （未啟用）語音 API 拆分預留
│
├─ actions/
│  └─ send_email.py        # 寄信行為（Outlook / mailto）
│
├─ requirements.txt        # 後端依賴
├─ Procfile                # 部署設定（Render / Railway）
├─ README.md
└─ .gitignore

🔎 關於未啟用的檔案

backend/action_router.py、intent_parser.py、voice_api.py
為 未來模組化與擴充語音行為 所預留，目前 MVP 階段尚未接入。

🚀 執行方式
1️⃣ 設定環境變數
export GEMINI_API_KEY=你的_API_KEY

2️⃣ 安裝套件
pip install -r requirements.txt

3️⃣ 啟動後端
uvicorn backend.main:app --reload

4️⃣ 開啟前端

直接用瀏覽器打開：
index.html

🧠 設計理念
任務導向優先於聊天
AI 是「流程的一部分」，不是全部
敏感行為（寄信）交由使用者最終確認
MVP 階段允許集中在 main.py，避免過度工程化

🔮 未來可擴充方向（Planned）
將 main.py 拆分為 router / service
多任務行為（行事曆、提醒、排程）
WebSocket 即時對話
使用者帳號與長期記憶

Action router 正式接入（目前已預留）
