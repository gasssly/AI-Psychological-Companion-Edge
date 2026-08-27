# AI Psychological Companion with Edge Inference & Multimodal Emotion Recognition
> 基於邊緣推論與多模態感知之 AI 心理陪伴系統 (阿光)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Fullstack-FF4B4B.svg)
![Edge AI](https://img.shields.io/badge/Edge%20AI-GTX%201660-success.svg)
![Quantization](https://img.shields.io/badge/Quantization-4--bit%20NF4-yellow.svg)

## 📖 專案概述 (Project Overview)
阿光是一套完全在本地端（Edge Device）執行的雙模態 AI 心理輔導系統。

當初開發的動機，是因為在測試時發現：傳統 LLM 根本聽不懂人類的「語氣」，而且非常容易對使用者「說教」。為了解決這些痛點，我用 3,500 多行程式碼從頭刻出了這套系統，主要包含兩個模組：
1. **感知前端 (Perception)**：我利用 WavLM 開發的多模態語音情緒辨識系統 (SER)。
2. **決策後端 (Decision)**：團隊合作開發的 Qwen-1.5B 邊緣推論大腦，裡面整合了 FSM 狀態機與心理學微技巧。

---

##  核心工程亮點 (Core Engineering Highlights)

### 1. 多模態語音情緒感知 (SER) 與 Late Fusion
* **工程做法**：我寫了 `analyze_voice_and_emotion()` 來處理音訊流。先把 Whisper ASR 抓到的逐字稿丟給 BERT 抽語意，再跟 WavLM 抓到的聲學特徵做 Late Fusion（後期融合）。
* **Data Blending (資料融合)**：因為原本 RAVDESS 資料庫的文字太單一（純語意準確率只有 6.45%），所以我自己動手清洗了 `GoEmotions` 和 `DailyDialog`，總共 5 萬筆資料，跨資料庫重新對齊標籤。
* **實測結果**：驗證準確率拉高到 **81.9%** (F1-Score 0.7408)。後來實測發現，就算在有環境雜訊的情況下，聲學模型還是能維持 0.85 的信心度。

* ![混淆矩陣分析圖](confusion_matrix.png)

### 2. 邊緣運算極限壓縮 (Edge LLM Quantization)
* **工程做法**：醫療資料絕對不能上雲端，所以我們挑戰在入門級的顯卡 (NVIDIA GTX 1660 6GB) 上跑 Qwen-1.5B。我用 `bitsandbytes` 掛了 **4-bit NF4 量化**，再加上 PEFT (LoRA) 微調。
* **實測結果**：成功把 VRAM 佔用壓到 **4GB 以內**，推論速度非常順暢，完全實現 100% 離線運算。

### 3. FSM 對話狀態機與動態心理路由
* **工程做法**：為了防止開放式 LLM 亂聊偏題甚至鬼打牆，我直接寫了一套 FSM 控制流。
  * `classify_dialogue_stage()` 會把對話強制鎖在 4 個階段：破冰 ➔ 評估 ➔ 陪伴 ➔ 收尾。
  * `classify_psychology_route()` 則會看使用者的狀況，動態切換 SFBT(焦點解決) 或 EFT(情緒取向) 等流派的 System Prompt。

### 4. 雙軌 RAG 記憶檢索與幻覺控制
* **工程做法**：用 `ChromaDB` 架了向量資料庫，並且用 ThreadPoolExecutor 跑並行檢索。
  * **過濾廢話**：`extract_user_facts()` 會先判斷並抽出客觀事實再存檔，不會把使用者的語氣詞也塞進資料庫。
  * **專業知識**：遇到高風險關鍵字時，`retrieve_pdf_knowledge()` 會自動去撈背後掛載的《精神科學教科書》。
* **解決幻覺**：為了徹底消滅 LLM 的「說教病」，我寫了 `check_emotion_consistency()` 來確保 AI 產出的文字跟前端 SER 聽到的情緒沒有衝突。最後把 Temperature 鎖在 0.2，實測說教率降到 0%。

### 5. 全端系統工程：Mood-Aware UI 與 PDF 報告
* **動態 UI**：介面是用 Streamlit 刻的，最特別的是背景顏色跟 CSS 動畫會看使用者現在是「平靜、低落還是焦慮」來動態變換，甚至還有內建互動式的呼吸練習 (`L1226`)。
* **臨床報告**：我調用 `reportlab` 寫了 `generate_pdf_report()`，系統聊完天會自動畫出情緒圓餅圖跟分數趨勢圖，同時產生給「使用者看」跟「醫師看」的兩種 PDF 報告。

---

##  系統架構 (System Architecture)

![系統架構圖](image.png)

---

##  技術棧 (Tech Stack)
* **後端與深度學習**: PyTorch, HuggingFace Transformers, ChromaDB
* **聲學與語音處理**: WavLM, Whisper, Edge TTS
* **模型量化與微調**: bitsandbytes (4-bit NF4), PEFT (LoRA)
* **前端與資料視覺化**: Streamlit, ReportLab, Pandas

---

##  學術聲明與版權 (Academic Integrity & License)
為了符合學術倫理與開源規範，本專案聲明如下：
1. **專案分工**：
   * **決策後端與系統介面 (阿光系統)**：為團隊合作開發之專案，包含基礎 LLM 串接、RAG 架構與 UI 介面實作。
   * **感知前端 (SER)**：為**本人獨立研究與開發**之模組，包含 WavLM 聲學特徵萃取、Data Blending 資料融合演算法、以及模型量化最佳化。
2. **資料集規範**：本專案使用之 RAVDESS、GoEmotions 等資料集僅供學術研究使用。為遵守原作者授權條款，本 Repository **不包含**原始音檔與文本資料，僅開源模型架構與資料處理腳本。
3. **隱私宣告**：本程式碼已移除所有與伺服器或個人帳號相關之環境變數與 API Keys。


