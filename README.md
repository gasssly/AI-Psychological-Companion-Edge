# AI Psychological Companion with Edge Inference & Multimodal Emotion Recognition
> 基於邊緣推論與多模態感知之 AI 心理陪伴系統

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![Edge AI](https://img.shields.io/badge/Edge%20AI-GTX%201660-success.svg)
![Quantization](https://img.shields.io/badge/Quantization-4--bit%20NF4-yellow.svg)

## 📖 專案概述 (Project Overview)
本專案為一個運行於本地端（Edge Device）的高隱私、具備同理心之雙模態 AI 心理輔導系統。為解決傳統語言模型（LLM）無法感知非語文線索（如語氣情緒）以及容易產生「說教引導 (Directive Bias)」的痛點，本系統分為兩大核心模組：
1. **感知前端 (Perception Front-end)**：基於 WavLM 建立的多模態語音情緒辨識系統 (SER)。
2. **決策後端 (Decision Back-end)**：基於 Qwen-1.5B 並導入心理學「微技巧 (Micro-skills)」約束之邊緣推論引擎。

---

##  核心亮點 (Key Features)

### 1. 多模態語音情緒辨識與資料融合 (Data Blending)
* **痛點**：標準 RAVDESS 語音資料庫文字內容單一，導致傳統語意模型 (BERT) 準確率僅 6.45%。
* **解法**：自主引入 5 萬筆 `GoEmotions` 與 `DailyDialog` 資料，進行跨資料庫的情緒標籤映射與融合。
* **成果**：在嚴謹的測試集驗證下，模型最佳驗證準確率達 **81.9%**，測試集 F1-Score 達 **0.7408**。

### 2. 本地端高隱私推論 (Edge AI & Quantization)
* **痛點**：心理諮商具備極高機敏性，若使用雲端 API (如 OpenAI) 將面臨醫療資料外洩風險。
* **解法**：於入門級邊緣運算設備 (**NVIDIA GTX 1660 6GB**) 上部署 Qwen-1.5B 模型。
* **成果**：透過 **4-bit NF4 量化**與 **LoRA 微調**技術，將模型 VRAM 佔用成功壓縮至 **4GB 內**，實現 100% 離線推論。

### 3. 臨床微技巧約束 (Micro-skills Prompting)
* **解法**：將心理學大師 Allen E. Ivey 的「微技巧階層」轉化為 System Prompt。強制模型遵循「情感反映 ➔ 重述 ➔ 開放式探問」之對話邏輯。
* **成果**：經參數消融實驗 (Temperature=0.2)，在極端壓力情境下成功將 LLM 的「說教幻覺」發生率**降至 0%**。

---

##  系統架構 (System Architecture)

![系統架構圖](image.png)


---

## 🛠️ 技術棧 (Tech Stack)
* **深度學習框架**: PyTorch, HuggingFace Transformers
* **聲學與語音處理**: WavLM, Whisper
* **模型量化與微調**: bitsandbytes (4-bit NF4), PEFT (LoRA)
* **大型語言模型**: Qwen-1.5B-Chat

---

## ⚠️ 學術聲明與版權 (Academic Integrity & License)
為了符合學術倫理與開源規範，本專案聲明如下：
1. **專案分工**：
   * **決策後端 (阿光系統)**：為團隊合作開發之專案，包含基礎 LLM 串接與 UI 介面實作。
   * **感知前端 (SER)**：為**本人獨立研究與開發**之模組，包含 WavLM 聲學特徵萃取、Data Blending 演算法、以及模型量化最佳化。
2. **資料集規範**：本專案使用之 RAVDESS、GoEmotions 等資料集僅供學術研究使用。為遵守原作者授權條款，本 Repository **不包含**原始音檔與文本資料，僅開源模型架構與資料處理腳本。
3. **機敏資訊**：本程式碼已移除所有與伺服器或個人帳號相關之環境變數與 API Keys。
