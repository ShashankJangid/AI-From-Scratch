# Complete Guide: Running Your AI From Scratch in Colab & Kaggle

This project provides a **Decoder-only Generative AI (LLM)** engineered strictly from first principles and mathematical foundations. It does not use pre-trained black-box wrappers (like Hugging Face `AutoModelForCausalLM`). Instead, every single module is handcrafted.

---

## 🛠️ Architecture Highlights
1. **Rotary Positional Embeddings (RoPE)**: Dynamic rotation of query and key projections in complex space, replacing rigid absolute learned positions.
2. **RMSNorm (Root Mean Square Layer Normalization)**: Fast, stable normalization without re-centering offsets.
3. **SwiGLU Non-Linearity**: Gated feed-forward projection layers $\text{SwiGLU}(x) = (\text{SiLU}(xW_1) \otimes xW_3)W_2$.
4. **Causal Multi-Head Self-Attention**: Lower-triangular masking prohibiting attention leakage into future tokens.
5. **Dynamic Visualizer**: Real-time extraction of attention weights across heads and interactive loss curve plotting.
6. **Cosine Annealing Scheduler**: Linear warmup combined with cosine decay down to minimum learning rate.

---

## 🚀 How to Run in Google Colab

1. **Open Google Colab**:
   - Go to [colab.research.google.com](https://colab.research.google.com).
2. **Upload Notebook**:
   - Click **File** > **Upload notebook**.
   - Select [`full_ai_from_scratch.ipynb`](file:///c:/Users/D&T/Documents/Kaggle_AI/full_ai_from_scratch.ipynb) from your local folder.
3. **Turn on Free GPU Acceleration**:
   - Go to **Runtime** > **Change runtime type**.
   - Under **Hardware accelerator**, select **T4 GPU** and click **Save**.
4. **Execute**:
   - Click **Runtime** > **Run all** (or press `Ctrl + F9`).
   - The notebook downloads data, builds the AI, trains it with GPU acceleration, outputs loss curves, renders attention heatmaps, and generates custom text.

---

## ⚡ How to Run in Kaggle Notebooks

1. **Open Kaggle**:
   - Go to [kaggle.com/code](https://www.kaggle.com/code) and click **New Notebook**.
2. **Upload Notebook**:
   - In the top menu, click **File** > **Import Notebook**.
   - Upload [`full_ai_from_scratch.ipynb`](file:///c:/Users/D&T/Documents/Kaggle_AI/full_ai_from_scratch.ipynb).
3. **Turn on GPU**:
   - On the right settings sidebar, find **Accelerator**.
   - Change from **None** to **GPU P100** or **GPU T4 x 2**.
   - Ensure **Internet** toggle is set to **On** (needed to fetch the sample dataset).
4. **Run**:
   - Click **Run All**.

---

## 🎯 Customizing Output to Your Needs

To tailor what the model outputs:
1. **Custom Data**: In Cell 4, change `raw_text` to any file or text you want the AI to learn (e.g. your company documents, coding snippets, books, or chat logs):
   ```python
   with open("my_custom_data.txt", "r", encoding="utf-8") as f:
       raw_text = f.read()
   ```
2. **Custom Prompts & Inference**: In Cell 8, call `generate_text()` with your own prompts:
   ```python
   prompt = "Your custom prompt here:"
   output = generate_text(prompt, max_tokens=200, temperature=0.7)
   print(output)
   ```
   - **Lower Temperature (`0.2 - 0.5`)**: More deterministic, focused, logical predictions.
   - **Higher Temperature (`0.7 - 1.0`)**: More creative, diverse responses.

---

## 📁 Generated Files in Workspace
- [`full_ai_from_scratch.ipynb`](file:///c:/Users/D&T/Documents/Kaggle_AI/full_ai_from_scratch.ipynb): The complete Jupyter notebook ready for upload to Colab or Kaggle.
- [`model_scratch.py`](file:///c:/Users/D&T/Documents/Kaggle_AI/model_scratch.py): Clean Python standalone script with end-to-end training and visualization logic.
- [`loss_curve.png`](file:///c:/Users/D&T/Documents/Kaggle_AI/loss_curve.png): Training convergence graph.
- [`attention_heatmap.png`](file:///c:/Users/D&T/Documents/Kaggle_AI/attention_heatmap.png): Attention heads activation heatmap.
