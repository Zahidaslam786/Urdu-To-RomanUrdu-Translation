# 📜 Urdu to Roman Urdu Ghazal Translator (Fine-Tuned mBART)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Model](https://img.shields.io/badge/mBART-Large--50-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Project Overview
This project focuses on bridging the linguistic gap between traditional Urdu script and the Roman Urdu script commonly used by the younger generation on social media. By fine-tuning the **Facebook mBART-large-50** model, I created a sequence-to-sequence translation system specifically tailored for **Urdu Ghazals**.

The model takes Urdu text (e.g., "سلسلے توڑ گیا وہ سبھی جاتے جاتے") and converts it into accurate Roman Urdu (e.g., "silsile toD gaya vo sabhi jaate jaate"), preserving the poetic flow and pronunciation.

## 🔗 Live Demo
Check out the working demo on Hugging Face Spaces:
👉 **[Click Here for Live Demo](https://huggingface.co/spaces/zahidaslam/urdu-to-roman-demo)**

## 📊 Dataset
The model was trained on the **Rekhta Ghazals Dataset**.
- **Source:** Scraped from Rekhta.org (via GitHub repository `amir9ume/urdu_ghazals_rekhta`).
- **Data Size:** ~1,314 Ghazals.
- **Preprocessing:** Cleaned extraneous newlines and aligned Urdu-Roman pairs.
- **Split:** 50% Training, 25% Validation, 25% Test.

## 🛠️ Methodology & Tech Stack
- **Base Model:** `facebook/mbart-large-50-many-to-many-mmt`
- **Tokenizer:** `MBart50TokenizerFast` (Src: `ur_PK`, Tgt: `en_XX`)
- **Frameworks:** PyTorch, Hugging Face Transformers, Datasets.
- **Compute:** Trained on Google Colab (T4 GPU).
- **Evaluation Metrics:** BLEU Score and Character Error Rate (CER).

## 📈 Results
After experimenting with batch sizes and learning rates, the best performance was achieved with:
- **Epochs:** 10
- **Batch Size:** 2 (Accumulation steps: 4)
- **Learning Rate:** 2e-5

| Metric | Score |
|--------|-------|
| **BLEU Score** | **61.83%** |
| **CER** | **14.55%** |

These scores indicate a high level of semantic and phonetic accuracy in the translations.

## 💻 Installation & Usage

To use this model in your own python code:

```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# 1. Load Model
model_name = "zahidaslam/mbart-urdu-to-roman-ghazal"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

# 2. Prepare Input
urdu_text = "دلِ ناداں تجھے ہوا کیا ہے"
tokenizer.src_lang = "ur_PK"
encoded_urdu = tokenizer(urdu_text, return_tensors="pt")

# 3. Generate Translation
generated_tokens = model.generate(
    **encoded_urdu,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)
translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

print(translation)
# Output: dil-e-nā-dāñ tujhe huā kyā hai
