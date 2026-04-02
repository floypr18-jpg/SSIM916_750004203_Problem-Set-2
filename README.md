# SSIM916_750004203_Problem-Set-2
# Problem Set 2 – Sentiment Classification of Software Reviews Using Machine Learning

This project classifies Amazon Software reviews as positive or negative using two approaches: **TF-IDF + Logistic Regression** and **DistilBERT (frozen) + Logistic Regression**.  
Developed and tested using Python 3.13 (Anaconda) and scikit-learn.

---

## Repository Structure

- `analysis.ipynb` — Main notebook containing the full analysis pipeline  
- `data/software_sample.csv` — Subsampled dataset (12,000 reviews, included)  
- `data/train_embeddings.npy` — Not included (exceeds GitHub's 25 MB file limit). Regenerated automatically on first run.  
- `data/test_embeddings.npy` — Cached DistilBERT embeddings, test set (included)  
- `data/model_comparison.csv` — Summary metrics table  
- `data/fig1–fig5.png` — Figures produced by the notebook  

---

## Dataset

The dataset is the Software category subset of the Amazon Reviews 2023 corpus:

Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). *Bridging language and items for retrieval and recommendation.* arXiv:2403.03952.  
https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

The raw file (`Software.jsonl`, ~1.87 GB) is **not included** in this repository and is not needed to reproduce results. 

If you wish to download it directly: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Software.jsonl

The pre-processed subsample (`software_sample.csv`) and cached test embeddings are included and loaded automatically by the notebook.

The dataset contains **12,000 reviews** with a binary sentiment label derived from star ratings (1–2 stars = negative, 4–5 stars = positive, 3-star reviews excluded).

If you wish to regenerate from the raw file:

1. Download `Software.jsonl` from the HuggingFace link above.
2. Place it in the `data/` folder.
3. Delete `software_sample.csv`, `train_embeddings.npy`, and `test_embeddings.npy`.
4. Run the notebook from top to bottom — all files will be regenerated automatically.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/floypr18-jpg/SSIM916_750004203_problem-set-2.git
cd SSIM916_750004203_problem-set-2
```

### 2. (Optional but recommended) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
```

For Windows:
```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open Jupyter Notebook

```bash
jupyter notebook
```

### 5. Run `analysis.ipynb` from top to bottom.

---

## Expected Outputs

The notebook produces:

- Corpus overview plot (class distribution and review length distribution)  
- TF-IDF + LR metrics (Accuracy, Macro F1, ROC-AUC, Confusion Matrix)  
- DistilBERT + LR metrics (Accuracy, Macro F1, ROC-AUC, Confusion Matrix)  
- Model comparison bar chart  
- Top 15 predictive terms by sentiment class  
- RandomizedSearchCV score distributions for both models  

All results are reproducible by running the notebook sequentially.

---

## Notes

- The train–test split uses stratification with `random_state=42`.
- Both models are tuned using `RandomizedSearchCV(cv=5, scoring='f1_macro')`.
- Best parameters are selected by the search — no manual configuration.
- `train_embeddings.npy` is not included due to GitHub's 25 MB file size limit. The notebook detects this and regenerates it automatically (~5 minutes). A HuggingFace account may be required for the DistilBERT download — create a free Read token at https://huggingface.co/settings/tokens and run `from huggingface_hub import login; login()` if prompted.
- `test_embeddings.npy` and `software_sample.csv` are included, so the dataset loading step and test set evaluation run instantly from cache.
- All random seeds are fixed at 42 throughout.
