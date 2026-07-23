# Presentation Guideline

> Extracted and organized from the teacher's handwritten notes. This is a checklist of points your project presentation should cover.

---

## 1. Dataset
Describe your dataset clearly:
- **Features** — what columns / variables does it contain?
- **Source** — where did the data come from?

---

## 2. Identify the Problem Type
Decide what kind of ML problem you have, based on whether a **target** is available:

| Situation | Approach |
|-----------|----------|
| Target **not** given | Unsupervised / Clustering |
| Target **is** given | Supervised |

> **Target = what is the correlation / relation** you are trying to predict or explain.

---

## 3. What Technique Are You Using?
State clearly whether it is **supervised** or **unsupervised**, then identify the specific task:

- **Classification** → predicts a *category*
- **Clustering** → groups into *categories* (unsupervised)
- **Regression** → *forecasting*; maps input → output / works on series data
- **Reinforcement Learning** → reward-based action learning

**Key question:** *What are we trying to do with our dataset* — clustering, regression, or classification?

---

## 4. Feature Engineering & Importance
Cover these in order:

1. **Feature importance** — explain *how* you determine it
2. **Feature engineering** — how features are created / transformed
3. **Feature selection** — do this **last**

**Practical tools for feature importance:**
- Shallow / tree-based methods are useful here
- **Random Forest → XGBoost** → use for **feature importance detection**

---

## 5. Research Gap
Identify the gap your work addresses, and point toward **cutting-edge models** you want to use:

- LSTM
- CNN
- Transformer
- GAN
- GNN

---

## 6. Base Model / Benchmark / Base Paper Comparison
Compare your approach against a base model, benchmark, or base paper.

**Use reputable sources — Journals, NOT Conferences:**
- ACM
- IEEE / IEEE Transactions
- Springer
- Elsevier
- MDPI
- Nature (Scientific Reports)
- Frontiers

---

### Quick Checklist
- [ ] Dataset: features + source described
- [ ] Problem type identified (supervised vs unsupervised)
- [ ] Target / correlation defined
- [ ] Technique chosen (classification / clustering / regression / RL)
- [ ] Feature engineering, importance & selection explained
- [ ] Research gap stated with cutting-edge models
- [ ] Benchmark / base paper comparison (journal sources)
