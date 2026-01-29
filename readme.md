# **Diff-RAG: Unifying Symbolic Paths and Continuous Diffusion for Robust Multi-Hop Retrieval** 



## 🛠️ **Usage**

### 1️⃣ Install Dependencies  

**Step 1: Install Python packages**

```bash
pip install -r requirements.txt
```

**Step 2: Set up your LLM API key in `llm_config.py`**

**Step 3: Set up your Embedding API in `config.py`**

**Step 4: Download the weight file and place it in the `checkpoints/model.pt`**
weight link: https://drive.google.com/file/d/1EvsBxLm7mH6e7-EoYd8Q_fkm0S_lXnsN/view?usp=sharing

Due to the file size constraints of the supplementary material, we provide the pre-trained model weights via an external cloud storage link. 

**Anonymity Assurance:** To strictly comply with the double-blind review policy of ICML 2026, the provided Google Drive link originates from a dedicated, identity-neutral account created specifically for this submission. This access point has been configured to ensure that no author-identifying metadata, profiles, or account information are accessible to reviewers.

### 2️⃣ Quick Start Example

```bash
python preprocess_dataset.py
python run.py
```
