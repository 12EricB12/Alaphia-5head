# Alaphia Inference App: Deployment & Retraining Guide

This covers everything you need to deploy the Streamlit tester app to Streamlit Community Cloud, retrain the model, and securely manage your Hugging Face credentials.

---

## 1. Deployment Instructions

To successfully deploy your app to Streamlit Community Cloud, first create a *public* GitHub repository. Your GitHub repository must have the following structure:

```text
Your-Repo-Name/
├── streamlit_app.py
├── requirements.txt
└── alaphia_5head_deberta/
    ├── __init__.py
    ├── model.py
    ├── inference.py
    └── ... (the rest of the python files)
```

**Deploying to Streamlit Cloud:**
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
2. Click **New app** and select your repository, branch, and `streamlit_app.py` as the main file path.
3. Click **Advanced settings**.
4. In the **Secrets** box, provide your Hugging Face credentials (see Section 2).
5. Click **Save** and **Deploy**.

---

## 2. Hugging Face Keys, Uploading, & Secrets

Because the model (presumably) contains proprietary or sensitive data, it must be hosted privately.

**How to get keys and upload the model:**
1. Go to [huggingface.co](https://huggingface.co/) and log in.
2. Click your profile picture -> **New Model**.
3. Give it a name (e.g., `alaphia-5head`) and **CRITICALLY: Select "Private"**.
4. Go to the **Files and versions** tab in your new repo, click **Add file**, and upload your `best_model.pt`.
5. Go to your Hugging Face **Settings -> Access Tokens**. Create a new token with **Read** permissions and copy it.

**Adding Keys to Streamlit Secrets:**
When you deploy the app (or in the Streamlit Cloud dashboard under App Settings -> Secrets), paste your credentials using this exact TOML format:

```toml
HF_REPO_ID = "YourUsername/Your-Model-Name"
HF_TOKEN = "hf_your_copied_token_here"
```

These can be found under the "Advanced settings" menu, as described earlier. Make sure to set the Python version to 3.10.

*Note: Never put these secrets directly in your code or push them to GitHub.*

---

## 3. Retraining the Model

When you receive new data, you can retrain the model locally to improve its accuracy.

1. Place your updated JSON records in `alaphia_5head_deberta/training_data.json`.
2. From your terminal at the root of the project, run:

```bash
python -m alaphia_5head_deberta.train \
  --data_path alaphia_5head_deberta/training_data.json \
  --output_dir alaphia_5head_deberta \
  --batch_size 16 \
  --epochs 5 \
  --lr 2e-5
```

This will automatically evaluate the model and save the most accurate version as a new `best_model.pt` in the folder.

**Important Requirements Fix:** 
If Streamlit Cloud defaults to a newer Python version (like 3.14), you must remove the strict version numbers inside `requirements.txt` to prevent Rust compilation crashes. It should look like this:
```text
streamlit>=1.35.0
torch>=2.1.0
huggingface_hub>=0.23.0
tiktoken>=0.7.0
sentencepiece>=0.1.99
protobuf>=4.21.0
setfit
datasets
transformers
sentence-transformers
```

### Changing the Model / Redeployment
If you retrain the model and decide to upload it to a *new* Hugging Face repository, or if your Hugging Face credentials change, **you must redeploy or reboot the Streamlit app.**

The old app session holds onto the old credentials and points to the old model. To update it:
1. Go to your Streamlit Cloud dashboard. (https://share.streamlit.io/)
2. Update your Secrets to point to the new `HF_REPO_ID`. This can be done by clicking on the three dots next to the streamlit app, clicking "settings", and then "secrets".
3. Click the three dots (`⋮`) next to your app and hit **Reboot** (or delete and redeploy). 
This forces the app to wake up, read the new secrets, and download the freshly trained model.
