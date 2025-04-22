# ValueVision
Multimodal House Price Prediction

This project explores the use of multimodal machine learning techniques to predict real estate prices by combining structured property data with unstructured textual descriptions and house images.

Leveraging BERT for text embeddings and CNN for image features, the model integrates various data sources into a unified embedding for downstream regression tasks. Multiple models including Random Forest, Ridge, Lasso, and Gradient Boosting were evaluated on a dataset of 50K+ Melbourne property listings.

Key Features:

- Uses structured features (e.g., rooms, location) alongside unstructured data

- Incorporates BERT-encoded textual descriptions of property listings

- Extracts visual embeddings from property images using CNN

- Evaluates model performance using MAE and RMSE, with error analysis

- Modular codebase for experimenting with multimodal feature fusion

Dataset: Real estate transaction data from Melbourne, Australia (2013–2015)


---

### 🔍 Text-Only DistilBERT Model (Fine-Tuned)

This model was fine-tuned on real estate listing descriptions to predict house prices using DistilBERT.  
It serves as the **text-only baseline** in our multimodal house price prediction pipeline.

📦 **Download model:**  
[model.safetensors (via GitHub Release)](https://github.com/diyagangwar1/ValueVision/releases/download/v1.0-text-only-model/model.safetensors)

📥 **How to load it in code:**

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "./fine_tuned_text_only_model",  # or use a manual download path
    local_files_only=True
)

