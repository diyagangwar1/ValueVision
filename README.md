# ValueVision
Multimodal House Price Prediction

This project explores the use of multimodal machine learning techniques to predict real estate prices by combining structured property data with unstructured textual descriptions and house images.

Leveraging BERT for text embeddings and CNN for image features, the model integrates various data sources into a unified embedding for downstream regression tasks. Multiple models including Random Forest, Ridge, Lasso, and Gradient Boosting were evaluated on a dataset of 50K+ Melbourne property listings.

Key Features:

📊 Uses structured features (e.g., rooms, location) alongside unstructured data

📝 Incorporates SBERT-encoded textual descriptions of property listings

🖼️ Extracts visual embeddings from property images using CLIP

📈 Evaluates model performance using MAE and RMSE, with error analysis

🧪 Modular codebase for experimenting with multimodal feature fusion

Dataset: Real estate transaction data from Melbourne, Australia (2013–2015)
