# Semantic Search E-Commerce Tool

## Project Description

This project builds a semantic search engine tailored for e-commerce product recommendations. Using advanced natural language processing techniques, it computes embeddings for product descriptions and leverages cosine similarity to rank and retrieve similar products to user queries.

## Dataset

The dataset used for this project is based on toy products from Amazon, available on [Kaggle](https://www.kaggle.com/datasets/PromptCloudHQ/toy-products-on-amazon). The dataset includes various attributes such as product names, descriptions, prices, availability, reviews, and categories.

### Sample Dataset Preview

| uniq_id                  | product_name                                                                 | manufacturer | price   | number_available_in_stock | number_of_reviews | amazon_category_and_sub_category                                  | description                                                                                           |
|--------------------------|----------------------------------------------------------------------------|--------------|---------|---------------------------|------------------|----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| eac7efa5dbd3d667f26eb3d3 | Hornby 2014 Catalogue                                                         | Hornby       | £3.42   | 5 new                     | 15               | Hobbies > Model Trains & Railway Sets > Rail Vehicles               | Product Description Hornby 2014 Catalogue Box Set Includes Everything Needed for Beginners                |
| b17540ef7e86e461d37f3ae5 | FunkyBuys® Large Christmas Holiday Express Festival Train Set                  | FunkyBuys    | £16.99  | NaN                         | 2                | Hobbies > Model Trains & Railway Sets > Rail Vehicles               | Size Name:Large FunkyBuys® Large Christmas Holiday Express Festival Train Set                           |

## Prerequisites

- Python 3.x
- Libraries:
  - pandas
  - sentence-transformers
  - torch
  - streamlit 

## Steps to Run the Tool

### 1. Clone the Repository

```bash
git clone git@github.com:abe16s/Semantic-Search-E-Commerce.git
cd Semantic-Search-E-Commerce
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Preprocess Data

The preprocessing script cleans and prepares the dataset for embedding computations.

```bash
python preprocess.py --input_path datasets/toy-products-on-amazon.csv --output_path dataset/processed_products.csv
```

### 4. Load and Compute Embeddings

Compute or load precomputed embeddings for the dataset.

```bash
python app.py
```

or

If you have precomputed embeddings:

```bash
python app.py --embedding_path embeddings/product_embeddings.pt
```

### 5. Search for Products

Run a search query to find similar products. Using Streamlit:

```bash
streamlit run app.py
```

* Once the app starts, it will provide a local development URL (e.g., http://localhost:8501/) in the terminal.
* Navigate to this URL in your web browser.
* You will see a simple UI where you can input your search query and specify categories to filter results.


## Notes

- Ensure your dataset is in the specified format (`dataset/processed_products.csv`) for compatibility.
- Precomputed embeddings can greatly reduce processing time during search operations.
- For deploying to a web interface, consider integrating **Streamlit** for a simple UI.