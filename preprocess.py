import pandas as pd
import re

def clean_text(text):
    """
    Clean text by removing special characters, HTML tags, and extra spaces.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.lower().strip()

def preprocess_data(input_path, output_path):
    """
    Preprocess the dataset and save the cleaned dataset.
    """
    # Load dataset
    df = pd.read_csv(input_path)
    
    # Handle missing values
    df.fillna('', inplace=True)
    
    # Combine text fields for embedding
    df['combined_text'] = (
        df['product_name'] + " " +
        df['description'] + " " +
        df['amazon_category_and_sub_category']
    )
    df['combined_text'] = df['combined_text'].apply(clean_text)
    
    # Clean price field
    df['price'] = df['price'].str.replace('£', '').str.strip()  # Remove '£' and extra spaces
    df['price'] = pd.to_numeric(df['price'], errors='coerce')   # Convert to numeric, set invalid values to NaN
    
    # Extract numeric ratings
    df['average_review_rating'] = df['average_review_rating'].str.extract(r'([\d.]+)').astype(float)
    
    # Selecting only relevant columns for semantic search
    df = df[['product_name', 'description', 'amazon_category_and_sub_category', 'price', 'combined_text', 'average_review_rating']]

    # Save preprocessed dataset
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

# Run preprocessing (example usage)
if __name__ == "__main__":
    preprocess_data("datasets/amazon_co-ecommerce_sample.csv", "datasets/processed_products.csv")
