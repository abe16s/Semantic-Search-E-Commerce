import streamlit as st
from search_engine import SemanticSearchEngine

# Initialize search engine
search_engine = SemanticSearchEngine(embedding_path="embeddings/product_embeddings.pt")
search_engine.load_data("datasets/processed_products.csv")
search_engine.load_embeddings()

# Streamlit interface
st.title("Toy Products Semantic Search")
st.write("Search for the best toy products using natural language!")

# Input query
query = st.text_input("Enter your search query:")
min_price, max_price = st.slider("Price Range (£):", 0, 100, (10, 50))
selected_category = st.text_input("Filter by category (optional):")

# Search button
if st.button("Search"):
    results = search_engine.search(query, top_k=10)
    
    # Filter results
    if selected_category:
        # Filter results by selected category
        results.dropna(subset=['amazon_category_and_sub_category'], inplace=True)
        results = results[results['amazon_category_and_sub_category'].str.contains(selected_category, case=False)]
    results = results[(results['price'] >= min_price) & (results['price'] <= max_price)]
    
    # Display results
    st.write(f"Found {len(results)} results:")
    for _, row in results.iterrows():
        st.subheader(row['product_name'])
        st.write(f"Price: £{row['price']}")
        st.write(row['description'])
