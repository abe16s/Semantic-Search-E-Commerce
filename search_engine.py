from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

class SemanticSearchEngine:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2', embedding_path=None):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.dataset = None
        self.embedding_path = embedding_path

    def load_data(self, dataset_path):
        """
        Load and prepare the dataset.
        """
        self.dataset = pd.read_csv(dataset_path)
        print(f"Loaded dataset with {len(self.dataset)} products.")

    def compute_embeddings(self):
        """
        Compute embeddings for product descriptions.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Please load the dataset first.")
        
        print("Computing embeddings...")
        self.embeddings = self.model.encode(
            self.dataset['combined_text'].tolist(), 
            convert_to_tensor=True
        )
        
        # Save embeddings for reuse
        if self.embedding_path:
            torch.save(self.embeddings, self.embedding_path)
            print(f"Embeddings saved to {self.embedding_path}")

    def load_embeddings(self):
        """
        Load precomputed embeddings.
        """
        if self.embedding_path:
            self.embeddings = torch.load(self.embedding_path)
            print(f"Loaded embeddings from {self.embedding_path}")

    def search(self, query, top_k=10):
        """
        Search for products similar to the user query.
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not computed or loaded. Please compute or load embeddings.")
        
        # Encode the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Compute similarities
        similarities = util.cos_sim(query_embedding, self.embeddings)
        
        # Rank products by similarity
        top_results = similarities.argsort(descending=True)[0][:top_k]
        return self.dataset.iloc[top_results.cpu().numpy()]

# Example usage
if __name__ == "__main__":
    search_engine = SemanticSearchEngine(embedding_path="embeddings/product_embeddings.pt")
    search_engine.load_data("datasets/processed_products.csv")
    search_engine.compute_embeddings()
    
    # Example query
    query = "Train set for kids"
    results = search_engine.search(query, top_k=5)
    for _, row in results.iterrows():
        print(f"Product: {row['product_name']} - £{row['price']}")
