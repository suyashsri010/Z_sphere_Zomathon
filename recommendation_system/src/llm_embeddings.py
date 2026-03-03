"""
LLM-based embedding generation for semantic understanding (AI Edge).
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import Dict, List
import pickle
from pathlib import Path


class LLMEmbeddingGenerator:
    """Generate semantic embeddings using pre-trained language models."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize with lightweight sentence transformer model.
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2, 80MB, fast)
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        self.item_embeddings = {}
        self.user_embeddings = {}
        
    def generate_item_embeddings(self, items: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all menu items.
        
        Args:
            items: DataFrame of menu items
        
        Returns:
            Dictionary mapping item_id to embedding vector
        """
        print(f"Generating embeddings for {len(items)} items...")
        
        # Create rich text representations
        item_texts = []
        item_ids = []
        
        for _, item in items.iterrows():
            # Format: "{cuisine} {category}: {name}. Price: {price}"
            text = f"{item.get('name', 'Item')} - {item.get('category', 'food')} dish"
            if item.get('is_veg', False):
                text += " (vegetarian)"
            text += f". Price: ₹{item.get('price', 0):.0f}"
            
            item_texts.append(text)
            item_ids.append(item['item_id'])
        
        # Generate embeddings in batches
        embeddings = self.model.encode(item_texts, batch_size=32, show_progress_bar=True)
        
        # Store in dictionary
        self.item_embeddings = {
            item_id: embedding 
            for item_id, embedding in zip(item_ids, embeddings)
        }
        
        return self.item_embeddings
    
    def generate_user_embeddings(
        self,
        users: pd.DataFrame,
        sessions: pd.DataFrame,
        items: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Generate user preference embeddings from order history.
        
        Args:
            users: DataFrame of users
            sessions: DataFrame of sessions (for order history)
            items: DataFrame of items
        
        Returns:
            Dictionary mapping user_id to embedding vector
        """
        print(f"Generating user embeddings for {len(users)} users...")
        
        # Ensure item embeddings exist
        if not self.item_embeddings:
            self.generate_item_embeddings(items)
        
        # Group sessions by user to get order history
        user_orders = sessions.groupby('user_id')['cart_items'].apply(list).to_dict()
        
        for _, user in users.iterrows():
            user_id = user['user_id']
            
            # Get user's ordered items
            if user_id in user_orders:
                all_cart_items = []
                for cart_str in user_orders[user_id][:10]:  # Last 10 orders
                    all_cart_items.extend(cart_str.split(','))
                
                # Get embeddings for ordered items
                item_embeds = []
                for item_id in set(all_cart_items):
                    if item_id in self.item_embeddings:
                        item_embeds.append(self.item_embeddings[item_id])
                
                if item_embeds:
                    # Average embeddings (simple but effective)
                    user_embedding = np.mean(item_embeds, axis=0)
                else:
                    # Cold-start: use segment-based embedding
                    user_embedding = self._get_segment_embedding(user['segment'])
            else:
                # Cold-start: use segment-based embedding
                user_embedding = self._get_segment_embedding(user['segment'])
            
            self.user_embeddings[user_id] = user_embedding
        
        return self.user_embeddings
    
    def _get_segment_embedding(self, segment: str) -> np.ndarray:
        """Get embedding for user segment (cold-start fallback)."""
        segment_texts = {
            'budget': 'affordable food, value meals, budget-friendly options',
            'premium': 'premium dining, gourmet food, high-quality meals',
            'frequent': 'regular customer, diverse food preferences, variety'
        }
        text = segment_texts.get(segment, segment_texts['frequent'])
        return self.model.encode([text])[0]
    
    def compute_complementarity_score(
        self,
        cart_item_ids: List[str],
        candidate_item_id: str
    ) -> float:
        """
        Compute semantic complementarity between cart and candidate item.
        
        Args:
            cart_item_ids: List of item IDs in cart
            candidate_item_id: Candidate item ID
        
        Returns:
            Complementarity score [0, 1]
        """
        if not self.item_embeddings:
            return 0.5  # Neutral score if embeddings not available
        
        # Get cart embedding (average of cart items)
        cart_embeds = [
            self.item_embeddings[iid] 
            for iid in cart_item_ids 
            if iid in self.item_embeddings
        ]
        
        if not cart_embeds or candidate_item_id not in self.item_embeddings:
            return 0.5
        
        cart_embedding = np.mean(cart_embeds, axis=0)
        candidate_embedding = self.item_embeddings[candidate_item_id]
        
        # Cosine similarity
        similarity = np.dot(cart_embedding, candidate_embedding) / (
            np.linalg.norm(cart_embedding) * np.linalg.norm(candidate_embedding) + 1e-8
        )
        
        # Normalize to [0, 1]
        score = (similarity + 1) / 2
        
        return float(score)
    
    def get_embedding_features(
        self,
        cart_item_ids: List[str],
        candidate_item_id: str,
        user_id: str = None
    ) -> np.ndarray:
        """
        Get embedding-based features for model input.
        
        Args:
            cart_item_ids: List of item IDs in cart
            candidate_item_id: Candidate item ID
            user_id: Optional user ID
        
        Returns:
            Feature vector with embedding-based features
        """
        features = []
        
        # Complementarity score
        comp_score = self.compute_complementarity_score(cart_item_ids, candidate_item_id)
        features.append(comp_score)
        
        # User-item similarity (if user embedding available)
        if user_id and user_id in self.user_embeddings and candidate_item_id in self.item_embeddings:
            user_embed = self.user_embeddings[user_id]
            item_embed = self.item_embeddings[candidate_item_id]
            
            similarity = np.dot(user_embed, item_embed) / (
                np.linalg.norm(user_embed) * np.linalg.norm(item_embed) + 1e-8
            )
            user_item_sim = (similarity + 1) / 2
        else:
            user_item_sim = 0.5
        
        features.append(user_item_sim)
        
        return np.array(features)
    
    def save_embeddings(self, path: str):
        """Save generated embeddings."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'item_embeddings': self.item_embeddings,
                'user_embeddings': self.user_embeddings,
                'embedding_dim': self.embedding_dim
            }, f)
        print(f"Embeddings saved to {path}")
    
    def load_embeddings(self, path: str):
        """Load pre-computed embeddings."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.item_embeddings = data['item_embeddings']
            self.user_embeddings = data['user_embeddings']
            self.embedding_dim = data['embedding_dim']
        print(f"Embeddings loaded from {path}")
