from clients import chroma_client,open_ai_client
import pandas as pd
from config import Config
from sklearn.manifold import TSNE
import numpy as np
import plotly.express as px

class Visualization:
    def __init__(self):
        collection = chroma_client.get_collection(name=Config.collection_name)
        result = collection.get(include=["embeddings","metadatas","documents"])
        self.embeddings = np.array(result['embeddings'])
        self.metadatas = [item["doc_type"] for item in result["metadatas"]]
        self.documents = [item for item in result["documents"]]
    
    def visualize_patient_data_2d(self):
        # Placeholder for visualization logic
        tsne = TSNE(n_components=2, random_state=42)

        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(self.embeddings)


        # Create DataFrame
        df = pd.DataFrame({
            "x": embeddings_2d[:, 0],
            "y": embeddings_2d[:, 1],
            "doc_type": self.metadatas,
            "document_preview": [doc[:100] for doc in self.documents]
        })

        import plotly.express as px
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="doc_type",
            hover_data=["document_preview"],
            title="t-SNE Visualization of Medical Data Embeddings"
        )

        fig.show()
    
    def visualize_patient_data_3d(self):
        tsne = TSNE(n_components=3, random_state=42)
        embeddings_3d = tsne.fit_transform(self.embeddings)

        df_3d = pd.DataFrame({
            "x": embeddings_3d[:, 0],
            "y": embeddings_3d[:, 1],
            "z": embeddings_3d[:, 2],
            "doc_type": self.metadatas,
            "document_preview": [doc[:100] for doc in self.documents]
        })

        fig_3d = px.scatter_3d(
            df_3d,
            x="x",
            y="y",
            z="z",
            color="doc_type",
            hover_data=["document_preview"],
            title="3D t-SNE Visualization of Medical Data Embeddings"
        )
        fig_3d.show()
        


        