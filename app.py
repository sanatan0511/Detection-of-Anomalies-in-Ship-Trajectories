import pandas as pd
import numpy as np
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SelfMadeLLM(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            
            nn.Linear(hidden_dim // 4, 2),  
        )
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        encoded = self.encoder(x)
        
        encoded_seq = encoded.unsqueeze(1)
        attn_output, _ = self.attention(encoded_seq, encoded_seq, encoded_seq)
        attn_output = attn_output.squeeze(1)
        
        combined = encoded + attn_output
        
        output = self.decoder(combined)
        return output


@st.cache_data
def load_and_process_data():
    FILE_PATH = "ar_index_global_traj.txt.gz"
    
    df_df = pd.read_csv(
        FILE_PATH,
        compression="gzip",
        comment="#",
        sep=","
    )
    
    print("Total trajectories:", len(df_df))
    
    numeric_cols = [
        "latitude_max",
        "latitude_min",
        "longitude_max",
        "longitude_min",
        "profiler_type"
    ]
    
    df_df[numeric_cols] = df_df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    
    df_df = df_df.dropna(subset=[
        "latitude_max",
        "latitude_min",
        "longitude_max",
        "longitude_min"
    ])
    
    df_df["lat_span"] = df_df.latitude_max - df_df.latitude_min
    df_df["lon_span"] = df_df.longitude_max - df_df.longitude_min
    df_df["center_lat"] = (df_df.latitude_max + df_df.latitude_min) / 2
    df_df["center_lon"] = (df_df.longitude_max + df_df.longitude_min) / 2
    
    df_df = df_df[
        (df_df.center_lat.between(-30, 30)) &
        (df_df.center_lon.between(20, 120))
    ].copy()
    
    print("Indian Ocean trajectories:", len(df_df))
    
    df_df["labels"] = (
        (df_df.lat_span > 25) |
        (df_df.lon_span > 40)
    ).astype(int)
    
    return df_df

def train_llm(df_df: pd.DataFrame) -> SelfMadeLLM:
    
    features = df_df[["lat_span", "lon_span", "center_lat", "center_lon"]].values
    labels = df_df["labels"].values
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    X_tensor = torch.FloatTensor(features_scaled)
    y_tensor = torch.LongTensor(labels)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True
    )
    
    llm = SelfMadeLLM(input_dim=4, hidden_dim=256)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(llm.parameters(), lr=0.001)
    
    llm.train()
    for epoch in range(10):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = llm(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss/len(train_loader):.4f}")
    
    return llm, scaler


def create_interactive_map(df_df):
    
    m = folium.Map(location=[0, 70], zoom_start=3)
    
    marker_cluster = MarkerCluster().add_to(m)
    
    colors = {0: 'blue', 1: 'red', -1: 'black'} 
    llm_colors = {0: 'green', 1: 'orange'}
    
    for idx, row in df_df[df_df['llm_prediction'] == 0].iterrows():
        folium.CircleMarker(
            location=[row['center_lat'], row['center_lon']],
            radius=2,
            color=llm_colors[0],
            fill=True,
            fill_opacity=0.7,
            popup=f"""
            <b>Trajectory ID:</b> {idx}<br>
            <b>Lat Span:</b> {row['lat_span']:.2f}<br>
            <b>Lon Span:</b> {row['lon_span']:.2f}<br>
            <b>LLM Prediction:</b> Normal<br>
            <b>Cluster:</b> {row['cluster']}<br>
            <b>Rule Anomaly:</b> {row['rule_anomaly']}
            """
        ).add_to(marker_cluster)
    
    for idx, row in df_df[df_df['llm_prediction'] == 1].iterrows():
        folium.CircleMarker(
            location=[row['center_lat'], row['center_lon']],
            radius=3,
            color=llm_colors[1],
            fill=True,
            fill_opacity=0.9,
            popup=f"""
            <b>Trajectory ID:</b> {idx}<br>
            <b>Lat Span:</b> {row['lat_span']:.2f}<br>
            <b>Lon Span:</b> {row['lon_span']:.2f}<br>
            <b>LLM Prediction:</b> Anomalous<br>
            <b>Cluster:</b> {row['cluster']}<br>
            <b>Rule Anomaly:</b> {row['rule_anomaly']}<br>
            <b>ML Anomaly:</b> {row['ml_anomaly'] == -1}
            """
        ).add_to(marker_cluster)
    
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 180px; height: 150px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px;">
    <b>Map Legend</b><br>
    <i style="background: green; width: 10px; height: 10px; 
              border-radius: 50%; display: inline-block;"></i> Normal (LLM)<br>
    <i style="background: orange; width: 10px; height: 10px; 
              border-radius: 50%; display: inline-block;"></i> Anomaly (LLM)<br>
    <i style="background: blue; width: 10px; height: 10px; 
              border-radius: 50%; display: inline-block;"></i> Cluster 0<br>
    <i style="background: red; width: 10px; height: 10px; 
              border-radius: 50%; display: inline-block;"></i> Cluster 1<br>
    <i style="background: black; width: 10px; height: 10px; 
              border-radius: 50%; display: inline-block;"></i> ML Anomaly
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


def main():
    st.set_page_config(layout="wide")
    st.title("🌊 ARGO Trajectory Anomaly Detection with Self-Made LLM")
    
    with st.spinner("Loading and processing data..."):
        df_df = load_and_process_data()
    
    st.sidebar.header("📊 Data Information")
    st.sidebar.write(f"Total trajectories: {len(df_df):,}")
    st.sidebar.write(f"Anomaly ratio: {df_df['labels'].mean():.2%}")
    st.sidebar.write(f"Data shape: {df_df.shape}")
    
    df_df["rule_anomaly"] = (
        (df_df.lat_span > 25) |
        (df_df.lon_span > 40)
    )
    
    features_ml = df_df[["lat_span", "lon_span"]].fillna(0)
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    df_df["ml_anomaly"] = iso.fit_predict(features_ml)
    
    with st.spinner("🧠 Training self-made LLM (requires GPU)..."):
        llm, scaler = train_llm(df_df)
    
    llm.eval()
    with torch.no_grad():
        features_llm = df_df[["lat_span", "lon_span", "center_lat", "center_lon"]].values
        features_scaled = scaler.transform(features_llm)
        X_tensor = torch.FloatTensor(features_scaled)
        outputs = llm(X_tensor)
        llm_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        df_df["llm_prediction"] = llm_predictions
    
    # 2nd part
    X_cluster = df_df[["lat_span", "lon_span"]].values
    scaler_cluster = StandardScaler()
    X_scaled_cluster = scaler_cluster.fit_transform(X_cluster)
    
    k = 2
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_df["cluster"] = kmeans.fit_predict(X_scaled_cluster)
    
    sil_score = silhouette_score(X_scaled_cluster, df_df["cluster"])
    
    def purity_score(y_true, y_pred):
        contingency = {}
        for cluster in np.unique(y_pred):
            labels = y_true[y_pred == cluster]
            most_common = Counter(labels).most_common(1)[0][1]
            contingency[cluster] = most_common
        return sum(contingency.values()) / len(y_true)
    
    purity = purity_score(df_df["labels"].values, df_df["cluster"].values)
    
    llm_accuracy = accuracy_score(df_df["labels"], df_df["llm_prediction"])
    
    st.subheader("📈 Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Silhouette Score", f"{sil_score:.4f}")
    with col2:
        st.metric("Purity Score", f"{purity:.4f}")
    with col3:
        st.metric("LLM Accuracy", f"{llm_accuracy:.4f}")
    with col4:
        st.metric("Rule Anomalies", f"{df_df['rule_anomaly'].sum():,}")
    with col5:
        st.metric("ML Anomalies", f"{(df_df['ml_anomaly'] == -1).sum():,}")
    
    tab1, tab2, tab3 = st.tabs(["🗺️ Interactive Map", "📊 Statistical Charts", "📋 Data Preview"])
    
    with tab1:
        st.subheader("Interactive Map of Trajectory Centers")
        with st.spinner("Generating interactive map..."):
            folium_map = create_interactive_map(df_df)
            folium_static(folium_map, width=1200, height=600)
        
        st.info("""
        **Map Features:**
        - **Green markers**: Normal trajectories (LLM prediction)
        - **Orange markers**: Anomalous trajectories (LLM prediction)
        - **Marker size**: Larger = anomalous, Smaller = normal
        - **Click markers** for detailed information
        - **Zoom** to see cluster patterns
        """)
    
    with tab2:
        st.subheader("Statistical Analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(
            df_df.lon_span,
            df_df.lat_span,
            c=df_df.llm_prediction,
            cmap='viridis',
            s=10,
            alpha=0.6
        )
        ax1.set_xlabel("Longitude Span")
        ax1.set_ylabel("Latitude Span")
        ax1.set_title("Self-Made LLM Predictions")
        plt.colorbar(scatter1, ax=ax1, label='LLM Prediction (0=Normal, 1=Anomaly)')
        
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(
            df_df.lon_span,
            df_df.lat_span,
            c=df_df.cluster,
            cmap='tab10',
            s=10,
            alpha=0.6
        )
        ax2.set_xlabel("Longitude Span")
        ax2.set_ylabel("Latitude Span")
        ax2.set_title("KMeans Clustering Results")
        plt.colorbar(scatter2, ax=ax2, label='Cluster')
        
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(
            df_df.lon_span,
            df_df.lat_span,
            c=df_df.labels,
            cmap='RdYlBu_r',
            s=10,
            alpha=0.6
        )
        ax3.set_xlabel("Longitude Span")
        ax3.set_ylabel("Latitude Span")
        ax3.set_title("Ground Truth Labels")
        plt.colorbar(scatter3, ax=ax3, label='Anomaly (1=Yes)')
        
        ax4 = axes[1, 1]
        correct = (df_df.llm_prediction == df_df.labels)
        colors = ['green' if c else 'red' for c in correct]
        ax4.scatter(
            df_df.lon_span,
            df_df.lat_span,
            c=colors,
            s=10,
            alpha=0.6
        )
        ax4.set_xlabel("Longitude Span")
        ax4.set_ylabel("Latitude Span")
        ax4.set_title("LLM Accuracy (Green=Correct, Red=Wrong)")
        ax4.text(0.02, 0.98, f'Accuracy: {llm_accuracy:.2%}', 
                transform=ax4.transAxes, fontsize=12, verticalalignment='top')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Processed Data Preview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            show_type = st.selectbox(
                "Show data type:",
                ["All", "Anomalies Only", "Normal Only"]
            )
        with col2:
            n_rows = st.slider("Number of rows to display:", 10, 100, 50)
        with col3:
            sort_by = st.selectbox(
                "Sort by:",
                ["None", "lat_span", "lon_span", "llm_prediction", "labels"]
            )
        
        display_df = df_df.copy()
        if show_type == "Anomalies Only":
            display_df = display_df[display_df['llm_prediction'] == 1]
        elif show_type == "Normal Only":
            display_df = display_df[display_df['llm_prediction'] == 0]
        
        if sort_by != "None":
            display_df = display_df.sort_values(sort_by, ascending=False)
        
        st.dataframe(display_df.head(n_rows))
        
        st.subheader("Data Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Column Descriptions:**")
            st.write("""
            - **center_lat/lon**: Center point of trajectory
            - **lat_span/lon_span**: Geographic span
            - **labels**: Ground truth anomaly (rule-based)
            - **llm_prediction**: Self-made LLM prediction
            - **cluster**: KMeans cluster assignment
            - **rule_anomaly**: Rule-based anomaly detection
            - **ml_anomaly**: Isolation Forest prediction
            """)
        
        with col2:
            st.write("**Statistics:**")
            stats_df = pd.DataFrame({
                'Mean': df_df[['lat_span', 'lon_span', 'center_lat', 'center_lon']].mean(),
                'Std': df_df[['lat_span', 'lon_span', 'center_lat', 'center_lon']].std(),
                'Min': df_df[['lat_span', 'lon_span', 'center_lat', 'center_lon']].min(),
                'Max': df_df[['lat_span', 'lon_span', 'center_lat', 'center_lon']].max()
            })
            st.dataframe(stats_df.style.format("{:.2f}"))
    
    st.sidebar.header("💾 Export Data")
    
    csv = df_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name="argo_trajectories_processed.csv",
        mime="text/csv",
    )
    
    with st.sidebar.expander("🧠 LLM Architecture Info"):
        st.write("""
        **Self-Made LLM Architecture:**
        - **Encoder**: 3-layer MLP with LayerNorm & GELU
        - **Attention**: Multi-head (8 heads)
        - **Decoder**: 3-layer MLP (binary classification)
        - **Parameters**: ~500K
        - **Training**: 10 epochs, AdamW optimizer
        - **Input**: lat_span, lon_span, center_lat, center_lon
        """)
        
        st.write(f"**GPU Status:** {'✅ Available' if torch.cuda.is_available() else '❌ Not available'}")
        if torch.cuda.is_available():
            st.write(f"**GPU Name:** {torch.cuda.get_device_name(0)}")
            st.write(f"**GPU Memory:** {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

if __name__ == "__main__":
    main()