import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
import time
import os
from datetime import datetime
import json
import zipfile
import shutil

# Configuration
st.set_page_config(
    page_title="Pipeline Wolof Complet",
    page_icon="🇸🇳",
    layout="wide"
)

# CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E8449;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Modèles africains
MODELS_CONFIG = {
    'AfroLM': 'bonadossou/afrolm_active_learning',
    'AfriBERTa': 'castorini/afriberta_base',
    'AfroXLMR Base': 'Davlan/afro-xlmr-base',
    'XLM-R Wolof': 'Davlan/xlm-roberta-base-finetuned-wolof'
}

# Paramètres
PARAMS = {
    'learning_rate': 2e-5,
    'batch_size': 16,
    'num_epochs': 3,
    'max_length': 128,
    'warmup_steps': 500,
    'weight_decay': 0.01
}

# Session state
if 'phase1_results' not in st.session_state:
    st.session_state.phase1_results = []
if 'phase2_result' not in st.session_state:
    st.session_state.phase2_result = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'saved_models' not in st.session_state:
    st.session_state.saved_models = []

# Header
st.markdown('<div class="main-header">🇸🇳 Pipeline Complet - Détection Discours Haineux Wolof</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Paramètres d'entraînement
    st.subheader("🎯 Paramètres")
    epochs = st.slider("Epochs", 1, 10, PARAMS['num_epochs'])
    batch_size = st.selectbox("Batch size", [8, 16, 32], index=1)
    learning_rate = st.select_slider(
        "Learning rate",
        options=[1e-5, 2e-5, 3e-5, 5e-5],
        value=PARAMS['learning_rate'],
        format_func=lambda x: f"{x:.0e}"
    )
    max_length = st.slider("Max length", 64, 256, PARAMS['max_length'])
    
    st.markdown("---")
    
    # Sélection des modèles Phase 1
    st.subheader("📦 Modèles Phase 1")
    selected_models = {}
    for model_name in MODELS_CONFIG.keys():
        if st.checkbox(model_name, value=True, key=f"model_{model_name}"):
            selected_models[model_name] = MODELS_CONFIG[model_name]
    
    st.markdown("---")
    
    # Info GPU
    if torch.cuda.is_available():
        st.success(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("⚠️ CPU uniquement")
    
    # Dossier de sauvegarde
    st.markdown("---")
    st.subheader("💾 Sauvegarde")
    save_dir = st.text_input("Dossier", value="./trained_models")

# Fonctions
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    if 'Tweet' not in df.columns or 'Class' not in df.columns:
        st.error("❌ CSV doit contenir 'Tweet' et 'Class'")
        return None
    df = df[['Tweet', 'Class']].dropna()
    df['Tweet'] = df['Tweet'].astype(str).str.strip()
    df['Class'] = df['Class'].astype(int)
    return df

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def train_single_model(model_name, model_path, train_dataset, test_dataset, params, progress_container):
    try:
        with progress_container:
            st.info(f"🔄 {model_name} - Chargement...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, ignore_mismatched_sizes=True)
        
        def tokenize(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=params['max_length'])
        
        with progress_container:
            st.info(f"🔧 {model_name} - Tokenization...")
        
        tok_train = train_dataset.map(tokenize, batched=True)
        tok_test = test_dataset.map(tokenize, batched=True)
        
        training_args = TrainingArguments(
            output_dir=f'./temp_{model_name.replace(" ", "_")}',
            evaluation_strategy='epoch',
            save_strategy='no',
            learning_rate=params['learning_rate'],
            per_device_train_batch_size=params['batch_size'],
            per_device_eval_batch_size=params['batch_size'],
            num_train_epochs=params['num_epochs'],
            weight_decay=params['weight_decay'],
            warmup_steps=params['warmup_steps'],
            load_best_model_at_end=False,
            metric_for_best_model='f1',
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            report_to='none',
            seed=42
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tok_train,
            eval_dataset=tok_test,
            compute_metrics=compute_metrics
        )
        
        with progress_container:
            st.info(f"🎯 {model_name} - Entraînement {params['num_epochs']} epochs...")
        
        start = time.time()
        trainer.train()
        duration = time.time() - start
        
        results = trainer.evaluate()
        predictions = trainer.predict(tok_test)
        preds = predictions.predictions.argmax(-1)
        cm = confusion_matrix(tok_test['label'], preds)
        
        with progress_container:
            st.success(f"✅ {model_name} - F1: {results['eval_f1']*100:.2f}%")
        
        return {
            'model_name': model_name,
            'model_path': model_path,
            'accuracy': results['eval_accuracy'],
            'precision': results['eval_precision'],
            'recall': results['eval_recall'],
            'f1': results['eval_f1'],
            'loss': results['eval_loss'],
            'training_time': duration,
            'confusion_matrix': cm,
            'trainer': trainer,
            'tokenizer': tokenizer
        }
    except Exception as e:
        with progress_container:
            st.error(f"❌ {model_name}: {str(e)}")
        return None

def save_model_to_disk(result, save_dir, phase):
    """Sauvegarde un modèle entraîné sur le disque"""
    model_name_clean = result['model_name'].replace(' ', '_').replace('/', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(save_dir, f"{phase}_{model_name_clean}_{timestamp}")
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Sauvegarder le modèle et tokenizer
    result['trainer'].save_model(model_dir)
    result['tokenizer'].save_pretrained(model_dir)
    
    # Sauvegarder les métadonnées
    metadata = {
        'model_name': result['model_name'],
        'model_path': result['model_path'],
        'phase': phase,
        'timestamp': timestamp,
        'accuracy': float(result['accuracy']),
        'precision': float(result['precision']),
        'recall': float(result['recall']),
        'f1': float(result['f1']),
        'loss': float(result['loss']),
        'training_time': float(result['training_time'])
    }
    
    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return model_dir

def create_zip_download(model_dir):
    """Crée un zip pour téléchargement"""
    zip_path = f"{model_dir}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(model_dir))
                zipf.write(file_path, arcname)
    return zip_path

def predict_text(text, trainer, tokenizer, max_length=128):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    trainer.model.eval()
    with torch.no_grad():
        outputs = trainer.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = probs.argmax().item()
    conf = probs[0][pred].item()
    return {
        'label': 'Abusif' if pred == 1 else 'Non-abusif',
        'confidence': conf * 100,
        'prob_non_abusif': probs[0][0].item() * 100,
        'prob_abusif': probs[0][1].item() * 100
    }

# Onglets
tabs = st.tabs(["📊 Dataset", "🚀 Phase 1 - Comparaison", "🔄 Phase 2 - Ré-entraînement", "🧪 Testeur", "💾 Modèles Sauvegardés"])

# TAB 1: Dataset
with tabs[0]:
    st.header("📊 Upload Dataset Phase 1")
    
    uploaded_file = st.file_uploader("Charger CSV pour Phase 1", type=['csv'], key="phase1_csv")
    
    if uploaded_file:
        df = load_data(uploaded_file)
        
        if df is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total", len(df))
            with col2:
                st.metric("Non-abusif", f"{(df['Class']==0).sum()} ({(df['Class']==0).sum()/len(df)*100:.1f}%)")
            with col3:
                st.metric("Abusif", f"{(df['Class']==1).sum()} ({(df['Class']==1).sum()/len(df)*100:.1f}%)")
            
            fig = go.Figure(data=[go.Pie(
                labels=['Non-abusif', 'Abusif'],
                values=[(df['Class']==0).sum(), (df['Class']==1).sum()],
                hole=0.3,
                marker_colors=['#2ECC71', '#E74C3C']
            )])
            fig.update_layout(title="Distribution des Classes")
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df.head(20), use_container_width=True)

# TAB 2: Phase 1
with tabs[1]:
    st.header("🚀 Phase 1 - Comparaison des 4 Modèles")
    
    if uploaded_file and df is not None:
        st.info(f"📦 {len(selected_models)} modèle(s) sélectionné(s)")
        
        if st.button("🚀 Lancer Phase 1", type="primary", use_container_width=True):
            st.session_state.phase1_results = []
            
            # Split
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                df['Tweet'].tolist(), df['Class'].tolist(),
                test_size=0.2, random_state=42, stratify=df['Class']
            )
            train_ds = Dataset.from_dict({'text': train_texts, 'label': train_labels})
            test_ds = Dataset.from_dict({'text': test_texts, 'label': test_labels})
            
            params = {
                'num_epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'max_length': max_length,
                'warmup_steps': PARAMS['warmup_steps'],
                'weight_decay': PARAMS['weight_decay']
            }
            
            progress_bar = st.progress(0)
            progress_container = st.empty()
            
            for idx, (name, path) in enumerate(selected_models.items()):
                result = train_single_model(name, path, train_ds, test_ds, params, progress_container)
                
                if result:
                    st.session_state.phase1_results.append(result)
                    
                    # Sauvegarder automatiquement
                    model_dir = save_model_to_disk(result, save_dir, "phase1")
                    st.session_state.saved_models.append({
                        'phase': 'Phase 1',
                        'model_name': name,
                        'path': model_dir,
                        'f1': result['f1'],
                        'timestamp': datetime.now()
                    })
                
                progress_bar.progress((idx + 1) / len(selected_models))
            
            st.session_state.phase1_results.sort(key=lambda x: x['f1'], reverse=True)
            st.session_state.best_model = st.session_state.phase1_results[0]
            
            progress_container.success(f"✅ Phase 1 terminée! Meilleur: {st.session_state.best_model['model_name']}")
            st.balloons()
    
    # Afficher résultats
    if st.session_state.phase1_results:
        st.markdown("---")
        st.subheader("📊 Résultats Phase 1")
        
        # Tableau détaillé avec toutes les métriques
        results_df = pd.DataFrame([{
            'Rang': f"#{i+1}",
            'Modèle': r['model_name'],
            'F1-Score': f"{r['f1']*100:.2f}%",
            'Accuracy': f"{r['accuracy']*100:.2f}%",
            'Precision': f"{r['precision']*100:.2f}%",
            'Recall': f"{r['recall']*100:.2f}%",
            'Loss': f"{r['loss']:.4f}",
            'Temps': f"{r['training_time']:.1f}s"
        } for i, r in enumerate(st.session_state.phase1_results)])
        
        # Style du tableau avec mise en surbrillance du meilleur
        def highlight_best(s):
            return ['background-color: #FFD700; font-weight: bold' if s.name == 0 else '' for _ in s]
        
        styled_df = results_df.style.apply(highlight_best, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("📈 Visualisations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique F1-Score
            fig_f1 = go.Figure(data=[go.Bar(
                y=[r['model_name'] for r in st.session_state.phase1_results],
                x=[r['f1']*100 for r in st.session_state.phase1_results],
                orientation='h',
                marker_color=['#FFD700' if i==0 else '#87CEEB' for i in range(len(st.session_state.phase1_results))],
                text=[f"{r['f1']*100:.2f}%" for r in st.session_state.phase1_results],
                textposition='auto',
                textfont=dict(size=12, color='black')
            )])
            fig_f1.update_layout(
                title="🏆 Comparaison F1-Score",
                xaxis_title="F1-Score (%)",
                yaxis_title="Modèle",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_f1, use_container_width=True)
        
        with col2:
            # Graphique Radar Multi-Modèles
            fig_radar = go.Figure()
            
            for r in st.session_state.phase1_results:
                fig_radar.add_trace(go.Scatterpolar(
                    r=[
                        r['accuracy']*100,
                        r['precision']*100,
                        r['recall']*100,
                        r['f1']*100
                    ],
                    theta=['Accuracy', 'Precision', 'Recall', 'F1'],
                    fill='toself',
                    name=r['model_name']
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title="📊 Métriques Multi-Modèles",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Matrice de confusion du meilleur modèle
        st.markdown("---")
        st.subheader(f"🎯 Détails du Meilleur Modèle: {st.session_state.best_model['model_name']}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Matrice de confusion
            cm = st.session_state.best_model['confusion_matrix']
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Prédit Non-abusif', 'Prédit Abusif'],
                y=['Réel Non-abusif', 'Réel Abusif'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                showscale=False
            ))
            fig_cm.update_layout(
                title="Matrice de Confusion",
                height=350
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # Métriques du meilleur modèle
            st.metric("🎯 Accuracy", f"{st.session_state.best_model['accuracy']*100:.2f}%")
            st.metric("🎯 Precision", f"{st.session_state.best_model['precision']*100:.2f}%")
            st.metric("🎯 Recall", f"{st.session_state.best_model['recall']*100:.2f}%")
            st.metric("🎯 F1-Score", f"{st.session_state.best_model['f1']*100:.2f}%")
            st.metric("📉 Loss", f"{st.session_state.best_model['loss']:.4f}")
            st.metric("⏱️ Temps", f"{st.session_state.best_model['training_time']:.1f}s")
        
        # Comparaison toutes métriques
        st.markdown("---")
        st.subheader("📊 Comparaison Détaillée de Toutes les Métriques")
        
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        model_names = [r['model_name'] for r in st.session_state.phase1_results]
        
        fig_compare = go.Figure()
        
        for metric in metrics_names:
            values = []
            for r in st.session_state.phase1_results:
                if metric == 'Accuracy':
                    values.append(r['accuracy']*100)
                elif metric == 'Precision':
                    values.append(r['precision']*100)
                elif metric == 'Recall':
                    values.append(r['recall']*100)
                else:  # F1-Score
                    values.append(r['f1']*100)
            
            fig_compare.add_trace(go.Bar(
                name=metric,
                x=model_names,
                y=values,
                text=[f"{v:.1f}%" for v in values],
                textposition='auto'
            ))
        
        fig_compare.update_layout(
            title="Comparaison Complète des Métriques",
            xaxis_title="Modèle",
            yaxis_title="Score (%)",
            barmode='group',
            height=450,
            showlegend=True
        )
        st.plotly_chart(fig_compare, use_container_width=True)

# TAB 3: Phase 2
with tabs[2]:
    st.header("🔄 Phase 2 - Ré-entraînement avec Nouvelles Données")
    
    if st.session_state.best_model:
        st.success(f"🏆 Modèle sélectionné: {st.session_state.best_model['model_name']}")
        st.info(f"📊 Phase 1 - F1: {st.session_state.best_model['f1']*100:.2f}%")
        
        uploaded_file2 = st.file_uploader("Charger NOUVEAU CSV pour Phase 2", type=['csv'], key="phase2_csv")
        
        if uploaded_file2:
            df2 = load_data(uploaded_file2)
            
            if df2 is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Nouveau dataset", f"{len(df2)} exemples")
                with col2:
                    diff = len(df2) - len(df) if 'df' in locals() else 0
                    st.metric("Différence", f"{diff:+d} ({(diff/len(df)*100):+.1f}%)" if 'df' in locals() else "N/A")
                
                if st.button("🚀 Lancer Phase 2", type="primary", use_container_width=True):
                    # Split
                    train_texts2, test_texts2, train_labels2, test_labels2 = train_test_split(
                        df2['Tweet'].tolist(), df2['Class'].tolist(),
                        test_size=0.2, random_state=42, stratify=df2['Class']
                    )
                    train_ds2 = Dataset.from_dict({'text': train_texts2, 'label': train_labels2})
                    test_ds2 = Dataset.from_dict({'text': test_texts2, 'label': test_labels2})
                    
                    params = {
                        'num_epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'max_length': max_length,
                        'warmup_steps': PARAMS['warmup_steps'],
                        'weight_decay': PARAMS['weight_decay']
                    }
                    
                    progress_container = st.empty()
                    
                    result2 = train_single_model(
                        st.session_state.best_model['model_name'],
                        st.session_state.best_model['model_path'],
                        train_ds2, test_ds2, params, progress_container
                    )
                    
                    if result2:
                        st.session_state.phase2_result = result2
                        
                        # Sauvegarder
                        model_dir = save_model_to_disk(result2, save_dir, "phase2")
                        st.session_state.saved_models.append({
                            'phase': 'Phase 2',
                            'model_name': result2['model_name'],
                            'path': model_dir,
                            'f1': result2['f1'],
                            'timestamp': datetime.now()
                        })
                        
                        st.success("✅ Phase 2 terminée!")
                        st.balloons()
        
        # Afficher comparaison
        if st.session_state.phase2_result:
            st.markdown("---")
            st.subheader("📊 Comparaison Phase 1 vs Phase 2")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Phase 1 F1", f"{st.session_state.best_model['f1']*100:.2f}%")
            with col2:
                st.metric("Phase 2 F1", f"{st.session_state.phase2_result['f1']*100:.2f}%")
            with col3:
                diff = (st.session_state.phase2_result['f1'] - st.session_state.best_model['f1']) * 100
                st.metric("Amélioration", f"{diff:+.2f}%")
            
            # Graphique
            fig = go.Figure(data=[go.Bar(
                x=['Phase 1', 'Phase 2'],
                y=[st.session_state.best_model['f1']*100, st.session_state.phase2_result['f1']*100],
                marker_color=['lightblue', 'gold'],
                text=[f"{st.session_state.best_model['f1']*100:.2f}%", 
                      f"{st.session_state.phase2_result['f1']*100:.2f}%"],
                textposition='auto'
            )])
            fig.update_layout(title="Évolution F1-Score", yaxis_title="F1 (%)", height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("👈 Complétez d'abord la Phase 1")

# TAB 4: Testeur
with tabs[3]:
    st.header("🧪 Testeur")
    
    model_to_use = st.session_state.phase2_result if st.session_state.phase2_result else st.session_state.best_model
    
    if model_to_use:
        st.info(f"Modèle: {model_to_use['model_name']} (F1: {model_to_use['f1']*100:.2f}%)")
        
        test_text = st.text_area("Texte en wolof:", placeholder="Ex: Nanga def?")
        
        if st.button("Analyser"):
            if test_text.strip():
                result = predict_text(test_text, model_to_use['trainer'], model_to_use['tokenizer'], max_length)
                
                if result['label'] == 'Abusif':
                    st.error(f"⚠️ {result['label']} ({result['confidence']:.1f}%)")
                else:
                    st.success(f"✅ {result['label']} ({result['confidence']:.1f}%)")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Non-abusif", f"{result['prob_non_abusif']:.1f}%")
                    st.progress(result['prob_non_abusif']/100)
                with col2:
                    st.metric("Abusif", f"{result['prob_abusif']:.1f}%")
                    st.progress(result['prob_abusif']/100)
    else:
        st.warning("Entraînez d'abord un modèle")

# TAB 5: Modèles sauvegardés
with tabs[4]:
    st.header("💾 Modèles Sauvegardés")
    
    if st.session_state.saved_models:
        for model_info in st.session_state.saved_models:
            with st.expander(f"{model_info['phase']} - {model_info['model_name']} (F1: {model_info['f1']*100:.2f}%)"):
                st.write(f"**Chemin:** `{model_info['path']}`")
                st.write(f"**Sauvegardé:** {model_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Bouton de téléchargement
                if st.button(f"📥 Créer ZIP", key=f"zip_{model_info['path']}"):
                    with st.spinner("Création du ZIP..."):
                        zip_path = create_zip_download(model_info['path'])
                        with open(zip_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Télécharger ZIP",
                                data=f,
                                file_name=os.path.basename(zip_path),
                                mime='application/zip'
                            )
    else:
        st.info("Aucun modèle sauvegardé pour le moment")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
    🇸🇳 Détection de Discours Haineux Wolof | Modèles automatiquement sauvegardés
    </div>
""", unsafe_allow_html=True)