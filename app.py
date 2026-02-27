import streamlit as st
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import io
import os
import warnings
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import roc_curve, auc
from scipy.stats import skew, kurtosis
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mne.decoding import CSP
import joblib
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
try:
    from pyriemann.transfer import TSSpawn
except ImportError:
    TSSpawn = None
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

class FBCSP(BaseEstimator, TransformerMixin):
    """
    Filter-Bank Common Spatial Pattern (FBCSP) transformer.
    Applies CSP to data filtered into multiple frequency bands.
    This is a state-of-the-art technique for motor imagery classification.
    """
    def __init__(self, sfreq, filter_bands=None, n_components=4, reg='ledoit_wolf', log=True, adaptive=False):
        self.sfreq = sfreq
        self.filter_bands = filter_bands
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.adaptive = adaptive
        self.csps_ = []
        self.selected_bands_ = []

    def fit(self, X, y):
        if self.adaptive:
            # Generate candidate bands (4-40 Hz, 4Hz width, 2Hz overlap)
            self.selected_bands_ = [(f, f+4) for f in range(4, 38, 2)]
        else:
            self.selected_bands_ = self.filter_bands

        self.csps_ = [CSP(n_components=self.n_components, reg=self.reg, log=self.log, norm_trace=False)
                      for _ in self.selected_bands_]

        features_list = []
        for i, (fmin, fmax) in enumerate(self.selected_bands_):
            X_filtered = mne.filter.filter_data(X.copy(), self.sfreq, l_freq=fmin, h_freq=fmax, verbose=False, n_jobs=1)
            self.csps_[i].fit(X_filtered, y)
            if self.adaptive:
                features_list.append(self.csps_[i].transform(X_filtered))
        
        if self.adaptive:
            X_features = np.concatenate(features_list, axis=1)
            # Select top features using Mutual Information
            selector = SelectKBest(mutual_info_classif, k=min(self.n_components * 2, X_features.shape[1]))
            selector.fit(X_features, y)
            selected_indices = selector.get_support(indices=True)
            selected_band_indices = np.unique(selected_indices // self.n_components)
            self.selected_bands_ = [self.selected_bands_[i] for i in selected_band_indices]
            self.csps_ = [self.csps_[i] for i in selected_band_indices]
            
        return self

    def transform(self, X):
        all_features = []
        for i, (fmin, fmax) in enumerate(self.selected_bands_):
            X_filtered = mne.filter.filter_data(X.copy(), self.sfreq, l_freq=fmin, h_freq=fmax, verbose=False, n_jobs=1)
            features = self.csps_[i].transform(X_filtered)
            all_features.append(features)
        return np.concatenate(all_features, axis=1)

# Page Configuration
st.set_page_config(
    page_title="NeuroFlow Research Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utility Functions ---
def load_eeg(uploaded_file, downsample_freq=None):
    """
    Save uploaded file to temp and load with MNE.
    Returns raw object and temp path (for cleanup).
    """
    if uploaded_file is None:
        return None, None
    
    # MNE requires a file path, so we write the upload to a temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tfile.write(uploaded_file.read())
    tfile.close()
    
    try:
        # Attempt to read based on extension, defaulting to EDF for this example
        if uploaded_file.name.lower().endswith('.edf'):
            raw = mne.io.read_raw_edf(tfile.name, preload=True, verbose=False)
        elif uploaded_file.name.lower().endswith('.fif'):
            raw = mne.io.read_raw_fif(tfile.name, preload=True, verbose=False)
        else:
            st.error("Unsupported file format. Please upload .edf or .fif")
            return None, tfile.name
        
        # Clean channel names (remove dots, 'EEG ', '-REF') and set montage
        def clean_name(name):
            return name.replace('.', '').replace('EEG ', '').replace('-REF', '').replace('POL ', '').replace('-LE', '').strip()
        
        mapping = {ch: clean_name(ch) for ch in raw.ch_names}
        raw.rename_channels(mapping)
        try:
            montage = mne.channels.make_standard_montage('standard_1005')
            
            # Auto-detect EEG channels if they match standard names
            # This fixes issues where channels are marked as 'misc' or 'unknown'
            standard_chs_upper = {ch.upper() for ch in montage.ch_names}
            ch_types = {ch: 'eeg' for ch in raw.ch_names if ch.upper() in standard_chs_upper}
            
            if ch_types:
                try:
                    raw.set_channel_types(ch_types, on_unit_change='ignore')
                except Exception:
                    pass

            # Capture warnings when setting montage
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                raw.set_montage(montage, match_case=False, on_missing='warn')
                if w:
                    # Store warnings in session state to display them later
                    st.session_state.montage_warnings = [str(warn.message) for warn in w]
        except Exception:
            pass # Continue without montage if matching fails

        # Efficiency: Downsample if requested
        if downsample_freq and downsample_freq < raw.info['sfreq']:
            try:
                raw.resample(downsample_freq, verbose=False)
            except Exception:
                pass

        return raw, tfile.name
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, tfile.name

def plot_psd_comparison(raws, labels):
    """Plot PSD comparison for multiple raw objects."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for raw, label in zip(raws, labels):
        # Calculate PSD
        spectrum = raw.compute_psd(fmin=1, fmax=40, n_fft=256, verbose=False)
        psds, freqs = spectrum.get_data(return_freqs=True)
        # Average across channels
        psd_mean = psds.mean(axis=0)
        # Convert to dB
        psd_db = 10 * np.log10(psd_mean)
        ax.plot(freqs, psd_db, label=label)
    
    ax.set_title("Power Spectral Density Comparison")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def calculate_band_powers(raw):
    """Calculate relative band powers."""
    bands = {'Delta (0.5-4 Hz)': (0.5, 4), 'Theta (4-8 Hz)': (4, 8),
             'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30),
             'Gamma (30-45 Hz)': (30, 45)}
    
    spectrum = raw.compute_psd(fmin=0.5, fmax=45, n_fft=256, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    # Normalize PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)
    
    band_powers = {}
    for band, (fmin, fmax) in bands.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_powers[band] = psds[:, idx].mean()
        
    return band_powers

# --- Sidebar Navigation ---
st.sidebar.title("🧠 NeuroFlow")
module = st.sidebar.radio(
    "Select Module:",
    [
        "1. Session Comparison",
        "2. Anomaly Detection",
        "3. Annotation Tool",
        "4. BCI Classification",
        "5. Preprocessing (ICA)",
        "6. BCI Inference",
        "7. ERP Analysis"
    ]
)
st.sidebar.info("v1.0 Research Build")
downsample_rate = st.sidebar.number_input("Downsample Data (Hz)", min_value=0, value=100, help="0 to disable. Lower values speed up processing.")
ds_freq = downsample_rate if downsample_rate > 0 else None

# --- Module 1: Session Comparison ---
if module == "1. Session Comparison":
    st.header("📊 Module 1: Session Comparison")
    st.markdown("Upload multiple EEG sessions (e.g., Pre-medication vs Post-medication) to compare spectral power.")
    
    uploaded_files = st.file_uploader("Upload EEG Files (.edf)", accept_multiple_files=True, type=['edf', 'fif'])
    
    if uploaded_files and len(uploaded_files) > 0:
        raw_objects = []
        labels = []
        
        for u_file in uploaded_files:
            raw, tpath = load_eeg(u_file, downsample_freq=ds_freq)
            if raw:
                raw_objects.append(raw)
                labels.append(u_file.name)
                # Clean up temp file
                os.unlink(tpath)
        
        if len(raw_objects) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Spectral Comparison")
                fig = plot_psd_comparison(raw_objects, labels)
                st.pyplot(fig)
                
                st.subheader("Topographic Activity (Alpha Band)")
                # Plot topomap for the first file as an example
                if raw_objects[0].get_montage():
                    fig_topo, ax_topo = plt.subplots(figsize=(5, 5))
                    spectrum = raw_objects[0].compute_psd(fmin=8, fmax=12, verbose=False)
                    spectrum.plot_topomap(ch_type='eeg', axes=ax_topo, show=False, bands={'Alpha (8-12 Hz)': (8, 12)})
                    st.pyplot(fig_topo)
                else:
                    st.warning("No standard montage found. Cannot plot topomaps.")
            
            with col2:
                st.subheader("Session Details")
                all_bands = []
                for i, raw in enumerate(raw_objects):
                    with st.expander(f"Session: {labels[i]}"):
                        st.write(f"**Channels:** {len(raw.ch_names)}")
                        st.write(f"**Duration:** {raw.times[-1]:.2f}s")
                        
                        bands = calculate_band_powers(raw)
                        df_bands = pd.DataFrame(list(bands.items()), columns=['Band', 'Power'])
                        st.dataframe(df_bands, hide_index=True)
                        all_bands.append(bands)
                
                # Comparison Chart
                if len(all_bands) > 1:
                    st.subheader("Band Power Trend")
                    df_compare = pd.DataFrame(all_bands, index=labels)
                    st.bar_chart(df_compare)
                    
                    # Innovation: Statistical Test
                    if len(all_bands) == 2:
                        st.subheader("Statistical Significance (t-test)")
                        st.caption("Comparing first two sessions.")
                        # This is a simplified t-test on the aggregated band powers (heuristic)
                        # In a real scenario, you would t-test the epochs or raw segments.
                        t_stat, p_val = ttest_ind(list(all_bands[0].values()), list(all_bands[1].values()))
                        st.metric("P-Value (Band Power Dist.)", f"{p_val:.4f}", delta="Significant" if p_val < 0.05 else "Not Significant")
                
                # --- Time-Frequency Analysis ---
                st.write("---")
                st.subheader("Time-Frequency Analysis (First Channel)")
                if st.button("Compute TFR (Morlet Wavelets)"):
                    for i, raw in enumerate(raw_objects):
                        # Create fixed length epochs for TFR to handle continuous data
                        epochs_tfr = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True, verbose=False)
                        freqs = np.logspace(*np.log10([4, 40]), num=15)
                        n_cycles = freqs / 2.
                        
                        # Compute TFR on first channel only for speed
                        power = tfr_morlet(epochs_tfr, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                           return_itc=False, decim=2, n_jobs=1, picks=[0], average=True, verbose=False)
                        
                        st.write(f"**{labels[i]}**")
                        fig_tfr, ax_tfr = plt.subplots(figsize=(10, 4))
                        power.plot([0], baseline=(None, None), mode='logratio', axes=ax_tfr, show=False, colorbar=True)
                        st.pyplot(fig_tfr)

# --- Module 2: Anomaly Detection ---
elif module == "2. Anomaly Detection":
    st.header("⚠️ Module 2: Anomaly Detection")
    st.markdown("AI-powered detection using Isolation Forest to find statistical outliers in signal variance.")
    
    uploaded_file = st.file_uploader("Upload EEG File for Screening", type=['edf', 'fif'])
    
    if uploaded_file:
        raw, tpath = load_eeg(uploaded_file, downsample_freq=ds_freq)
        if raw:
            # 1. Feature Extraction (Sliding Window)
            window_size = 1.0 # seconds
            sfreq = raw.info['sfreq']
            data = raw.get_data()
            n_samples = data.shape[1]
            samples_per_window = int(window_size * sfreq)
            
            features = []
            time_indices = []
            
            # Calculate variance and peak-to-peak for every window
            for start in range(0, n_samples, samples_per_window):
                end = start + samples_per_window
                if end <= n_samples:
                    window = data[:, start:end]
                    # Feature: Mean Variance across channels
                    var = np.mean(np.var(window, axis=1))
                    # Feature: Max Peak-to-Peak amplitude
                    ptp = np.max(np.ptp(window, axis=1))
                    # Feature: Skewness (asymmetry of the signal)
                    skw = np.mean(np.abs(skew(window, axis=1)))
                    # Feature: Kurtosis (tailedness of the signal)
                    krt = np.mean(kurtosis(window, axis=1))
                    features.append([var, ptp, skw, krt])
                    time_indices.append(raw.times[start])
            
            X = np.array(features)
            
            # Scale features for more robust outlier detection
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 2. AI Model: Isolation Forest
            contamination = st.slider("Sensitivity (Contamination)", 0.01, 0.2, 0.05)
            clf = IsolationForest(contamination=contamination, random_state=42)
            preds = clf.fit_predict(X_scaled)  # -1 is anomaly, 1 is normal
            
            anomalies = np.where(preds == -1)[0]
            
            st.metric("Anomalies Detected", len(anomalies))
            
            if len(anomalies) > 0:
                st.error(f"Abnormal patterns detected in {len(anomalies)} windows.")
                
                # Plot
                fig, ax = plt.subplots(figsize=(12, 4))
                # Plot first channel for reference
                ax.plot(raw.times, data[0, :]*1e6, color='black', linewidth=0.5, alpha=0.6, label='EEG Signal')
                
                # Highlight anomaly windows
                for idx in anomalies:
                    start_t = time_indices[idx]
                    ax.axvspan(start_t, start_t + window_size, color='red', alpha=0.3)
                
                ax.set_title("AI Anomaly Detection (Red zones indicate statistical outliers)")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude (µV)")
                ax.legend()
                st.pyplot(fig)
                
                # Innovation: PCA Visualization of Anomalies
                st.subheader("Feature Space Visualization (PCA)")
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                fig_pca = plt.figure(figsize=(8, 6))
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=preds, cmap='coolwarm', alpha=0.7, edgecolor='k')
                plt.title("PCA Projection of Signal Windows")
                plt.xlabel("Principal Component 1")
                plt.ylabel("Principal Component 2")
                # Legend: -1 (Red) is Anomaly, 1 (Blue) is Normal
                st.pyplot(fig_pca)
            else:
                st.success("No anomalies detected above threshold.")
            
            os.unlink(tpath)

# --- Module 3: Annotation Tool ---
elif module == "3. Annotation Tool":
    st.header("✍️ Module 3: Annotation Tool")
    st.markdown("Manually mark segments for AI training data.")
    
    uploaded_file = st.file_uploader("Upload EEG to Annotate", type=['edf', 'fif'])
    
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []

    if uploaded_file:
        raw, tpath = load_eeg(uploaded_file, downsample_freq=ds_freq)
        if raw:
            # Interactive inputs
            col1, col2, col3 = st.columns(3)
            with col1:
                start_time = st.number_input("Start Time (s)", min_value=0.0, max_value=raw.times[-1])
            with col2:
                duration = st.number_input("Duration (s)", min_value=0.1, value=1.0)
            with col3:
                label_type = st.selectbox("Label", ["Seizure", "Artifact", "Sleep_Stage_1", "Normal"])
            
            if st.button("Add Annotation"):
                st.session_state.annotations.append({
                    "onset": start_time,
                    "duration": duration,
                    "description": label_type,
                    "file": uploaded_file.name
                })
                st.success(f"Added {label_type} at {start_time}s")
            
            # Show current annotations
            if st.session_state.annotations:
                st.subheader("Current Labels")
                df_annot = pd.DataFrame(st.session_state.annotations)
                st.dataframe(df_annot)
                
                # Download button
                csv = df_annot.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Labels CSV",
                    csv,
                    "eeg_labels.csv",
                    "text/csv",
                    key='download-csv'
                )
            
            # Interactive Visualization using Plotly
            st.subheader("Interactive Signal Preview")
            data, times = raw.get_data(start=int(start_time*raw.info['sfreq']),
                                     stop=int((start_time+duration+5)*raw.info['sfreq']),
                                     return_times=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=data[0, :]*1e6, mode='lines', name='EEG Channel 1'))
            # Add highlight shape
            fig.add_vrect(x0=start_time, x1=start_time+duration, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Selection")
            fig.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude (µV)", height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            os.unlink(tpath)

# --- Module 4: BCI Classification ---
@st.cache_data
def get_top_k_channels(_raw, events, event_id, k):
    """
    Selects the top K channels based on mutual information with the labels.
    This uses simple band-power features for speed.
    """
    epochs = mne.Epochs(_raw, events, event_id, tmin=0, tmax=4, preload=True, verbose=False, baseline=None)
    epochs.pick_types(eeg=True)
    
    # Simple band-power features (alpha and beta)
    # Use get_data() and np.var directly to avoid issues with apply_function return shapes/types
    alpha_epochs = epochs.copy().filter(8, 12, verbose=False)
    alpha_features = np.var(alpha_epochs.get_data(copy=False), axis=-1)
    
    beta_epochs = epochs.copy().filter(12, 30, verbose=False)
    beta_features = np.var(beta_epochs.get_data(copy=False), axis=-1)
    
    features = np.hstack([alpha_features, beta_features]) # Shape: (n_epochs, 2 * n_channels)

    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(features, epochs.events[:, -1])
    
    # The selector gives indices of features, not channels. We need to map back.
    # The features are [alpha_ch1, alpha_ch2, ..., beta_ch1, beta_ch2, ...].
    n_eeg_channels = len(epochs.ch_names)
    selected_feature_indices = selector.get_support(indices=True)
    
    # A feature's channel index can be found with modulo n_channels.
    # Use np.unique to get each channel index only once.
    selected_channel_indices = np.unique(selected_feature_indices % n_eeg_channels)
    
    # Return the names of the selected channels from the epochs object, not the raw object.
    return [epochs.ch_names[i] for i in selected_channel_indices]

if module == "4. BCI Classification":
    st.header("Module 4: BCI Task Classification 🧩")
    st.markdown("Train and evaluate BCI models. Upload a training session, and optionally a second session to test generalization.")

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        uploaded_file = st.file_uploader("Upload Training Data", type=['edf', 'fif'], key="train")
    with col_u2:
        test_file = st.file_uploader("Upload Test Data (Optional)", type=['edf', 'fif'], key="test", help="Upload a separate session to test session-to-session transfer.")

    if uploaded_file:
        raw, tpath = load_eeg(uploaded_file, downsample_freq=ds_freq)
        
        raw_test = None
        tpath_test = None
        if test_file:
            raw_test, tpath_test = load_eeg(test_file, downsample_freq=ds_freq)
            if raw_test:
                st.success(f"Test session loaded: {test_file.name}")

        if raw:
            # Display montage warnings if any
            if 'montage_warnings' in st.session_state and st.session_state.montage_warnings:
                with st.expander("Montage Warnings (some channels could not be located)"):
                    st.warning(
                        "The following channels in the data file could not be matched to standard "
                        "locations. This prevents topographic map visualization."
                    )
                    st.json(st.session_state.montage_warnings)
                # Clear after showing
                del st.session_state.montage_warnings

            st.info("Extracting events and creating epochs...")

            # Apply 1Hz high-pass filter to remove drift (crucial for CSP/Riemannian)
            # We must apply this to BOTH train and test data to avoid domain shift.
            raw.filter(l_freq=1.0, h_freq=None, fir_design='firwin', verbose=False)
            if raw_test:
                raw_test.filter(l_freq=1.0, h_freq=None, fir_design='firwin', verbose=False)
            
            try:
                # Extract events
                events, event_id = mne.events_from_annotations(raw)
                st.write(f"Found {len(events)} events: {event_id}")

                # --- Advanced Settings ---
                with st.expander("⚙️ Advanced Model Settings", expanded=True):
                    st.caption("Adjust these to improve accuracy. Try cropping the time window to 0.5s - 2.5s.")
                    col1, col2 = st.columns(2)
                    
                    clf_options = ["SVM (RBF) + FBCSP", "LDA + FBCSP", "Adaptive FBCSP + SVM", "CSP + LDA (Baseline)", "Riemannian MDM", "Riemannian Tangent Space"]
                    if TSSpawn is not None:
                        clf_options.append("Riemannian + TSSpawn (Adaptation)")
                    clf_options.append("Compare All Classifiers")

                    clf_type = col2.selectbox(
                        "Classifier Method",
                        clf_options,
                        index=0,
                        help="Select a model. 'TSSpawn' is for adapting to a new session with a few calibration trials."
                    )

                    if "TSSpawn" in clf_type:
                        n_calib_trials = st.slider("Calibration Trials per Class", 2, 10, 5, help="Number of trials from the start of the test set to use for adapting the model.")

                    with col1:
                        t_start = st.number_input("Epoch Start (s)", value=0.5, step=0.1, help="Start time relative to event. Motor imagery usually starts ~0.5s after cue.")
                        t_end = st.number_input("Epoch End (s)", value=2.5, step=0.1, help="End time relative to event.")
                        
                        # Show settings based on selection
                        show_bands = ("FBCSP" in clf_type and "Adaptive" not in clf_type) or clf_type == "Compare All Classifiers"
                        show_filter = "Riemannian" in clf_type or "CSP" in clf_type or clf_type == "Compare All Classifiers"

                        if show_bands:
                            freq_bands_str = st.text_input("Frequency Bands (Hz)", "8-12, 12-16, 16-24, 24-30", help="Comma-separated frequency bands for FBCSP.")
                        if show_filter:
                            f_low = st.number_input("Filter Low (Hz)", value=4.0, step=1.0)
                            f_high = st.number_input("Filter High (Hz)", value=40.0, step=1.0)

                    with col2:
                        st.info("Hyperparameters for SVM and Tangent Space models are tuned automatically using GridSearchCV for optimal performance.")
                    
                    # --- Channel Selection ---
                    with st.expander("📡 Channel Selection Strategy (Crucial for Performance)", expanded=True):
                        st.markdown(
                            "**Why this matters:** Using all channels can introduce noise and lead to overfitting. "
                            "A focused channel set often yields better, more generalizable models. This section "
                            "provides a framework for a channel reduction study."
                        )
                        ch_selection_method = st.radio(
                            "Method:",
                            ["Automatic (Top K)", "Manual"],
                            horizontal=True,
                            help="**Automatic:** Ranks channels by feature importance and selects the best K. **Manual:** You choose the channels."
                        )
                        if ch_selection_method == "Automatic (Top K)":
                            k_channels = st.slider("Number of Channels (K)", min_value=4, max_value=len(raw.ch_names), value=8, step=2)
                        else:
                            # Smart defaults for Motor Imagery
                            all_chs = raw.ch_names
                            motor_chs = ['C3', 'C4', 'Cz', 'FC3', 'FC4', 'CP3', 'CP4', 'C1', 'C2', 'C5', 'C6', 'FCz', 'CPz']
                            default_chs = [ch for ch in all_chs if ch in motor_chs]
                            if not default_chs: default_chs = all_chs[:min(len(all_chs), 8)] # Fallback
                            selected_channels = st.multiselect("Select Channels:", options=all_chs, default=default_chs)

                # Allow user to select specific events to classify
                # This is crucial because 'Rest' (T0) often confuses models trained on T1/T2
                selected_event_names = st.multiselect(
                    "Select events to classify (select at least 2):",
                    options=list(event_id.keys()),
                    default=list(event_id.keys())
                )
                
                if len(selected_event_names) < 2:
                    st.warning(
                        "Please select at least 2 classes to train the model."
                    )
                else:
                    # Filter event_id based on selection
                    selected_event_id = {name: event_id[name] for name in selected_event_names}

                    tmin, tmax = t_start, t_end
                    epochs = mne.Epochs(
                        raw,
                        events,
                        selected_event_id,
                        tmin,
                        tmax,
                        proj=True,
                        baseline=None, # No baseline correction if we crop start > 0
                        preload=True,
                        verbose=False
                    )
                    epochs.pick_types(eeg=True, meg=False, stim=False, eog=False, exclude='bads')

                    # --- Apply Channel Selection ---
                    if ch_selection_method == "Automatic (Top K)":
                        with st.spinner(f"Selecting Top {k_channels} channels..."):
                            try:
                                top_k_chs = get_top_k_channels(raw, events, selected_event_id, k=k_channels)
                                selected_channels = top_k_chs
                                st.success(f"Selected channels: {', '.join(selected_channels)}")
                            except Exception as e:
                                st.error(f"Automatic channel selection failed: {e}. Using all channels.")
                                selected_channels = raw.ch_names
                    
                    # Filter epochs based on selected channels
                    if 'selected_channels' in locals() and selected_channels:
                        valid_chs = [ch for ch in selected_channels if ch in epochs.ch_names]
                        if len(valid_chs) >= 2:
                            epochs.pick_channels(valid_chs)
                        else:
                            st.warning("Selected fewer than 2 valid EEG channels. Reverting to all channels.")
                    
                    # --- Handle Mismatched Channels for Cross-Session ---
                    if raw_test:
                        st.info("Cross-session analysis enabled. Finding common channels...")

                    # --- DIAGNOSTICS ---
                    if len(epochs.ch_names) == 0:
                        st.error("No EEG channels found! Check channel names or montage.")
                        st.stop()
                    
                    with st.expander("📊 Data Quality & Class Balance Checks", expanded=True):
                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            st.write(f"**Channels ({len(epochs.ch_names)}):** {', '.join(epochs.ch_names[:5])}...")
                            st.write(f"**Sampling Rate:** {epochs.info['sfreq']} Hz")
                        with col_d2:
                            # Class balance
                            counts = pd.Series(epochs.events[:, -1]).value_counts()
                            id_to_name = {v: k for k, v in selected_event_id.items()}
                            counts.index = [id_to_name.get(i, i) for i in counts.index]
                            st.write("**Class Counts:**")
                            st.dataframe(counts, height=100)
                        
                        # ERP Plot to check timing/signal
                        if st.checkbox("Show Class Averages (ERP)"):
                            fig_erp, ax_erp = plt.subplots(figsize=(10, 3))
                            for label_name, label_id in selected_event_id.items():
                                if label_id in epochs.events[:, -1]:
                                    evoked = epochs[label_name].average()
                                    ax_erp.plot(evoked.times, evoked.data.mean(axis=0)*1e6, label=label_name)
                            ax_erp.legend()
                            ax_erp.set_title("Class Averages (Mean of all channels)")
                            ax_erp.set_xlabel("Time (s)")
                            st.pyplot(fig_erp)

                    # Prepare Test Data if available
                    epochs_test = None
                    if raw_test:
                        try:
                            events_test, event_id_test = mne.events_from_annotations(raw_test, verbose=False)
                            # Ensure same events exist
                            if all(k in event_id_test for k in selected_event_id):
                                epochs_test = mne.Epochs(
                                    raw_test, events_test, selected_event_id, tmin, tmax,
                                    proj=True, baseline=None, preload=True, verbose=False
                                )
                                epochs_test.pick_types(eeg=True, meg=False, stim=False, eog=False, exclude='bads')
                                # Ensure same channels
                                common_chs = list(set(epochs.ch_names) & set(epochs_test.ch_names))
                                if len(common_chs) < 2:
                                     st.error("Training and Test sets have fewer than 2 common channels after selection.")
                                else:
                                     epochs.pick_channels(common_chs)
                                     epochs_test.pick_channels(common_chs)
                                     st.success(f"Using {len(common_chs)} common channels for transfer learning.")
                            else:
                                st.warning("Test file does not contain the same event types as training file.")
                        except Exception as e:
                            st.warning(f"Could not process test file: {e}")

                    # --- Model Training and Evaluation ---
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    
                    # Define models to run
                    models_to_run = []
                    if clf_type == "Compare All Classifiers":
                        models_to_run = [
                            ("CSP + LDA (Baseline)", "csp_lda"),
                            ("SVM (RBF) + FBCSP", "svm_fbcsp"),
                            ("Adaptive FBCSP + SVM", "adaptive_fbcsp"),
                            ("LDA + FBCSP", "lda_fbcsp"),
                            ("Riemannian MDM", "rm_mdm"),
                            ("Riemannian Tangent Space", "rm_ts")
                        ]
                    else:
                        models_to_run = [(clf_type, "selected")]

                    results_data = []
                    best_model_obj = None
                    best_model_name = ""
                    best_acc = 0.0
                    
                    # Get data once
                    X_train = epochs.get_data(copy=True)
                    
                    st.success(f"✅ Training model using **{len(epochs.ch_names)} channels**: {', '.join(epochs.ch_names)}")
                    
                    y_train = epochs.events[:, -1]
                    
                    X_test = epochs_test.get_data(copy=True) if epochs_test else None
                    y_test = epochs_test.events[:, -1] if epochs_test else None

                    progress_bar = st.progress(0)
                    
                    for idx, (model_name, _) in enumerate(models_to_run):
                        with st.spinner(f"Training {model_name}..."):

                            # --- Special Case: TSSpawn for Domain Adaptation ---
                            if "TSSpawn" in model_name:
                                if X_test is None:
                                    st.warning("TSSpawn requires a separate test file for adaptation. Skipping this model.")
                                    results_data.append({"Model": model_name, "CV Accuracy": "N/A", "CV Std": "N/A", "Session Transfer Acc": "N/A"})
                                    continue

                                # 1. Prepare data
                                X_train_filt = mne.filter.filter_data(X_train, raw.info['sfreq'], f_low, f_high, verbose=False)
                                X_test_filt = mne.filter.filter_data(X_test, raw.info['sfreq'], f_low, f_high, verbose=False)

                                # 2. Split test set into calibration and final test sets
                                calib_indices = []
                                final_test_indices = list(range(len(y_test)))
                                for label in np.unique(y_test):
                                    label_indices = np.where(y_test == label)[0]
                                    if len(label_indices) > n_calib_trials:
                                        calib_indices.extend(label_indices[:n_calib_trials])
                                    else: # Not enough trials, use all for calibration
                                        calib_indices.extend(label_indices)
                                
                                # Create final test set by removing calibration trials
                                final_test_indices = np.setdiff1d(np.arange(len(y_test)), calib_indices)

                                if not calib_indices:
                                    st.error("Not enough trials in the test set for calibration. Need at least one trial per class.")
                                    continue

                                X_calib, y_calib = X_test_filt[calib_indices], y_test[calib_indices]
                                X_test_final, y_test_final = X_test_filt[final_test_indices], y_test[final_test_indices]

                                st.info(f"TSSpawn: Using {len(X_calib)} trials for adaptation and {len(X_test_final)} for final testing.")

                                # 3. Define and run the adaptation pipeline
                                cov = Covariances(estimator='lwf')
                                ts_spawn = TSSpawn(reg=0.1)
                                ts = TangentSpace()
                                lr = LogisticRegression(solver='lbfgs')
                                
                                cov_train = cov.fit_transform(X_train_filt, y_train)
                                cov_calib = cov.transform(X_calib)

                                X_spawned, y_spawned = ts_spawn.fit_transform(cov_train, y_train, Xt=cov_calib)
                                
                                clf = Pipeline([('ts', ts), ('lr', lr)])
                                clf.fit(X_spawned, y_spawned)
                                
                                test_acc = clf.score(cov.transform(X_test_final), y_test_final) if len(X_test_final) > 0 else np.nan
                                results_data.append({"Model": model_name, "CV Accuracy": "N/A", "CV Std": "N/A", "Session Transfer Acc": test_acc})
                                continue # Go to next model
                            
                            # 1. Data Prep based on model type
                            if "Riemannian" in model_name or "CSP + LDA" in model_name:
                                # Filter data for these models
                                X_curr = mne.filter.filter_data(X_train, raw.info['sfreq'], f_low, f_high, verbose=False, copy=True)
                                X_curr_test = mne.filter.filter_data(X_test, raw.info['sfreq'], f_low, f_high, verbose=False, copy=True) if X_test is not None else None
                            else:
                                # FBCSP handles filtering
                                X_curr = X_train
                                X_curr_test = X_test

                            # 2. Pipeline Construction
                            clf = None
                            if model_name == "CSP + LDA (Baseline)":
                                csp = CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False)
                                lda = LinearDiscriminantAnalysis()
                                clf = Pipeline([('CSP', csp), ('LDA', lda)])
                                
                            elif "Riemannian MDM" in model_name:
                                clf = Pipeline([('cov', Covariances(estimator='lwf')), ('mdm', MDM())])
                                
                            elif "Riemannian Tangent Space" in model_name:
                                # Use GridSearchCV for Logistic Regression to improve Kappa
                                ts_clf = Pipeline([('cov', Covariances(estimator='lwf')), ('ts', TangentSpace()), ('lr', LogisticRegression(solver='lbfgs', max_iter=1000))])
                                clf = GridSearchCV(ts_clf, {'lr__C': [0.1, 1.0, 10.0]}, cv=3, n_jobs=1)
                                
                            elif "FBCSP" in model_name:
                                # Parse bands
                                bands = []
                                try:
                                    for band in freq_bands_str.split(','):
                                        parts = [float(p.strip()) for p in band.split('-')]
                                        if len(parts) == 2: bands.append(tuple(parts))
                                except:
                                    bands = [(8,12), (12,16), (16,24), (24,30)]
                                
                                is_adaptive = "Adaptive" in model_name
                                fbcsp = FBCSP(sfreq=raw.info['sfreq'], filter_bands=bands, adaptive=is_adaptive)
                                scaler = StandardScaler()
                                if "SVM" in model_name:
                                    model_obj = SVC(kernel='rbf', probability=True)
                                    clf = Pipeline([('FBCSP', fbcsp), ('Scaler', scaler), ('Model', model_obj)])
                                    # Simplified grid for speed in comparison mode
                                    if clf_type != "Compare All Classifiers":
                                        param_grid = {'FBCSP__n_components': [4], 'Model__C': [1, 10]}
                                        clf = GridSearchCV(clf, param_grid, cv=3, n_jobs=1)
                                else: # LDA
                                    model_obj = LinearDiscriminantAnalysis()
                                    clf = Pipeline([('FBCSP', fbcsp), ('Scaler', scaler), ('Model', model_obj)])

                            # 3. Training & CV
                            scores = cross_val_score(clf, X_curr, y_train, cv=cv, n_jobs=1)
                            mean_acc = np.mean(scores)
                            std_acc = np.std(scores)
                            
                            # Fit on full train data
                            clf.fit(X_curr, y_train)
                            
                            # Show selected bands for Adaptive FBCSP
                            if "Adaptive" in model_name:
                                fbcsp_step = clf.best_estimator_.named_steps['FBCSP'] if hasattr(clf, 'best_estimator_') else clf.named_steps['FBCSP']
                                st.caption(f"Adaptive FBCSP selected {len(fbcsp_step.selected_bands_)} bands: {fbcsp_step.selected_bands_}")
                            
                            # 4. Session-to-Session Test
                            test_acc = np.nan
                            if X_curr_test is not None:
                                test_acc = clf.score(X_curr_test, y_test)

                            results_data.append({
                                "Model": model_name,
                                "CV Accuracy": mean_acc,
                                "CV Std": std_acc,
                                "Session Transfer Acc": test_acc if X_curr_test is not None else "N/A"
                            })

                            if mean_acc > best_acc:
                                best_acc = mean_acc
                                best_model_obj = clf
                                best_model_name = model_name
                                # Store predictions for the best model for confusion matrix
                                if "GridSearchCV" in str(type(clf)):
                                    best_est = clf.best_estimator_
                                else:
                                    best_est = clf
                                y_pred = cross_val_predict(best_est, X_curr, y_train, cv=cv, n_jobs=1)

                        progress_bar.progress((idx + 1) / len(models_to_run))

                    progress_bar.empty()
                    
                    # --- Results Section ---
                    st.markdown("### 📈 Results & Analysis")
                    
                    # 1. Comparison Table
                    df_results = pd.DataFrame(results_data)
                    df_results = df_results.sort_values(by="CV Accuracy", ascending=False)
                    st.dataframe(df_results.style.format({"CV Accuracy": "{:.2%}", "CV Std": "{:.2%}", "Session Transfer Acc": "{:.2%}" if X_test is not None else "{}"}))
                    
                    # Innovation: Export results for publication
                    csv = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results Table (CSV)",
                        csv,
                        "bci_model_comparison.csv",
                        "text/csv",
                        key='download-results-csv',
                        help="Download this table to include in your research paper."
                    )

                    st.success(f"Best Model: **{best_model_name}** with **{best_acc:.2%}** CV Accuracy.")

                    # --- Confusion Matrix & Report (Restored) ---
                    st.subheader(f"Detailed Analysis ({best_model_name})")
                    col_cm, col_rep = st.columns(2)
                    with col_cm:
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_train, y_pred, labels=np.unique(y_train))
                        fig_cm = plt.figure(figsize=(5, 4))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                    xticklabels=list(selected_event_id.keys()), 
                                    yticklabels=list(selected_event_id.keys()))
                        plt.ylabel('True Label')
                        plt.xlabel('Predicted Label')                        
                        st.pyplot(fig_cm)
                        
                        # Feature for IEEE Publication: High-Res Download
                        buf = io.BytesIO()
                        fig_cm.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                        st.download_button(
                            label="Download High-Res Figure (300 DPI)",
                            data=buf.getvalue(),
                            file_name="confusion_matrix.png",
                            mime="image/png",
                            help="Download this figure for your IEEE paper."
                        )
                    
                    with col_rep:
                        st.subheader("Classification Report")
                        report = classification_report(y_train, y_pred, target_names=list(selected_event_id.keys()), output_dict=True)
                        df_report = pd.DataFrame(report).transpose()
                        st.dataframe(df_report.style.format("{:.2f}"))
                        kappa = cohen_kappa_score(y_train, y_pred)
                        st.metric("Cohen's Kappa", f"{kappa:.2f}", help="A robust statistic for inter-rater agreement. 0.6-0.8 is substantial, >0.8 is near perfect.")

                    # --- ROC Curve (for binary classification) ---
                    unique_labels = np.unique(y_train)
                    if len(unique_labels) == 2:
                        st.subheader("ROC Curve (Binary)")
                        try:
                            # Get probabilities for the positive class
                            if hasattr(best_model_obj, "predict_proba"):
                                y_prob = cross_val_predict(best_model_obj, X_curr, y_train, cv=cv, method='predict_proba', n_jobs=1)[:, 1]
                                fpr, tpr, _ = roc_curve(y_train, y_prob, pos_label=unique_labels[1])
                                roc_auc = auc(fpr, tpr)
                                fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
                                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                                ax_roc.set_xlabel('False Positive Rate')
                                ax_roc.set_ylabel('True Positive Rate')
                                ax_roc.legend(loc="lower right")
                                st.pyplot(fig_roc)
                                
                                # IEEE Download: ROC
                                buf_roc = io.BytesIO()
                                fig_roc.savefig(buf_roc, format="png", dpi=300, bbox_inches='tight')
                                st.download_button(
                                    label="Download ROC Curve (300 DPI)",
                                    data=buf_roc.getvalue(),
                                    file_name="roc_curve.png",
                                    mime="image/png"
                                )
                        except Exception as e:
                            st.info(f"ROC Curve not available: {e}")

                    # Helper to get the best pipeline from GridSearchCV or Pipeline
                    if hasattr(best_model_obj, 'best_estimator_'):
                        best_pipeline = best_model_obj.best_estimator_
                    else:
                        best_pipeline = best_model_obj

                    # Re-prepare data for visualization if needed (filtering)
                    if "Riemannian" in best_model_name or "CSP + LDA" in best_model_name:
                         X_vis = mne.filter.filter_data(X_train, raw.info['sfreq'], f_low, f_high, verbose=False, copy=True)
                    else:
                         X_vis = X_train

                    # --- Model Interpretation ---
                    st.subheader("Model Interpretation")

                    if "Riemannian" in best_model_name:
                        st.info("For Riemannian classifiers, the most informative features are the covariance matrices, which capture the spatial relationships between EEG channels for each class.")
                        with st.spinner("Calculating covariance matrices..."):
                            cov_estimator = Covariances(estimator='lwf')
                            covs = cov_estimator.fit_transform(X_vis, y_train)
                            
                            unique_labels = np.unique(y_train)
                            id_to_name = {v: k for k, v in selected_event_id.items()}
                            class_names = [id_to_name.get(l, f'ID {l}') for l in unique_labels]                            
                            
                            n_classes = len(unique_labels)
                            fig_cov, axes_cov = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4), squeeze=False, layout="constrained")
                            
                            im = None
                            for i, label_val in enumerate(unique_labels):
                                class_covs = covs[y_train == label_val]
                                mean_cov = np.mean(class_covs, axis=0)
                                ax = axes_cov[0, i]
                                im = ax.imshow(mean_cov, cmap='viridis', interpolation='nearest')
                                ax.set_title(f"Class: {class_names[i]}")
                                ax.set_xlabel("Channel Index")
                            
                            axes_cov[0, 0].set_ylabel("Channel Index")
                            fig_cov.colorbar(im, ax=axes_cov.ravel().tolist(), shrink=0.8, label="Covariance")
                            st.pyplot(fig_cov)
                            plt.close(fig_cov)

                    elif "CSP" in best_model_name: # Catches FBCSP and CSP
                        st.info("CSP patterns are spatial filters optimized to maximize variance for one class while minimizing it for another, revealing the most discriminative brain areas.")
                        
                        # --- Interpretation Guide ---
                        id_to_name = {v: k for k, v in selected_event_id.items()}
                        # Ensure class names match the internal sort order of CSP (by label ID)
                        unique_labels = np.unique(y_train)
                        class_names_ordered = [id_to_name.get(l, f"Class {l}") for l in unique_labels]

                        if len(class_names_ordered) == 2:
                            st.markdown(f"""
                            **How to Interpret CSP Patterns:**
                            - **The Plot:** Represents the head viewed from above (nose at the top). The dots are electrodes.
                            - **The Colors (Red/Blue):** These show the **weights** (importance) of each electrode.
                                - **Strong Red or Blue:** High importance. The AI is focusing on these areas.
                                - **White/Faded:** Low importance. These areas are ignored.
                                - *Note: Red (+) and Blue (-) are equally important in brainwaves; they just indicate opposite phases (like a battery's + and -).*
                            - **The Order:**
                                - The **first** patterns (e.g., `CSP-1`) are optimized to detect **`{class_names_ordered[0]}`**.
                                - The **last** patterns (e.g., `CSP-4`) are optimized to detect **`{class_names_ordered[1]}`**.
                            """)

                        try:
                            if epochs.get_montage() is None:
                                st.warning("⚠️ No channel montage found. Cannot plot topographic maps. Please ensure channel names match standard 10-20 system (e.g., C3, C4, Cz).")
                                raise ValueError("No montage")

                            # FBCSP has multiple sets of patterns for different bands
                            if "FBCSP" in best_model_name:
                                fbcsp = best_pipeline.named_steps['FBCSP']
                                band_opts = [f"{b[0]}-{b[1]} Hz" for b in fbcsp.selected_bands_]
                                if not band_opts:
                                    st.warning("FBCSP model has no selected frequency bands to display.")
                                else:
                                    st.write("#### FBCSP Patterns")
                                    selected_band_idx = st.selectbox("Select Frequency Band for Patterns", range(len(band_opts)), format_func=lambda x: band_opts[x])
                                    csp_model = fbcsp.csps_[selected_band_idx]
                                    title = f"Patterns: '{class_names_ordered[0]}' vs '{class_names_ordered[1]}'" if len(class_names_ordered) == 2 else "FBCSP Patterns"
                                    fig_patterns = csp_model.plot_patterns(epochs.info, ch_type='eeg', units='AU', size=1.5)
                                    fig_patterns.suptitle(title)
                                    st.pyplot(fig_patterns)
                                    plt.close(fig_patterns)
                            
                            # Regular CSP
                            else:
                                st.write("#### Common Spatial Patterns")
                                csp_model = best_pipeline.named_steps['CSP']
                                title = f"Patterns: '{class_names_ordered[0]}' vs '{class_names_ordered[1]}'" if len(class_names_ordered) == 2 else "CSP Patterns"
                                fig_patterns = csp_model.plot_patterns(epochs.info, ch_type='eeg', units='AU', size=1.5)
                                fig_patterns.suptitle(title)
                                st.pyplot(fig_patterns)
                                plt.close(fig_patterns)

                            # --- NEW: 3D Source Visualization ---
                            if st.checkbox("Show 3D Source Localization of CSP Patterns (Experimental)"):
                                with st.spinner("Setting up 3D environment (may download template MRI)..."):
                                    try:
                                        fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
                                        src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
                                        bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
                                        
                                        fwd = mne.make_forward_solution(epochs.info, trans='fsaverage', src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=1, verbose=False)
                                        
                                        # Compute noise covariance (try baseline, fallback to data)
                                        try:
                                            # Attempt to use pre-stimulus baseline for noise covariance
                                            cov_epochs = mne.Epochs(raw, events, selected_event_id, tmin=-0.5, tmax=0, baseline=None, preload=True, verbose=False)
                                            cov_epochs.pick_types(eeg=True, meg=False, stim=False, eog=False, exclude='bads')
                                            cov_epochs.pick_channels(epochs.ch_names)
                                            noise_cov = mne.compute_covariance(cov_epochs, method='shrunk', rank=None, verbose=False)
                                        except Exception:
                                            # Fallback: use the task epochs themselves
                                            noise_cov = mne.compute_covariance(epochs, method='shrunk', rank=None, verbose=False)
                                            
                                        inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, forward=fwd, noise_cov=noise_cov, loose=0.2, depth=0.8, verbose=False)

                                        csp_model_3d = csp_model # Use the model selected above
                                        
                                        st.write(f"Visualizing sources for **'{class_names_ordered[0]}'** (first components) vs. **'{class_names_ordered[1]}'** (last components)")
                                        
                                        # This returns a pyvista Brain object
                                        brain = csp_model_3d.plot_sources(epochs.info, inverse_operator, n_components=2, verbose=False)
                                        
                                        # Get a screenshot to display in Streamlit
                                        screenshot = brain.screenshot()
                                        brain.close() # Close the pyvista plotter

                                        st.image(screenshot, caption="3D Source Localization of the first two and last two CSP components. Red indicates areas of high contribution.")

                                    except Exception as e:
                                        st.error(f"Could not generate 3D plot: {e}")
                                        st.info("This feature is experimental and requires 'pyvista' and a successful download of the 'fsaverage' template MRI. Please ensure you have a stable internet connection and that channel names match a standard montage.")

                        except Exception as e:
                            st.warning(f"Could not plot CSP patterns: {e}")

                    elif "Tangent Space" in best_model_name:
                        st.subheader("Tangent Space Feature Weights")
                        st.info("Visualizing the weights of the Logistic Regression classifier in the Tangent Space. High absolute weights indicate important covariance features.")
                        try:
                            if hasattr(best_model_obj, 'best_estimator_'):
                                est = best_model_obj.best_estimator_
                            else:
                                est = best_model_obj
                            
                            lr = est.named_steps['lr']
                            coef = lr.coef_
                            
                            fig_weights, ax_weights = plt.subplots(figsize=(10, 5))
                            if coef.shape[0] == 1:
                                ax_weights.plot(coef[0])
                                ax_weights.set_title("Feature Weights (Binary)")
                            else:
                                ax_weights.plot(coef.T)
                                ax_weights.set_title("Feature Weights (Multiclass)")
                            st.pyplot(fig_weights)
                        except Exception as e:
                            st.warning(f"Could not visualize weights: {e}")

                    if "FBCSP" in best_model_name or "CSP" in best_model_name:
                        st.subheader("Feature Importance (CSP Components)")
                        st.info("Analyzing which CSP features (Frequency Band + Spatial Filter) contribute most to the classification.")
                        try:
                            # 1. Extract features using the transformer part of the pipeline
                            # We need to separate transformer and classifier
                            if "FBCSP" in best_model_name:
                                transformer = best_pipeline.named_steps['FBCSP']
                                classifier = best_pipeline.named_steps['Model']
                                # If scaler exists
                                if 'Scaler' in best_pipeline.named_steps:
                                    scaler = best_pipeline.named_steps['Scaler']
                                    full_transformer = Pipeline([('FBCSP', transformer), ('Scaler', scaler)])
                                else:
                                    full_transformer = transformer
                            else: # CSP + LDA
                                transformer = best_pipeline.named_steps['CSP']
                                classifier = best_pipeline.named_steps['LDA']
                                full_transformer = transformer

                            # Transform data to get features
                            X_features = full_transformer.transform(X_vis)
                            
                            # 2. Run Permutation Importance on the CLASSIFIER using FEATURES
                            result = permutation_importance(
                                classifier, X_features, y_train, n_repeats=5, random_state=42, n_jobs=1
                            )
                            
                            # 3. Create labels for features
                            if "FBCSP" in best_model_name:
                                feat_labels = []
                                for band in transformer.selected_bands_:
                                    for i in range(transformer.n_components):
                                        feat_labels.append(f"{band[0]}-{band[1]}Hz CSP{i+1}")
                            else:
                                feat_labels = [f"CSP{i+1}" for i in range(transformer.n_components)]

                            # 4. Plot
                            importance_df = pd.DataFrame({
                                'Feature': feat_labels[:X_features.shape[1]], 
                                'Importance': result.importances_mean
                            })
                            importance_df = importance_df.sort_values(by='Importance', ascending=False).head(15)
                            
                            fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
                            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax_importance, palette='viridis')
                            ax_importance.set_title("Top Discriminative Features")
                            st.pyplot(fig_importance)
                            
                        except Exception as e:
                            st.error(f"Could not calculate feature importance: {e}")
                    else:
                        st.info("No specific interpretation plot available for this model type.")
                    
                    # --- Download Model (Restored) ---
                    st.subheader("Download Trained Model")
                    model_filename = f"neuroflow_model_{best_model_name.replace(' ', '_')}.joblib"
                    
                    # Save model with channel info
                    model_bundle = {
                        'model': best_model_obj,
                        'channels': epochs.ch_names
                    }
                    joblib.dump(model_bundle, model_filename)
                    with open(model_filename, "rb") as f:
                        st.download_button("Download Model (.joblib)", f, file_name=model_filename)

            except Exception as e:
                st.error(f"Could not process BCI data: {e}")

        if tpath and os.path.exists(tpath):
            os.unlink(tpath)
        if tpath_test and os.path.exists(tpath_test):
            os.unlink(tpath_test)

# --- Module 5: Preprocessing (ICA) ---
elif module == "5. Preprocessing (ICA)":
    st.header("🧼 Module 5: Artifact Removal (ICA)")
    st.markdown("Use Independent Component Analysis (ICA) to separate and remove artifacts (e.g., eye blinks, muscle noise) from the signal.")
    
    uploaded_file = st.file_uploader("Upload EEG File", type=['edf', 'fif'])
    
    if uploaded_file:
        raw, tpath = load_eeg(uploaded_file, downsample_freq=ds_freq)
        # Store original raw object in session state to compare against later
        # and reset state if a new file is uploaded.
        if 'ica_filename' not in st.session_state or st.session_state.get('ica_filename') != uploaded_file.name:
            st.session_state.ica_raw_orig = raw
            st.session_state.ica_filename = uploaded_file.name
            st.session_state.ica_applied = False # Reset on new file
        if raw:
            st.subheader("1. Signal Filtering")
            # Filter for ICA (1Hz highpass is recommended for ICA stability)
            raw_ica = raw.copy().filter(l_freq=1.0, h_freq=40.0, verbose=False)
            st.success("Data filtered (1-40 Hz) for ICA stability.")
            
            st.subheader("2. Run ICA")
            col1, col2 = st.columns(2)
            with col1:
                n_components = st.number_input("Number of Components", min_value=2, max_value=min(len(raw.ch_names), 20), value=15)
            with col2:
                method = st.selectbox("ICA Method", ["fastica", "picard"])
            
            if st.button("Fit ICA"):
                with st.spinner("Fitting ICA..."):
                    # Clear previous artifact indices to avoid range errors
                    for key in ['eog_indices', 'ecg_indices', 'mus_indices']:
                        st.session_state.pop(key, None)

                    ica = ICA(n_components=n_components, method=method, random_state=42)
                    ica.fit(raw_ica, verbose=False)
                    st.session_state['ica_obj'] = ica
                    st.success("ICA Fitted.")
            
            if 'ica_obj' in st.session_state:
                ica = st.session_state['ica_obj']
                
                # --- Automated EOG detection ---
                st.subheader("3. Automated Artifact Detection")
                
                col_art1, col_art2, col_art3 = st.columns(3)
                
                with col_art1:
                    if st.button("Find EOG (Blinks)"):
                        with st.spinner("Detecting EOG..."):
                            try:
                                # Try to find EOG channels automatically or use synthetic
                                eog_indices, eog_scores = ica.find_bads_eog(raw)
                                st.session_state['eog_indices'] = eog_indices
                                st.success(f"EOG Artifacts: {eog_indices}")
                            except Exception as e:
                                st.error(f"EOG detection failed: {e}")

                with col_art2:
                    if st.button("Find ECG (Heartbeat)"):
                        with st.spinner("Detecting ECG..."):
                            try:
                                ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
                                st.session_state['ecg_indices'] = ecg_indices
                                st.success(f"ECG Artifacts: {ecg_indices}")
                            except Exception as e:
                                st.error(f"ECG detection failed: {e}")

                with col_art3:
                    if st.button("Find Muscle Artifacts"):
                        with st.spinner("Detecting Muscle..."):
                            try:
                                mus_indices, mus_scores = ica.find_bads_muscle(raw)
                                st.session_state['mus_indices'] = mus_indices
                                st.success(f"Muscle Artifacts: {mus_indices}")
                            except Exception as e:
                                st.error(f"Muscle detection failed: {e}")

                st.subheader("4. Component Selection")
                st.write("Topomaps of Independent Components (Select artifacts to remove):")
                fig_ica = ica.plot_components(show=False)
                st.pyplot(fig_ica)
                
                # IEEE Download: ICA Components
                buf_ica = io.BytesIO()
                fig_ica.savefig(buf_ica, format="png", dpi=300, bbox_inches='tight')
                st.download_button(
                    label="Download ICA Components (300 DPI)",
                    data=buf_ica.getvalue(),
                    file_name="ica_components.png",
                    mime="image/png"
                )
                
                # Pre-select automatically found components
                default_exclude = list(set(
                    st.session_state.get('eog_indices', []) + 
                    st.session_state.get('ecg_indices', []) + 
                    st.session_state.get('mus_indices', [])
                ))
                exclude_indices = st.multiselect("Select components to exclude:", range(ica.n_components_), default=default_exclude)
                
                if st.button("Apply & Clean Data"):
                    ica.exclude = exclude_indices
                    raw_clean = raw.copy()
                    ica.apply(raw_clean)
                    st.session_state['ica_raw_clean'] = raw_clean
                    st.session_state['ica_applied'] = True
                    st.session_state.pop('cleaned_data_bytes', None) # Clear old cache
                
                # This block is now outside the button's if statement.
                # It will re-run when the selectbox value changes.
                if st.session_state.get('ica_applied', False):
                    st.subheader("5. Result Comparison")
                    raw_clean = st.session_state['ica_raw_clean']
                    raw_orig = st.session_state.ica_raw_orig

                    # Innovation: SNR Calculation
                    def calculate_snr(signal):
                        return np.mean(signal ** 2) / np.var(signal)
                    
                    snr_orig = calculate_snr(raw_orig.get_data())
                    snr_clean = calculate_snr(raw_clean.get_data())
                    st.metric("Signal Quality Improvement (SNR)", f"{snr_clean:.2f}", f"{snr_clean - snr_orig:.2f}")

                    # Add a channel selector to compare results
                    ch_to_plot = st.selectbox("Select a channel to compare:", raw_orig.ch_names)
                    ch_idx = raw_orig.ch_names.index(ch_to_plot)
                    
                    # Compare first 5 seconds for the selected channel
                    data_orig, times = raw_orig.get_data(picks=[ch_idx], start=0, stop=int(5*raw_orig.info['sfreq']), return_times=True)
                    data_clean, _ = raw_clean.get_data(picks=[ch_idx], start=0, stop=int(5*raw.info['sfreq']), return_times=True)
                    
                    fig_comp = go.Figure()
                    fig_comp.add_trace(go.Scatter(x=times, y=data_orig[0]*1e6, mode='lines', name='Original', line=dict(color='red', width=1.5)))
                    fig_comp.add_trace(go.Scatter(x=times, y=data_clean[0]*1e6, mode='lines', name='Cleaned (ICA Applied)', line=dict(color='blue', width=1.5)))
                    fig_comp.update_layout(title=f"Original vs. Cleaned Data for Channel: {ch_to_plot}",
                                           xaxis_title="Time (s)", yaxis_title="Amplitude (µV)", height=400)
                    st.plotly_chart(fig_comp, use_container_width=True)
            
            os.unlink(tpath)

# --- Module 6: BCI Inference ---
elif module == "6. BCI Inference":
    st.header("🚀 Module 6: BCI Inference")
    st.markdown("Load a previously trained model and apply it to new EEG data.")

    col1, col2 = st.columns(2)
    with col1:
        model_file = st.file_uploader("Upload Trained Model (.joblib)", type=['joblib'])
    with col2:
        eeg_file = st.file_uploader("Upload New EEG Data (.edf, .fif)", type=['edf', 'fif'])

    if model_file and eeg_file:
        try:
            # Load model
            loaded_obj = joblib.load(model_file)
            if isinstance(loaded_obj, dict) and 'model' in loaded_obj and 'channels' in loaded_obj:
                model = loaded_obj['model']
                model_channels = loaded_obj['channels']
                st.success(f"Model `{model_file.name}` loaded successfully.")
                st.info(f"Model was trained on **{len(model_channels)} channels**: {', '.join(model_channels)}")
            else:
                # Legacy model loading
                model = loaded_obj
                model_channels = None
                st.success(f"Legacy model `{model_file.name}` loaded successfully.")

                # Try to find the expected number of features for a better error message
                n_expected = None
                try:
                    pipeline_to_check = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
                    if 'FBCSP' in pipeline_to_check.named_steps:
                        fbcsp_step = pipeline_to_check.named_steps['FBCSP']
                        if fbcsp_step.csps_:
                            n_expected = fbcsp_step.csps_[0].n_features_in_
                    elif 'CSP' in pipeline_to_check.named_steps:
                        n_expected = pipeline_to_check.named_steps['CSP'].n_features_in_
                except Exception:
                    pass  # Can't determine it, will fail later with a less specific message

                if n_expected is not None:
                    st.error(
                        f"**Legacy Model File:** This model was trained on **{n_expected} channels**, but the channel names were not saved with it. "
                        "This app has been updated to save channel names to prevent this error.\n\n"
                        "**Action Required:** Please go back to **Module 4**, re-train your model, and use the newly downloaded model file for inference."
                    )
                    st.stop()
                else:
                    st.warning(
                        "Could not automatically determine channels used for training. The new data must have the exact same channels in the same order as the training data."
                    )

            # Load data
            raw, tpath = load_eeg(eeg_file, downsample_freq=ds_freq)
            if raw:
                st.write("---")
                st.subheader("Preprocessing Settings for Inference")
                st.warning("Ensure these settings match the ones used for training the model.")
                
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    t_start = st.number_input("Epoch Start (s)", value=0.5, step=0.1)
                    t_end = st.number_input("Epoch End (s)", value=2.5, step=0.1)
                with p_col2:
                    f_low = st.number_input("Filter Low (Hz)", value=8.0, step=1.0)
                    f_high = st.number_input("Filter High (Hz)", value=30.0, step=1.0)

                if st.button("Run Inference"):
                    with st.spinner("Preprocessing data and running predictions..."):
                        # Replicate preprocessing
                        events, event_id = mne.events_from_annotations(raw, verbose=False)
                        
                        # Conditionally filter based on model type
                        if 'FBCSP' not in model.named_steps:
                            st.write(f"Applying bandpass filter ({f_low}-{f_high} Hz)...")
                            raw.filter(f_low, f_high, fir_design='firwin', skip_by_annotation='edge', verbose=False)
                        else:
                            st.write("FBCSP model detected. Filtering will be handled by the model pipeline.")
                        
                        epochs = mne.Epochs(
                            raw, events, event_id, t_start, t_end,
                            proj=True, baseline=None, preload=True, verbose=False
                        )
                        epochs.pick_types(eeg=True, meg=False, stim=False, eog=False, exclude='bads')
                        
                        # --- CRITICAL FIX: Select channels based on the loaded model ---
                        if model_channels:
                            try:
                                st.write(f"Selecting the {len(model_channels)} channels the model was trained on...")
                                epochs.pick(model_channels)
                            except ValueError as e:
                                st.error(
                                    f"Channel mismatch: The model requires channels that are not in the new data file.\n"
                                    f"Details: {e}\n"
                                    f"Please ensure that the uploaded EEG data contains all the following channels: {model_channels}")
                                st.stop()
                        else:
                            st.warning("Attempting to use all available channels. This may fail if they don't match the training data.")
                        
                        data_to_predict = epochs.get_data()
                        
                        # Predict
                        predictions = model.predict(data_to_predict)
                        probabilities = model.predict_proba(data_to_predict)
                        
                        st.write("---")
                        st.subheader("Inference Results")
                        
                        # Get class names from the model's final estimator if available
                        try:
                            class_labels = model.classes_
                        except AttributeError:
                            # Fallback for pipelines where classes_ is not at the top level
                            class_labels = model.named_steps['Model'].classes_
                        
                        # Create a readable results dataframe
                        results_df = pd.DataFrame(probabilities, columns=[f"Prob({label})" for label in class_labels])
                        results_df['Predicted Label'] = predictions
                        results_df.index.name = "Trial Number"
                        
                        st.dataframe(results_df)
                        
                        # Plot probability distribution
                        fig_proba, ax_proba = plt.subplots()
                        results_df['Predicted Label'].value_counts().plot(kind='bar', ax=ax_proba)
                        ax_proba.set_title("Distribution of Predictions")
                        ax_proba.set_ylabel("Count")
                        ax_proba.set_xlabel("Predicted Class")
                        plt.xticks(rotation=45)
                        st.pyplot(fig_proba)

                        # Confusion Matrix
                        st.write("---")
                        with st.expander("Confusion Matrix & Metrics (Ground Truth Required)"):
                            y_true = epochs.events[:, -1]
                            
                            # Check for overlap between true labels and model classes
                            common_labels = np.intersect1d(np.unique(y_true), class_labels)
                            
                            if len(common_labels) > 0:
                                st.subheader("Confusion Matrix")
                                cm = confusion_matrix(y_true, predictions, labels=class_labels)
                                
                                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                            xticklabels=class_labels, 
                                            yticklabels=class_labels, ax=ax_cm)
                                ax_cm.set_ylabel('True Label (File Events)')
                                ax_cm.set_xlabel('Predicted Label')
                                st.pyplot(fig_cm)
                                
                                st.subheader("Classification Report")
                                report = classification_report(y_true, predictions, labels=class_labels, zero_division=0, output_dict=True)
                                st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))
                            else:
                                st.warning(f"No overlap between file event IDs {np.unique(y_true)} and model classes {class_labels}. Ensure event mapping is consistent.")

                os.unlink(tpath)

        except Exception as e:
            st.error(f"An error occurred during inference: {e}")
            if 'tpath' in locals() and os.path.exists(tpath):
                os.unlink(tpath)

# --- Module 7: ERP Analysis ---
elif module == "7. ERP Analysis":
    st.header("⚡ Module 7: Event-Related Potentials (ERP)")
    st.markdown("Analyze averaged EEG responses time-locked to specific events. This is essential for cognitive neuroscience research.")
    
    uploaded_file = st.file_uploader("Upload EEG File", type=['edf', 'fif'])
    
    if uploaded_file:
        raw, tpath = load_eeg(uploaded_file, downsample_freq=ds_freq)
        if raw:
            # Preprocessing for ERP
            with st.expander("Preprocessing & Epoching Settings", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    # MNE filter args: l_freq=highpass, h_freq=lowpass
                    hp_freq = st.number_input("High-pass Filter (Hz)", value=0.1, min_value=0.0, step=0.1, help="Removes slow drifts.")
                    lp_freq = st.number_input("Low-pass Filter (Hz)", value=30.0, min_value=1.0, step=1.0, help="Removes high-freq noise.")
                with col2:
                    tmin = st.number_input("Epoch Start (s)", value=-0.2, max_value=0.0, step=0.1)
                    tmax = st.number_input("Epoch End (s)", value=0.5, min_value=0.1, step=0.1)
            
            if st.button("Apply Filters"):
                raw.filter(l_freq=hp_freq, h_freq=lp_freq, verbose=False)
                st.success("Filters applied.")
                
            # Events
            try:
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                if len(events) > 0:
                    st.write(f"**Found Events:** {event_id}")
                    selected_events = st.multiselect("Select Events to Average", list(event_id.keys()), default=list(event_id.keys())[:2])
                    
                    if selected_events:
                        evokeds = {}
                        for event_name in selected_events:
                            eid = event_id[event_name]
                            # Create epochs and average them
                            epochs = mne.Epochs(raw, events, event_id={event_name: eid}, tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True, verbose=False)
                            evokeds[event_name] = epochs.average()
                        
                        # Plotting
                        st.subheader("ERP Visualization")
                        
                        # 1. Compare Evokeds (Butterfly)
                        st.write("### 1. Condition Comparison (Butterfly Plot)")
                        st.caption("Overlaid signals from all channels for selected conditions.")
                        fig_comp = mne.viz.plot_compare_evokeds(evokeds, combine='mean', show=False, legend='upper right')
                        final_fig = fig_comp[0] if isinstance(fig_comp, list) else fig_comp
                        final_fig = fig_comp[0] if isinstance(fig_comp, list) else fig_comp
                        st.pyplot(final_fig)
                        
                        # IEEE Download: ERP
                        buf_erp = io.BytesIO()
                        final_fig.savefig(buf_erp, format="png", dpi=300, bbox_inches='tight')
                        st.download_button(
                            label="Download ERP Plot (300 DPI)",
                            data=buf_erp.getvalue(),
                            file_name="erp_plot.png",
                            mime="image/png"
                        )

                        # 2. Joint Plot (Butterfly + Topo)
                        st.write("### 2. Spatio-Temporal Analysis (Joint Plot)")
                        st.caption("Shows the butterfly plot and topographic maps at peak latencies.")
                        for name, evk in evokeds.items():
                            st.write(f"**Condition: {name}**")
                            fig_joint = evk.plot_joint(times='peaks', show=False)
                            st.pyplot(fig_joint)
                else:
                    st.warning("No events found in data. Cannot compute ERPs.")
            except Exception as e:
                st.error(f"Error processing events: {e}")
            
            os.unlink(tpath)

st.sidebar.markdown("---")
st.sidebar.caption("NeuroFlow Research Platform")
