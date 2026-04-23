import mne
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

def load_and_preprocess_data(subject_list, window_size=2.0, step_size=0.5):
    """
    Kayan pencereler (Sliding Windows) ile veri setini çoğaltır.
    """
    raws = []
    if not isinstance(subject_list, list):
        subject_list = [subject_list]
        
    for item in subject_list:
        if isinstance(item, int):
            runs = [4, 8, 12]
            try:
                raw_fnames = mne.datasets.eegbci.load_data(item, runs)
                raws.extend([mne.io.read_raw_edf(f, preload=True, verbose='ERROR') for f in raw_fnames])
            except Exception as e:
                print(f"[WARNING] PhysioNet Subject {item} failed: {e}")
        elif isinstance(item, str) and os.path.exists(item):
            try:
                raws.append(mne.io.read_raw_edf(item, preload=True, verbose='ERROR'))
            except Exception as e:
                print(f"[WARNING] Custom file {item} failed: {e}")

    if not raws:
        raise ValueError("Yüklenecek geçerli bir veri bulunamadı!")

    raw = mne.concatenate_raws(raws)
    mne.datasets.eegbci.standardize(raw)
    raw.filter(8., 30., fir_design='firwin', skip_by_annotation='edge', verbose='ERROR')
    
    events, _ = mne.events_from_annotations(raw, verbose='ERROR')
    event_id = dict(left=2, right=3)
    picks = mne.pick_channels_regexp(raw.ch_names, '^C.*|^FC.*|^CP.*')
    
    # --- KAYAN PENCERELER (AUGMENTATION) ---
    sfreq = raw.info['sfreq']
    win_samples = int(window_size * sfreq)
    step_samples = int(step_size * sfreq)
    
    X_list, y_list = [], []
    
    # 0 ile 3.5 saniye arasında pencereleri kaydırarak örnek topluyoruz
    for start in np.arange(0.0, 1.5, step_size): # 0.0, 0.5, 1.0 saniyelerinden başla
        tmin, tmax = start, start + window_size
        epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, 
                            picks=picks, baseline=None, preload=True, verbose='ERROR')
        X_list.append(epochs.get_data(copy=False))
        y_list.append(epochs.events[:, -1])
        
    # Tüm pencereleri tek bir büyük veri setinde birleştir
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    return X, y

# Dosyanın en üstündeki importlara şunları eklediğinden emin ol:
# import warnings
# warnings.filterwarnings("ignore") # MOABB bazen gereksiz uyarılar verir, onları sustururuz.

def load_moabb_data(subject_list, window_size=2.0, step_size=0.1):
    """
    Dünya standartlarındaki BNCI_2014_001 veri setini MOABB üzerinden indirir,
    işler ve Kayan Pencereler (Sliding Windows) ile çoğaltır.
    """
    from moabb.datasets import BNCI2014_001 # Alt çizgi eklendi
    from moabb.paradigms import MotorImagery
    import mne
    
    print("\n[DATA CORE] Accessing MOABB Global Database (BNCI 2014-001)...")
    
    # Sadece Sağ ve Sol el hayallerini alıyoruz
    paradigm = MotorImagery(n_classes=2, events=['left_hand', 'right_hand'])
    dataset = BNCI2014_001() # Alt çizgi eklendi
    
    if not isinstance(subject_list, list):
        subject_list = [subject_list]
        
    # Sadece 1'den 9'a kadar denekleri var
    valid_subjects = [s for s in subject_list if 1 <= s <= 9]
    if not valid_subjects:
         raise ValueError("MOABB BNCI veri seti sadece 1-9 arası denekleri destekler!")
         
    # Veriyi MOABB üzerinden otomatik indir ve parse et
    X_raw, labels, metadata = paradigm.get_data(dataset=dataset, subjects=valid_subjects)
    
    # Etiketleri bizim AI motorumuza uyarlıyoruz: left_hand -> 2, right_hand -> 3
    y_raw = np.array([2 if label == 'left_hand' else 3 for label in labels])
    
    # BNCI verisi 250Hz'dir, bizim model 160Hz bekliyor, yeniden örnekleyelim (Resampling)
    # Epoch başına 321 örnek kalacak şekilde ayarlıyoruz
    X_resampled = mne.filter.resample(X_raw, up=160, down=250) 
    
    # Veriyi tam 321 uzunluğunda keselim (Engine.py'deki input_shape ile tam uyuşması için)
    X_final = X_resampled[:, :21, :321] # Sadece ilk 21 kanal ve 321 zaman adımı
    
    print(f"[DATA CORE] MOABB Data Loaded! Shape: {X_final.shape}")
    return X_final, y_raw