import mne
import numpy as np

def load_and_preprocess_data(subject_id=1):
    """
    Loads PhysioNet EEG data for a specific subject (1 to 109).
    If a list is provided, it concatenates multiple subjects (Big Data mode).
    """
    runs = [4, 8, 12]  # Motor imagery: left vs right hand
    
    # Eğer subject_id bir liste ise (Birden fazla denek seçildiyse)
    if isinstance(subject_id, list):
        raws = []
        for sid in subject_id:
            try:
                raw_fnames = mne.datasets.eegbci.load_data(sid, runs)
                raws.extend([mne.io.read_raw_edf(f, preload=True, verbose='ERROR') for f in raw_fnames])
            except:
                print(f"[WARNING] Subject {sid} data not found, skipping...")
        raw = mne.concatenate_raws(raws)
    else:
        # Tek bir denek seçildiyse
        raw_fnames = mne.datasets.eegbci.load_data(subject_id, runs)
        raws = [mne.io.read_raw_edf(f, preload=True, verbose='ERROR') for f in raw_fnames]
        raw = mne.concatenate_raws(raws)

    mne.datasets.eegbci.standardize(raw)
    raw.filter(8., 30., fir_design='firwin', skip_by_annotation='edge', verbose='ERROR')
    
    events, _ = mne.events_from_annotations(raw, verbose='ERROR')
    event_id = dict(left=2, right=3)
    
    # Sadece Motor Korteks (Central) kanalları
    picks = mne.pick_channels_regexp(raw.ch_names, '^C.*|^FC.*|^CP.*')
    
    epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=3.5, 
                        picks=picks, baseline=None, preload=True, verbose='ERROR')
    
    return epochs.get_data(copy=False), epochs.events[:, -1]