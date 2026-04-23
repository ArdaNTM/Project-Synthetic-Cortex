import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, AveragePooling2D, BatchNormalization, DepthwiseConv2D, SeparableConv2D, Activation, Reshape, GlobalAveragePooling2D, Multiply
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# --- DİNAMİK DOSYA YOLU BULUCU ---
# engine.py dosyasının bulunduğu 'src' klasörünü bulur
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
# Bir üst klasöre ('Project Synthetic Cortex v3.0' klasörüne) çıkar
PROJECT_DIR = os.path.dirname(BASE_DIR) 
# Modeli her zaman ana proje klasörünün içine kaydetmesi için rotayı çizer
DEFAULT_MODEL_PATH = os.path.join(PROJECT_DIR, "global_model.keras")

def create_bci_pipeline(chans=21, samples=321):
    """
    Yeni Nesil BCI Mimarisi: EEGNet + Spatial/Channel Attention Mechanism
    (Squeeze-and-Excitation bloğu entegre edilmiştir).
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    
    # Keras Sequential yerine, esnek Model API'ye geçiyoruz
    inputs = Input(shape=(chans, samples))
    x = Reshape((chans, samples, 1))(inputs)
    
    # 1. Blok: Zamansal Evrişim (Temporal Convolution)
    x = Conv2D(16, (1, 64), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    
    # 2. Blok: Uzamsal Evrişim (Spatial Convolution)
    x = DepthwiseConv2D((chans, 1), use_bias=False, depth_multiplier=2)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(0.3)(x)
    
    # 3. Blok: Ayrılabilir Evrişim (Separable Convolution)
    x = SeparableConv2D(16, (1, 16), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(0.3)(x)
    
    # --- YENİ NESİL: ATTENTION (SQUEEZE-AND-EXCITATION) BLOĞU ---
    # Bu blok, yapay zekanın sadece o an ateşlenen ilgili beyin lobuna 
    # (ve zaman dilimine) "Dikkat kesilmesini" sağlar.
    filters = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // 2, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    x = Multiply()([x, se]) 
    # -------------------------------------------------------------
    
    # Sınıflandırma Katmanı
    x = Flatten()(x)
    outputs = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, X_train, y_train):
    """Sıfırdan Big Data veya Global model eğitimi."""
    y_train_mapped = y_train - 2 
    y_train_cat = to_categorical(y_train_mapped, 2)
    
    print("\n[AI CORE] Deep Learning Training Initiated (From Scratch)...")
    model.fit(X_train, y_train_cat, epochs=30, batch_size=16, verbose=1)
    print("[AI CORE] Neural Network Weights Updated Successfully!\n")
    return model

# --- ÖĞRENME TRANSFERİ YETENEKLERİ (TRANSFER LEARNING) ---

def save_bci_model(model, filepath=DEFAULT_MODEL_PATH):
    """Global modeli her zaman projenin ana klasörüne kaydeder."""
    model.save(filepath)
    print(f"[AI CORE] Global Model Master Knowledge saved to {filepath}")

def load_bci_model(filepath=DEFAULT_MODEL_PATH):
    """Daha önce kaydedilmiş Global modeli çağırır."""
    if os.path.exists(filepath):
        print(f"[AI CORE] Loading Pre-Trained Master Model from {filepath}")
        return load_model(filepath)
    return None

# src/engine.py içindeki fine_tune_model fonksiyonunu şu şekilde güncelle:

def fine_tune_model(model, X_train, y_train):
    """Önceden eğitilmiş modeli alır ve kişiselleştirir."""
    y_train_mapped = y_train - 2 
    y_train_cat = to_categorical(y_train_mapped, 2)
    
    # Öğrenme hızını biraz daha artırıyoruz (0.001)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("\n[AI CORE] Transfer Learning: Fine-Tuning on Personal Data...")
    
    # 'validation_split=0.2' ekleyerek modelin eğitim sırasında 
    # kendi kendini test etmesini sağlıyoruz.
    model.fit(X_train, y_train_cat, 
              epochs=30, # Epoch sayısını artırdık
              batch_size=16, 
              validation_split=0.2, # Verinin %20'sini test için ayır
              verbose=1) 
    
    print("[AI CORE] Personal Fine-Tuning Complete!\n")
    return model

# ---------------------------------------------------

def get_prediction(model, single_epoch):
    """Canlı beyin dalgasını analiz edip karar verir."""
    probabilities = model.predict(single_epoch, verbose=0)[0]
    predicted_class = np.argmax(probabilities) + 2 
    confidence = np.max(probabilities) * 100
    return predicted_class, confidence