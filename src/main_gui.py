import pygame
import sys
import time
import random
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from utils import load_and_preprocess_data, load_moabb_data # KODUN EN BAŞINDAKİ İÇE AKTARMALARA ŞUNLARI EKLE/DEĞİŞTİR:
from engine import create_bci_pipeline, train_model, get_prediction, save_bci_model, load_bci_model, fine_tune_model
# --- DONANIM KÜTÜPHANESİ ---
try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.board_shim import BrainFlowError
    BRAINFLOW_INSTALLED = True
except ImportError:
    BRAINFLOW_INSTALLED = False

# ==========================================
# ⚙️ PYGAME INITIALIZATION & COLORS
# ==========================================
pygame.init()
WIDTH, HEIGHT = 950, 650
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Project Synthetic Cortex v3.1 - Advanced Neural OS")
clock = pygame.time.Clock()

BACKGROUND, GRID_COLOR = (8, 9, 13), (18, 20, 25)
WHITE, CYAN, MAGENTA = (240, 240, 245), (0, 255, 255), (255, 0, 127)
LIME, YELLOW, GRAY = (57, 255, 20), (255, 223, 0), (100, 105, 120)
DARK_PANEL = (12, 14, 18)

font_main = pygame.font.SysFont("segoeui", 20, bold=True)
font_title = pygame.font.SysFont("segoeui", 24, bold=True)
font_huge = pygame.font.SysFont("segoeui", 38, bold=True)
font_small = pygame.font.SysFont("consolas", 13)

def draw_text(text, x, y, font, color, align="left"):
    surf = font.render(text, True, color)
    if align == "center": rect = surf.get_rect(center=(x, y))
    elif align == "right": rect = surf.get_rect(topright=(x, y))
    else: rect = surf.get_rect(topleft=(x, y))
    screen.blit(surf, rect)

def draw_grid():
    for x in range(0, WIDTH, 40): pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, 40): pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y))

def draw_neon_button(surface, text, rect, font, color, hover, selected=False):
    alpha = 60 if hover or selected else 10
    s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    s.fill((*color, alpha))
    surface.blit(s, rect.topleft)
    border_color = LIME if selected else color
    pygame.draw.rect(surface, border_color, rect, 2 if selected else 1, border_radius=3)
    text_color = WHITE if hover or selected else color
    draw_text(text, rect.centerx, rect.centery, font, text_color, align="center")

def draw_real_eeg_signal(surface, signal_array, x_start, y_center, width, height, color):
    if len(signal_array) == 0: return
    points = []
    max_val = np.max(np.abs(signal_array)) 
    max_val = max_val if max_val != 0 else 1
    for i, val in enumerate(signal_array):
        x = x_start + (i / len(signal_array)) * width
        y = y_center - (val / max_val) * (height / 2)
        points.append((x, y))
    if len(points) > 1: pygame.draw.lines(surface, color, False, points, 2)

def open_file_dialog():
    """Kullanıcının yerel dosya seçmesi için Tkinter penceresi açar"""
    root = tk.Tk()
    root.withdraw() # Ana pencereyi gizle
    root.attributes('-topmost', True) # En üstte tut
    file_path = filedialog.askopenfilename(title="Select EEG File", filetypes=[("EDF Files", "*.edf"), ("All Files", "*.*")])
    root.destroy()
    return file_path

# ==========================================
# SYSTEM STATE VARIABLES
# ==========================================
state = 0 
OPERATION_MODE = ""

# --- Yeni Seçim Sistemi Değişkenleri ---
selected_physionet_ids = [] # Örn: [1, 2, 5]
custom_files = []           # Örn: ["C:/veri.edf"]
db_page = 0                 # PhysioNet ızgarası sayfa numarası
SUBJECTS_PER_PAGE = 24      # 30 yerine 24 yapıyoruz (3 sütun x 8 satır)
calib_idx, current_idx = 0, 0
last_action_time = time.time()
core_x, target_x = WIDTH//2, WIDTH//2
core_color = CYAN
status_msg, status_color = "AWAITING SIGNAL...", YELLOW
real_intent, confidence = "---", 0.0
score = 0
# YENİ OYUN DEĞİŞKENLERİ
asteroids = []
game_score = 0
collision_flash = 0
current_eeg_wave = []
hw_error_msg = "" 
display_mode = "GAME" # "CLASSIC" veya "GAME" olabilir

model = create_bci_pipeline()

while True:
    dt = clock.tick(60) / 1000.0
    mouse_pos = pygame.mouse.get_pos()
    mouse_click = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_click = True

    screen.fill(BACKGROUND)
    draw_grid()
    
    # ---------------------------------------------------
    # STATE 0: MAIN MENU 
    # ---------------------------------------------------
    if state == 0:
        draw_text("SYNTHETIC CORTEX", WIDTH//2, HEIGHT//3 - 60, font_huge, WHITE, align="center")
        draw_text("NEURAL OPERATING SYSTEM v3.1", WIDTH//2, HEIGHT//3 - 20, font_small, CYAN, align="center")

        btn1_rect = pygame.Rect(WIDTH//2 - 250, HEIGHT//2 - 30, 500, 60)
        draw_neon_button(screen, "DATABASE & CUSTOM DATA MODE", btn1_rect, font_main, CYAN, btn1_rect.collidepoint(mouse_pos))

        btn2_rect = pygame.Rect(WIDTH//2 - 250, HEIGHT//2 + 50, 500, 60)
        draw_neon_button(screen, "LIVE SENSOR MODE (HARDWARE)", btn2_rect, font_main, MAGENTA, btn2_rect.collidepoint(mouse_pos))

        if mouse_click:
            if btn1_rect.collidepoint(mouse_pos):
                OPERATION_MODE = "DATABASE"
                state = 0.1 
            elif btn2_rect.collidepoint(mouse_pos):
                OPERATION_MODE = "HARDWARE"
                hw_error_msg = ""
                state = 0.2 

    # ---------------------------------------------------
    # STATE 0.1: HYBRID DATA SELECTION (GRID UI)
    # ---------------------------------------------------
    elif state == 0.1:
        # Üst Panel (Silinen Geri Dön Butonu Onarıldı)
        btn_back_rect = pygame.Rect(20, 20, 120, 35)
        draw_neon_button(screen, "< BACK", btn_back_rect, font_small, GRAY, btn_back_rect.collidepoint(mouse_pos))
        if mouse_click and btn_back_rect.collidepoint(mouse_pos): state = 0
        
        draw_text("SELECT SUBJECTS FOR TRAINING", WIDTH//2, 35, font_title, CYAN, align="center")

        # --- SOL PANEL: MOABB VERİTABANI ---
        pygame.draw.rect(screen, DARK_PANEL, (20, 80, 500, 480))
        draw_text("MOABB LIBRARY (BNCI 1-9)", 270, 100, font_main, WHITE, align="center")
        
        start_id = db_page * SUBJECTS_PER_PAGE + 1
        end_id = min(109, (db_page + 1) * SUBJECTS_PER_PAGE)
        
        col, row = 0, 0
        for subj_id in range(start_id, end_id + 1):
            x = 40 + (col * 150)
            y = 140 + (row * 40)
            subj_rect = pygame.Rect(x, y, 130, 30)
            is_selected = subj_id in selected_physionet_ids
            draw_neon_button(screen, f"Subject {subj_id}", subj_rect, font_small, CYAN, subj_rect.collidepoint(mouse_pos), is_selected)
            
            if mouse_click and subj_rect.collidepoint(mouse_pos):
                if is_selected: selected_physionet_ids.remove(subj_id)
                else: selected_physionet_ids.append(subj_id)
            
            col += 1
            if col > 2:
                col = 0
                row += 1

        # Sayfalama Butonları (Çift tıklama engelli tekil versiyon)
        btn_prev = pygame.Rect(40, 500, 100, 30)
        btn_next = pygame.Rect(370, 500, 100, 30)
        draw_neon_button(screen, "<< PREV", btn_prev, font_small, GRAY, btn_prev.collidepoint(mouse_pos))
        draw_neon_button(screen, "NEXT >>", btn_next, font_small, GRAY, btn_next.collidepoint(mouse_pos))
        draw_text(f"PAGE {db_page+1}/5", 270, 515, font_small, GRAY, align="center")
        
        if mouse_click:
            if btn_prev.collidepoint(mouse_pos) and db_page > 0: 
                db_page -= 1
                mouse_click = False 
            elif btn_next.collidepoint(mouse_pos) and db_page < 4: 
                db_page += 1
                mouse_click = False 

        # --- SAĞ PANEL: ÖZEL DOSYALAR (Custom Data) ---
        pygame.draw.rect(screen, DARK_PANEL, (540, 80, 390, 400))
        draw_text("USER ADDED DATA (.EDF)", 735, 100, font_main, WHITE, align="center")
        
        btn_add_file = pygame.Rect(635, 130, 200, 35)
        draw_neon_button(screen, "+ UPLOAD LOCAL FILE", btn_add_file, font_small, MAGENTA, btn_add_file.collidepoint(mouse_pos))
        
        if mouse_click and btn_add_file.collidepoint(mouse_pos):
            new_file = open_file_dialog()
            if new_file and new_file not in custom_files:
                custom_files.append(new_file)

        for i, filepath in enumerate(custom_files):
            filename = os.path.basename(filepath)
            draw_text(f"• {filename[:30]}...", 560, 190 + (i*25), font_small, WHITE)

        # --- BAŞLAT BUTONU ---
        total_selected = len(selected_physionet_ids) + len(custom_files)
        btn_start = pygame.Rect(540, 500, 390, 60)
        can_start = total_selected > 0
        draw_neon_button(screen, f"START SIMULATION ({total_selected} Selected)", btn_start, font_title, LIME if can_start else GRAY, btn_start.collidepoint(mouse_pos))
        
        if mouse_click and btn_start.collidepoint(mouse_pos) and can_start:
            state = 0.5
            last_action_time = time.time()

    # ---------------------------------------------------
    # STATE 0.2: REAL HARDWARE SETUP
    # ---------------------------------------------------
    elif state == 0.2:
        btn_back_rect = pygame.Rect(30, 30, 150, 40)
        draw_neon_button(screen, "< MAIN MENU", btn_back_rect, font_small, GRAY, btn_back_rect.collidepoint(mouse_pos))
        if mouse_click and btn_back_rect.collidepoint(mouse_pos): state = 0

        draw_text("HARDWARE SENSOR SETUP", WIDTH//2, HEIGHT//3 - 40, font_title, MAGENTA, align="center")
        draw_text("1. Connect the USB Dongle / Bluetooth to your PC.", WIDTH//2, HEIGHT//3 + 10, font_main, WHITE, align="center")
        draw_text("2. Turn on your EEG Headset.", WIDTH//2, HEIGHT//3 + 40, font_main, WHITE, align="center")
        draw_text("3. Ensure electrodes are placed on Motor Cortex (C3/C4).", WIDTH//2, HEIGHT//3 + 70, font_main, WHITE, align="center")
        
        btn_connect_rect = pygame.Rect(WIDTH//2 - 150, HEIGHT//2 + 50, 300, 50)
        draw_neon_button(screen, "CONNECT & VERIFY", btn_connect_rect, font_title, LIME, btn_connect_rect.collidepoint(mouse_pos))

        if hw_error_msg: draw_text(hw_error_msg, WIDTH//2, HEIGHT//2 + 120, font_small, MAGENTA, align="center")

        if mouse_click and btn_connect_rect.collidepoint(mouse_pos):
            if not BRAINFLOW_INSTALLED:
                hw_error_msg = "FATAL: 'brainflow' library missing. Run 'pip install brainflow'."
            else:
                hw_error_msg = "SCANNING PORTS..."
                draw_text(hw_error_msg, WIDTH//2, HEIGHT//2 + 120, font_small, YELLOW, align="center")
                pygame.display.flip()
                try:
                    params = BrainFlowInputParams()
                    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
                    board.prepare_session()
                    board.start_stream()
                    state = 1
                    last_action_time = time.time()
                except BrainFlowError:
                    hw_error_msg = "CONNECTION FAILED: NO DEVICE DETECTED."

   # ---------------------------------------------------
    # STATE 0.5: DATA LOADING / PREPROCESSING 
    # ---------------------------------------------------
    elif state == 0.5:
        draw_text("ACCESSING MOABB GLOBAL DATABASE...", WIDTH//2, HEIGHT//2 - 20, font_title, YELLOW, align="center")
        draw_text("Downloading High-Fidelity BNCI Data. This may take a moment.", WIDTH//2, HEIGHT//2 + 20, font_small, GRAY, align="center")
        pygame.display.flip()
        
        # Sadece 1-9 arası seçili denekleri filtrele (BNCI 9 kişiliktir)
        moabb_subjects = [s for s in selected_physionet_ids if s <= 9]
        if not moabb_subjects: moabb_subjects = [1] 
        
        # --- KRİTİK DÜZELTME: State 2'nin çökmemesi için bu değişkeni tanımlıyoruz ---
        all_subjects_to_load = moabb_subjects
        
        try:
            X, y = load_moabb_data(subject_list=moabb_subjects)
            X_calib, X_test, y_calib, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
            calib_idx = 0
            state = 1
            last_action_time = time.time()
        except Exception as e:
            print(f"HATA: {e}")
            state = 0.1 # Hata olursa seçim ekranına dön

    # ---------------------------------------------------
    # STATE 1 & 2: CALIBRATION & TRAINING
    # ---------------------------------------------------
    elif state == 1:
        draw_text("SYSTEM CALIBRATION IN PROGRESS", WIDTH//2, 80, font_title, CYAN, align="center")
        progress = calib_idx / len(X_calib)
        pygame.draw.rect(screen, GRAY, (150, HEIGHT//2, WIDTH-300, 4), 1) 
        pygame.draw.rect(screen, CYAN, (150, HEIGHT//2, (WIDTH-300)*progress, 4)) 
        
        if time.time() - last_action_time > 0.3:
            command = "IMAGINE LEFT HAND" if y_calib[calib_idx] == 2 else "IMAGINE RIGHT HAND"
            draw_text(command, WIDTH//2, HEIGHT//2 - 40, font_title, CYAN if y_calib[calib_idx]==2 else MAGENTA, align="center")
            calib_idx += 1
            last_action_time = time.time()
            if calib_idx >= len(X_calib): state = 2

    # ---------------------------------------------------
    # STATE 1 & 2: CALIBRATION & TRAINING
    # ---------------------------------------------------
    elif state == 2:
        is_big_data = len(all_subjects_to_load) > 1 

        if is_big_data:
            draw_text("TRAINING GLOBAL NEURAL NETWORK...", WIDTH//2, HEIGHT//2 - 20, font_title, YELLOW, align="center")
            pygame.display.flip()
            train_model(model, X_calib, y_calib)
            save_bci_model(model) 
        else:
            loaded_model = load_bci_model()
            if loaded_model is not None:
                draw_text("TRANSFERRING GLOBAL KNOWLEDGE...", WIDTH//2, HEIGHT//2 - 20, font_title, CYAN, align="center")
                pygame.display.flip()
                model = fine_tune_model(loaded_model, X_calib, y_calib)
                # --- KRİTİK EKLEME: İnce ayardan sonra kişisel ilerlemeyi kaydet ---
                save_bci_model(model) 
                print("[AI CORE] Personal improvements saved to Global Model.")
            else:
                draw_text("NO GLOBAL MODEL FOUND. TRAINING FROM SCRATCH...", WIDTH//2, HEIGHT//2, font_title, YELLOW, align="center")
                pygame.display.flip()
                train_model(model, X_calib, y_calib)
                save_bci_model(model)

        time.sleep(1.5)
        state = 3
        core_color = YELLOW
        last_action_time = time.time()
    
    # ---------------------------------------------------
    # STATE 3: LIVE SIMULATION / NEURAL GAMIFICATION
    # ---------------------------------------------------
    elif state == 3:
        # Üst Menü Butonları
        btn_menu_rect = pygame.Rect(20, 20, 140, 30)
        btn_subj_rect = pygame.Rect(WIDTH - 160, 20, 140, 30)
        btn_toggle_rect = pygame.Rect(WIDTH//2 - 100, 20, 200, 30) # MOD DEĞİŞTİRME BUTONU

        draw_neon_button(screen, "< MAIN MENU", btn_menu_rect, font_small, GRAY, btn_menu_rect.collidepoint(mouse_pos))
        draw_neon_button(screen, "NEW SYNC >", btn_subj_rect, font_small, GRAY, btn_subj_rect.collidepoint(mouse_pos))
        draw_neon_button(screen, f"MODE: {display_mode}", btn_toggle_rect, font_small, CYAN, btn_toggle_rect.collidepoint(mouse_pos))

        if mouse_click:
            if btn_menu_rect.collidepoint(mouse_pos):
                state, current_idx, score, game_score = 0, 0, 0, 0
                asteroids.clear()
            elif btn_subj_rect.collidepoint(mouse_pos):
                state, current_idx, score, game_score = 0.1, 0, 0, 0
                asteroids.clear()
            elif btn_toggle_rect.collidepoint(mouse_pos):
                # Butona basınca modlar arası geçiş yap
                display_mode = "CLASSIC" if display_mode == "GAME" else "GAME"

        # 1. YAPAY ZEKA TAHMİN MOTORU 
        if current_idx >= len(X_test):
            draw_text("MISSION ACCOMPLISHED. AWAITING DATA...", WIDTH//2, HEIGHT//2, font_huge, LIME, align="center")
        elif time.time() - last_action_time > 1.2: 
            single_epoch = X_test[current_idx:current_idx+1]
            pred, conf = get_prediction(model, single_epoch)
            confidence = float(conf)
            
            if pred == 2: # LEFT HAND
                target_x, core_color = WIDTH // 4, CYAN 
            else:         # RIGHT HAND
                target_x, core_color = (WIDTH // 4) * 3, MAGENTA
                
            if y_test[current_idx] == pred:
                score += 1 
                
            current_idx += 1
            last_action_time = time.time()

        # 2. FİZİK MOTORU (Ortak Hız)
        core_x += (target_x - core_x) * 6 * dt 
        ship_y = HEIGHT - 180

        # 3. GÖRSELLEŞTİRME (Kullanıcının Seçtiği Moda Göre)
        if display_mode == "GAME":
            # --- UZAY GEMİSİ VE ASTEROİTLER ---
            if current_idx < len(X_test) and random.random() < 0.04: 
                asteroids.append({'x': random.randint(50, WIDTH-50), 'y': 50, 'speed': random.uniform(4, 8)})

            for ast in asteroids[:]:
                ast['y'] += ast['speed']
                pygame.draw.circle(screen, MAGENTA, (int(ast['x']), int(ast['y'])), 15, 2)
                pygame.draw.circle(screen, BACKGROUND, (int(ast['x']), int(ast['y'])), 13)
                
                dist = np.hypot(core_x - ast['x'], ship_y - ast['y'])
                if dist < 35:
                    collision_flash = 255 
                    game_score -= 10
                    asteroids.remove(ast)
                elif ast['y'] > HEIGHT:
                    game_score += 5 
                    asteroids.remove(ast)

            if collision_flash > 0:
                flash_surf = pygame.Surface((WIDTH, HEIGHT))
                flash_surf.set_alpha(collision_flash)
                flash_surf.fill((200, 0, 0))
                screen.blit(flash_surf, (0,0))
                collision_flash = max(0, collision_flash - 15)

            points = [(core_x, ship_y - 25), (core_x - 20, ship_y + 20), (core_x + 20, ship_y + 20)]
            pygame.draw.polygon(screen, core_color, points, 2)
            pygame.draw.circle(screen, LIME if current_idx < len(X_test) else GRAY, (int(core_x), int(ship_y + 5)), 5)

        else:
            # --- KLASİK HUD (RADAR ÇİZGİSİ) ---
            pygame.draw.line(screen, GRAY, (150, HEIGHT//2), (WIDTH-150, HEIGHT//2), 1)
            pygame.draw.circle(screen, core_color, (int(core_x), HEIGHT//2), 35, 2)
            pygame.draw.circle(screen, LIME if current_idx < len(X_test) else GRAY, (int(core_x), HEIGHT//2), 10)

        # 4. HUD VE İSTATİSTİKLER (Alt Panel)
        pygame.draw.line(screen, core_color, (0, HEIGHT - 130), (WIDTH, HEIGHT - 130), 2)
        pygame.draw.rect(screen, DARK_PANEL, (0, HEIGHT - 130, WIDTH, 130))
        
        acc = (score/current_idx)*100 if current_idx > 0 else 0
        
        if display_mode == "GAME":
            draw_text("GAME SCORE", 60, HEIGHT - 100, font_small, GRAY)
            draw_text(f"{game_score}", 60, HEIGHT - 70, font_huge, LIME if game_score >= 0 else MAGENTA)
        
        draw_text("NEURAL INTENT", WIDTH//2, HEIGHT - 100, font_small, GRAY, align="center")
        draw_text("LEFT" if target_x < WIDTH//2 else "RIGHT", WIDTH//2, HEIGHT - 65, font_huge, core_color, align="center")
        draw_text(f"AI ACCURACY: {acc:.1f}%", WIDTH//2, HEIGHT - 25, font_main, WHITE, align="center")

        draw_text("CONFIDENCE", WIDTH-60, HEIGHT - 100, font_small, GRAY, align="right")
        draw_text(f"{confidence:.1f}%", WIDTH-60, HEIGHT - 70, font_title, CYAN if confidence > 70 else YELLOW, align="right")

    pygame.display.flip()