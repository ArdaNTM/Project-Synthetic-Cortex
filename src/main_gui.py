import pygame
import sys
import time
import numpy as np
from sklearn.model_selection import train_test_split
from utils import load_and_preprocess_data
from engine import create_bci_pipeline, train_model, get_prediction

# --- GERÇEK DONANIM KÜTÜPHANESİ (BRAINFLOW) ---
try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.board_shim import BrainFlowError
    BRAINFLOW_INSTALLED = True
except ImportError:
    BRAINFLOW_INSTALLED = False

# ==========================================
# ⚙️ PROJECT CONFIGURATION
# ==========================================
OPERATION_MODE = "" 
SUBJECT_ID = 1      

# Pygame Init
pygame.init()
WIDTH, HEIGHT = 950, 650
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Project Synthetic Cortex v3.0 - Neural OS")
clock = pygame.time.Clock()

# --- YENİ MİNİMALİST RENK PALETİ ---
BACKGROUND = (8, 9, 13)      # Daha derin ve koyu arka plan
GRID_COLOR = (18, 20, 25)    # Çok hafif grid
WHITE = (240, 240, 245)
CYAN = (0, 255, 255)         # Neon Mavi
MAGENTA = (255, 0, 127)      # Neon Pembe/Kırmızı
LIME = (57, 255, 20)         # Neon Yeşil
YELLOW = (255, 223, 0)
GRAY = (100, 105, 120)
DARK_PANEL = (12, 14, 18)    # Şeffaf hissiyatlı panel

# Fonts
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

def draw_neon_button(surface, text, rect, font, color, hover):
    """Yeni nesil şeffaf ve parlayan buton tasarımı"""
    alpha = 40 if hover else 10
    s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    s.fill((*color, alpha))
    surface.blit(s, rect.topleft)
    pygame.draw.rect(surface, color, rect, 1, border_radius=3)
    draw_text(text, rect.centerx, rect.centery, font, WHITE if hover else color, align="center")

def draw_real_eeg_signal(surface, signal_array, x_start, y_center, width, height, color):
    if len(signal_array) == 0: return
    points = []
    max_val = np.max(np.abs(signal_array)) 
    max_val = max_val if max_val != 0 else 1
    for i, val in enumerate(signal_array):
        x = x_start + (i / len(signal_array)) * width
        y = y_center - (val / max_val) * (height / 2)
        points.append((x, y))
    if len(points) > 1:
        pygame.draw.lines(surface, color, False, points, 2)

# ==========================================
# SYSTEM STATE VARIABLES
# ==========================================
state = 0 
calib_idx, current_idx = 0, 0
last_action_time = time.time()
core_x, target_x = WIDTH//2, WIDTH//2
core_color = CYAN
status_msg, status_color = "AWAITING SIGNAL...", YELLOW
real_intent, confidence = "---", 0.0
score = 0
current_eeg_wave = []
hw_error_msg = "" # Donanım hatasını ekrana basmak için

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
            
        if event.type == pygame.KEYDOWN:
            if state == 0.1: 
                if event.key == pygame.K_UP: SUBJECT_ID = min(109, SUBJECT_ID + 1)
                if event.key == pygame.K_DOWN: SUBJECT_ID = max(1, SUBJECT_ID - 1)
                if event.key == pygame.K_RETURN: 
                    state = 0.5 
                    last_action_time = time.time()

    screen.fill(BACKGROUND)
    draw_grid()
    
    # ---------------------------------------------------
    # STATE 0: MAIN MENU 
    # ---------------------------------------------------
    if state == 0:
        draw_text("SYNTHETIC CORTEX", WIDTH//2, HEIGHT//3 - 60, font_huge, WHITE, align="center")
        draw_text("NEURAL OPERATING SYSTEM v3.0", WIDTH//2, HEIGHT//3 - 20, font_small, CYAN, align="center")

        btn1_rect = pygame.Rect(WIDTH//2 - 250, HEIGHT//2 - 30, 500, 60)
        btn1_hover = btn1_rect.collidepoint(mouse_pos)
        draw_neon_button(screen, "DATABASE MODE (PHYSIONET LIBRARY)", btn1_rect, font_main, CYAN, btn1_hover)

        btn2_rect = pygame.Rect(WIDTH//2 - 250, HEIGHT//2 + 50, 500, 60)
        btn2_hover = btn2_rect.collidepoint(mouse_pos)
        draw_neon_button(screen, "LIVE SENSOR MODE (HARDWARE CONNECTION)", btn2_rect, font_main, MAGENTA, btn2_hover)

        if mouse_click and btn1_hover:
            OPERATION_MODE = "DATABASE"
            state = 0.1 
        elif mouse_click and btn2_hover:
            OPERATION_MODE = "HARDWARE"
            hw_error_msg = ""
            state = 0.2 

    # ---------------------------------------------------
    # STATE 0.1: SUBJECT SELECTION 
    # ---------------------------------------------------
    elif state == 0.1:
        # Geri Dönüş Butonu
        btn_back_rect = pygame.Rect(30, 30, 150, 40)
        btn_back_hover = btn_back_rect.collidepoint(mouse_pos)
        draw_neon_button(screen, "< MAIN MENU", btn_back_rect, font_small, GRAY, btn_back_hover)
        if mouse_click and btn_back_hover: state = 0
        
        draw_text("LIBRARY DATABASE SELECTED", WIDTH//2, HEIGHT//3 - 30, font_title, CYAN, align="center")
        draw_text("SELECT SUBJECT ID", WIDTH//2, HEIGHT//2 - 40, font_small, GRAY, align="center")
        draw_text(f"<  {SUBJECT_ID}  >", WIDTH//2, HEIGHT//2, font_huge, WHITE, align="center")
        draw_text("(Use UP/DOWN arrows to change, ENTER to confirm)", WIDTH//2, HEIGHT//2 + 60, font_small, GRAY, align="center")

    # ---------------------------------------------------
    # STATE 0.2: REAL HARDWARE SETUP & CONNECTION
    # ---------------------------------------------------
    elif state == 0.2:
        btn_back_rect = pygame.Rect(30, 30, 150, 40)
        btn_back_hover = btn_back_rect.collidepoint(mouse_pos)
        draw_neon_button(screen, "< MAIN MENU", btn_back_rect, font_small, GRAY, btn_back_hover)
        if mouse_click and btn_back_hover: state = 0

        draw_text("HARDWARE SENSOR SETUP", WIDTH//2, HEIGHT//3 - 40, font_title, MAGENTA, align="center")
        draw_text("1. Connect the USB Dongle / Bluetooth to your PC.", WIDTH//2, HEIGHT//3 + 10, font_main, WHITE, align="center")
        draw_text("2. Turn on your EEG Headset (OpenBCI, Muse, etc.).", WIDTH//2, HEIGHT//3 + 40, font_main, WHITE, align="center")
        draw_text("3. Ensure electrodes are properly placed on Motor Cortex (C3/C4).", WIDTH//2, HEIGHT//3 + 70, font_main, WHITE, align="center")
        
        btn_connect_rect = pygame.Rect(WIDTH//2 - 150, HEIGHT//2 + 50, 300, 50)
        btn_connect_hover = btn_connect_rect.collidepoint(mouse_pos)
        draw_neon_button(screen, "CONNECT & VERIFY", btn_connect_rect, font_title, LIME, btn_connect_hover)

        if hw_error_msg:
            draw_text(hw_error_msg, WIDTH//2, HEIGHT//2 + 120, font_small, MAGENTA, align="center")

        # GERÇEK BAĞLANTI MANTIĞI
        if mouse_click and btn_connect_hover:
            if not BRAINFLOW_INSTALLED:
                hw_error_msg = "FATAL: 'brainflow' library missing. Run 'pip install brainflow'."
            else:
                hw_error_msg = "SCANNING PORTS..."
                draw_text(hw_error_msg, WIDTH//2, HEIGHT//2 + 120, font_small, YELLOW, align="center")
                pygame.display.flip()
                
                try:
                    # Gerçek cihaz bağlantı denemesi (Örnek: OpenBCI Cyton)
                    params = BrainFlowInputParams()
                    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
                    board.prepare_session() # EĞER CİHAZ YOKSA BURASI HATA FIRLATIR!
                    board.start_stream()
                    # Başarılı olursa (Sensör bağlarsan):
                    state = 1
                    last_action_time = time.time()
                except BrainFlowError as e:
                    # Cihaz yoksa gerçeği söyler ve geçirmez:
                    hw_error_msg = "CONNECTION FAILED: NO DEVICE DETECTED ON ACTIVE PORTS."

    # ---------------------------------------------------
    # STATE 0.5: DATA LOADING / PREPROCESSING 
    # ---------------------------------------------------
    elif state == 0.5:
        draw_text("ESTABLISHING NEURAL CONNECTION...", WIDTH//2, HEIGHT//2 - 20, font_title, YELLOW, align="center")
        draw_text("Downloading/Processing Data. This may take a moment.", WIDTH//2, HEIGHT//2 + 20, font_small, GRAY, align="center")
        pygame.display.flip()
        
        X, y = load_and_preprocess_data(subject_id=SUBJECT_ID)
        X_calib, X_test, y_calib, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        
        calib_idx = 0
        state = 1
        last_action_time = time.time()

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

    elif state == 2:
        draw_text("BUILDING PERSONAL NEURAL PROFILE...", WIDTH//2, HEIGHT//2, font_title, YELLOW, align="center")
        pygame.display.flip()
        train_model(model, X_calib, y_calib)
        time.sleep(1.5)
        state = 3
        core_color = YELLOW
        last_action_time = time.time()

    # ---------------------------------------------------
    # STATE 3: LIVE SIMULATION & HUD
    # ---------------------------------------------------
    elif state == 3:
        # Yeni Nesil Navigasyon Butonları
        btn_menu_rect = pygame.Rect(20, 20, 140, 30)
        btn_subj_rect = pygame.Rect(WIDTH - 160, 20, 140, 30)
        draw_neon_button(screen, "< MAIN MENU", btn_menu_rect, font_small, GRAY, btn_menu_rect.collidepoint(mouse_pos))
        draw_neon_button(screen, "CHANGE SUBJ >", btn_subj_rect, font_small, GRAY, btn_subj_rect.collidepoint(mouse_pos))

        if mouse_click:
            if btn_menu_rect.collidepoint(mouse_pos):
                state, current_idx, score = 0, 0, 0
            elif btn_subj_rect.collidepoint(mouse_pos):
                state, current_idx, score = 0.1, 0, 0

        # Simülasyon Mantığı
        if current_idx >= len(X_test):
            target_x, core_color = WIDTH//2, LIME
            status_msg, status_color = "SIMULATION COMPLETE", LIME
            current_eeg_wave = [] 
            draw_text("TEST FINISHED. AWAITING NEW DATA...", WIDTH//2, HEIGHT//2 + 50, font_title, LIME, align="center")
        
        elif time.time() - last_action_time > 2.5:
            single_epoch = X_test[current_idx:current_idx+1]
            pred, conf = get_prediction(model, single_epoch)
            real_intent = "LEFT HAND" if y_test[current_idx] == 2 else "RIGHT HAND"
            confidence = conf
            current_eeg_wave = single_epoch[0][0]
            
            if pred == 2: target_x, core_color, p_text = 250, CYAN, "LEFT"
            else: target_x, core_color, p_text = WIDTH-250, MAGENTA, "RIGHT"
            
            if y_test[current_idx] == pred:
                score += 1
                status_msg, status_color = f"SUCCESS: {p_text} DETECTED", LIME
            else:
                status_msg, status_color = f"ERROR: MISINTERPRETED AS {p_text}", MAGENTA
                
            current_idx += 1
            last_action_time = time.time()

        core_x += (target_x - core_x) * 6 * dt
        
        if current_idx < len(X_test) and time.time() - last_action_time > 1.8 and target_x != WIDTH//2:
            target_x, core_color = WIDTH//2, YELLOW
            status_msg, status_color, real_intent, confidence = "SCANNING NEURAL WAVES...", YELLOW, "---", 0.0
            current_eeg_wave = [] 

        # --- YENİ NESİL ÇİZİMLER ---
        pygame.draw.line(screen, GRAY, (150, HEIGHT//2 - 40), (WIDTH-150, HEIGHT//2 - 40), 1)
        
        if len(current_eeg_wave) > 0:
            draw_real_eeg_signal(screen, current_eeg_wave, x_start=150, y_center=HEIGHT//2 + 30, width=WIDTH-300, height=70, color=core_color)

        pygame.draw.circle(screen, BACKGROUND, (int(core_x), HEIGHT//2 - 40), 15) 
        pygame.draw.circle(screen, core_color, (int(core_x), HEIGHT//2 - 40), 35, 2)
        pygame.draw.circle(screen, core_color, (int(core_x), HEIGHT//2 - 40), 10)    

        # Üst Bilgiler
        draw_text("SYNTHETIC CORTEX", WIDTH//2, 25, font_title, WHITE, align="center")
        draw_text(f"MODE: {OPERATION_MODE} | SUBJ: {SUBJECT_ID}", WIDTH//2, 55, font_small, GRAY, align="center")
        
        # --- SİMETRİK ALT PANEL (DASHBOARD) ---
        panel_y = HEIGHT - 130
        pygame.draw.rect(screen, DARK_PANEL, (0, panel_y, WIDTH, 130))
        pygame.draw.line(screen, core_color, (0, panel_y), (WIDTH, panel_y), 2)

        # 1. Sütun: Ground Truth
        draw_text("GROUND TRUTH", 60, panel_y + 30, font_small, GRAY)
        draw_text(real_intent, 60, panel_y + 60, font_title, WHITE)

        # 2. Sütun: Ortada Karar ve İsabet
        acc = (score/current_idx)*100 if current_idx > 0 else 0
        draw_text("AI DECISION", WIDTH//2, panel_y + 20, font_small, GRAY, align="center")
        draw_text(status_msg, WIDTH//2, panel_y + 45, font_main, status_color, align="center")
        draw_text(f"GLOBAL ACCURACY: {acc:.1f}%", WIDTH//2, panel_y + 85, font_main, LIME if acc >= 70 else YELLOW, align="center")

        # 3. Sütun: Sağda Confidence Bar
        draw_text("CONFIDENCE", WIDTH-60, panel_y + 30, font_small, GRAY, align="right")
        draw_text(f"{confidence:.1f}%", WIDTH-60, panel_y + 55, font_title, CYAN if confidence > 70 else YELLOW, align="right")
        
        bar_width = 140
        pygame.draw.rect(screen, GRAY, (WIDTH - 60 - bar_width, panel_y + 90, bar_width, 4), 1) 
        if confidence > 0:
            fill_width = bar_width * (confidence / 100.0)
            pygame.draw.rect(screen, CYAN if confidence > 70 else YELLOW, (WIDTH - 60 - bar_width, panel_y + 90, fill_width, 4))

    pygame.display.flip()