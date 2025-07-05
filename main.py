import numpy as np
import pyaudio
import joblib
import time
import os
import requests
import pyautogui
import keyboard
import customtkinter as ctk
import webbrowser
import json
import threading
from collections import deque
from threading import Thread, Event
from PIL import Image
import sys

#ФИКС ПОИСКА ИКОНКИ И МОДЕЛИ
def resource_path(relative_path):
    """ Получает абсолютный путь для ресурсов в exe или исходном коде """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

#КОНФИГУРАЦИЯ
CONFIG_FILE = "config.json"
MODEL_FILE = "fish_model_2features.pkl" 
MODEL_FILE = (resource_path("fish_model_2features.pkl")) 
SAMPLE_RATE = 44100
WINDOW_SIZE = 0.15
BUFFER_SIZE = int(SAMPLE_RATE * WINDOW_SIZE)
STEP_SIZE = int(SAMPLE_RATE * 0.05)
ENERGY_THRESHOLD = 0.015
MIN_STRIKE_INTERVAL = 7.0

default_config = {
    "hotkey": "F11",
    "telegram_enabled": True,
    "telegram_token": "",
    "telegram_chat_id": ""
}



#класс детектора
class FishDetector:
    def __init__(self, config, log_callback=None):
        self.config = config
        self.log_callback = log_callback
        self.stop_event = Event()
        self.fish_strike_event = Event()
        self.auto_fishing = False
        self.audio_buffer = deque(maxlen=10 * SAMPLE_RATE)
        self.hotkey_id = None
        self.stream = None
        self.p = None
        self.detection_thread = None
        self.auto_fishing_thread = None
        
        #чек модели
        if not os.path.exists(MODEL_FILE):
            self.log("ОШИБКА: Модель не найдена!")
            raise FileNotFoundError(f"Модель {MODEL_FILE} не найдена!")
        
        try:
            self.model = joblib.load(MODEL_FILE)
        except Exception as e:
            self.log(f"Ошибка загрузки модели: {e}")
            raise

    def log(self, message):
        if self.log_callback:
            self.log_callback(message)

    def start(self):
        if self.detection_thread and self.detection_thread.is_alive():
            self.log("Детектор уже запущен!")
            return

        self.log("Запуск детектора...")
        self.stop_event.clear()
        
        #старт звука
        try:
            self.p = pyaudio.PyAudio()
            self.loopback_device_index = self.find_loopback_device()
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=STEP_SIZE,
                input_device_index=self.loopback_device_index,
                stream_callback=self.audio_callback
            )
            self.stream.start_stream()
        except Exception as e:
            self.log(f"Ошибка инициализации звука: {str(e)}")
            return

        #поиск детекта
        self.detection_thread = Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
        #хоткей регистр
        self.register_hotkey()
        self.log("Детектор запущен. Ожидание поклевки...")

    def stop(self):
        self.log("Остановка детектора...")
        self.stop_event.set()
        self.auto_fishing = False
        
        #стоп поток
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1.0)
        
        #стоп звук
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        
        #ремув хоткей
        if self.hotkey_id:
            keyboard.remove_hotkey(self.hotkey_id)
        
        self.log("Детектор остановлен")

    def find_loopback_device(self):
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            if "CABLE Output" in dev["name"] or "Stereo Mix" in dev["name"]:
                self.log(f"Устройство захвата: {dev['name']}")
                return dev["index"]
        self.log("Использую устройство по умолчанию")
        return self.p.get_default_input_device_info()["index"]

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_chunk = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer.extend(audio_chunk)
        return (in_data, pyaudio.paContinue)

    def detection_loop(self):
        last_detection_time = 0
        while not self.stop_event.is_set():
            if len(self.audio_buffer) >= BUFFER_SIZE:
                segment = np.array(list(self.audio_buffer)[-BUFFER_SIZE:])
                rms = np.sqrt(np.mean(segment**2))

                if rms < ENERGY_THRESHOLD:
                    time.sleep(0.01)
                    continue

                current_time = time.time()
                if current_time - last_detection_time < MIN_STRIKE_INTERVAL:
                    time.sleep(0.01)
                    continue

                try:
                    if self.detect_peck(segment):
                        self.log(f"!!! ПОКЛЕВКА !!! ({time.strftime('%H:%M:%S')})")
                        self.fish_strike_event.set()
                        if self.config["telegram_enabled"] and self.config["telegram_token"]:
                            self.send_telegram_notify()
                        last_detection_time = current_time
                except Exception as e:
                    self.log(f"Ошибка детекции: {str(e)}")

            time.sleep(0.001)

    def detect_peck(self, audio):
        features = self.extract_features(audio)
        return self.model.predict([features])[0] == 1

    def extract_features(self, audio):
        rms = np.sqrt(np.mean(audio**2))
        fft = np.abs(np.fft.rfft(audio)[:50])
        spectral_centroid = (
            np.sum(np.arange(len(fft)) * fft) / np.sum(fft) if np.sum(fft) > 0 else 0.0
        )
        return [rms, spectral_centroid]
    #уведомления в тг просто по ссылке
    def send_telegram_notify(self):
        try:
            text = "🎣 Рыбка поймана!"
            url = f"https://api.telegram.org/bot{self.config['telegram_token']}/sendMessage"
            params = {"chat_id": self.config["telegram_chat_id"], "text": text}
            requests.get(url, params=params, timeout=3)
            self.log("Уведомление отправлено в Telegram")
        except Exception as e:
            self.log(f"Ошибка Telegram: {e}")

    def toggle_auto_fishing(self):
        self.auto_fishing = not self.auto_fishing
        if self.auto_fishing:
            self.auto_fishing_thread = Thread(target=self.auto_fishing_loop, daemon=True)
            self.auto_fishing_thread.start()
            self.log("Авторыбалка АКТИВИРОВАНА")
        else:
            self.log("Авторыбалка ОСТАНОВЛЕНА")

    def auto_fishing_loop(self):
        self.log("Авторыбалка запущена...")
        while self.auto_fishing and not self.stop_event.is_set():
            pyautogui.mouseDown()
            self.fish_strike_event.clear()
            
            #ждем рыбку
            while self.auto_fishing and not self.fish_strike_event.wait(0.1):
                if self.stop_event.is_set():
                    break
            
            pyautogui.mouseUp()
            
            #задержка между забросами пока идет анимация, менять число в range
            for _ in range(6):
                if not self.auto_fishing or self.stop_event.is_set():
                    break
                time.sleep(1)
        
        pyautogui.mouseUp()  #отпускаем лкм

    def register_hotkey(self):
        if self.hotkey_id:
            keyboard.remove_hotkey(self.hotkey_id)
        try:
            self.hotkey_id = keyboard.add_hotkey(
                self.config["hotkey"], 
                self.toggle_auto_fishing
            )
            self.log(f"Горячая клавиша: {self.config['hotkey']}")
        except Exception as e:
            self.log(f"Ошибка горячей клавиши: {str(e)}")

    def update_config(self, new_config):
        self.config = new_config
        self.register_hotkey()
        self.log("Конфигурация обновлена")

#гуи


def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            return default_config.copy()
    return default_config.copy()

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DeepWater")
        self.geometry("460x570")
        self.iconbitmap(resource_path("icon.ico"))
        # старый метод self.iconbitmap("icon.ico")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.config = load_config()
        self.fish_detector = None
        self.is_running = False
        
        self.setup_ui()
        self.after(100, self.check_detector_status)

    def setup_ui(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        
        self.title_label = ctk.CTkLabel(self, text="DeepWater", font=("Segoe UI", 24, "bold"))
        self.title_label.pack(pady=(20, 5))

        self.link = ctk.CTkLabel(self, text="by Kailiix", text_color="skyblue", cursor="hand2")
        self.link.pack()
        self.link.bind("<Button-1>", lambda e: webbrowser.open("https://kailiix.com"))

        
        self.hotkey_label = ctk.CTkLabel(self, text="Горячая клавиша:")
        self.hotkey_label.pack(pady=(20, 5))
        
        self.hotkey_entry = ctk.CTkEntry(self)
        self.hotkey_entry.insert(0, self.config["hotkey"])
        self.hotkey_entry.pack()

        # настройка тг
        self.tg_checkbox = ctk.CTkCheckBox(
            self, 
            text="Уведомления в Telegram",
            variable=ctk.IntVar(value=int(self.config["telegram_enabled"])))
        self.tg_checkbox.pack(pady=(15, 5))

        self.token_label = ctk.CTkLabel(self, text="Telegram Token:")
        self.token_label.pack()
        
        self.token_entry = ctk.CTkEntry(self, width=320)
        self.token_entry.insert(0, self.config["telegram_token"])
        self.token_entry.pack()

        self.chat_id_label = ctk.CTkLabel(self, text="Chat ID:")
        self.chat_id_label.pack()
        
        self.chat_id_entry = ctk.CTkEntry(self, width=320)
        self.chat_id_entry.insert(0, self.config["telegram_chat_id"])
        self.chat_id_entry.pack()

        
        self.buttons_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.buttons_frame.pack(pady=20)

        self.start_btn = ctk.CTkButton(
            self.buttons_frame, 
            text="🚀 Запустить", 
            width=140, 
            command=self.on_start)
        self.start_btn.grid(row=0, column=0, padx=10)

        self.stop_btn = ctk.CTkButton(
            self.buttons_frame, 
            text="⛔ Остановить", 
            width=140, 
            command=self.on_stop,
            state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=10)

        self.save_btn = ctk.CTkButton(
            self, 
            text="💾 Сохранить настройки", 
            width=300, 
            command=self.on_save)
        self.save_btn.pack(pady=(0, 10))

        # логи
        self.log_label = ctk.CTkLabel(self, text="Лог событий:")
        self.log_label.pack(pady=(10, 0))

        self.log_box = ctk.CTkTextbox(
            self, 
            width=400, 
            height=120, 
            state="disabled", 
            wrap="word")
        self.log_box.pack(pady=(5, 20))

    def log_message(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def animate_label(self, label):
        label.after(2000, lambda: label.destroy())

    def on_save(self):
        self.config["hotkey"] = self.hotkey_entry.get()
        self.config["telegram_enabled"] = self.tg_checkbox.get() == 1
        self.config["telegram_token"] = self.token_entry.get()
        self.config["telegram_chat_id"] = self.chat_id_entry.get()
        save_config(self.config)

        if self.fish_detector and self.is_running:
            self.fish_detector.update_config(self.config)

        saved_label = ctk.CTkLabel(self, text="✅ Сохранено", text_color="green")
        saved_label.pack()
        self.animate_label(saved_label)
        self.log_message("Конфигурация сохранена")

    def on_start(self):
        if self.is_running:
            self.log_message("⚠️ Детектор уже запущен!")
            return

        try:
            self.fish_detector = FishDetector(
                config=self.config,
                log_callback=self.log_message
            )
            self.fish_detector.start()
            self.is_running = True
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
        except Exception as e:
            self.log_message(f"Ошибка запуска: {str(e)}")

    def on_stop(self):
        if not self.is_running:
            self.log_message("⚠️ Детектор не запущен!")
            return

        if self.fish_detector:
            self.fish_detector.stop()
            self.is_running = False
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")

    def check_detector_status(self):
        if self.fish_detector and self.is_running:
            if not self.fish_detector.detection_thread.is_alive():
                self.log_message("⚠️ Детектор аварийно остановлен!")
                self.is_running = False
                self.start_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
        self.after(1000, self.check_detector_status)

    def on_closing(self):
        if self.fish_detector and self.is_running:
            self.fish_detector.stop()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()