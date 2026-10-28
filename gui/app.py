import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import torch

from model import (
    DATA_DIR,
    DEMO_DATA_PATH,
    DEMO_EXAMPLES,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_LAYERS,
    EOS_TOKEN,
    LEGACY_MODEL_PATH,
    LEGACY_ROOT_MODEL_PATH,
    MODEL_DIR,
    VOCAB_SIZE,
    CodeModel,
    available_models,
    generate_code,
    load_training_examples,
    model_checkpoint_path,
    train_model,
)


class CodeGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cortex - Python-Codegenerator")
        self.root.geometry("900x780")
        self.model = None
        parquet_files = sorted(DATA_DIR.glob("*.parquet"))
        default_path = parquet_files[0] if len(parquet_files) == 1 else DEMO_DATA_PATH
        self.training_path = tk.StringVar(value=str(default_path))
        self.selected_model = tk.StringVar()
        self.epochs = tk.DoubleVar(value=5)
        self.max_examples = tk.StringVar(value="1,000")
        self.device = tk.StringVar(value="Automatisch")
        self.device_info = tk.StringVar()
        self.hidden_size = tk.IntVar(value=DEFAULT_HIDDEN_SIZE)
        self.num_layers = tk.IntVar(value=DEFAULT_NUM_LAYERS)
        self.learning_rate = tk.StringVar(value=str(DEFAULT_LEARNING_RATE))
        self.status = tk.StringVar(value="Kein trainiertes Modell geladen.")
        self.training_progress = tk.DoubleVar(value=0)
        self.training_time = tk.StringVar(value="Noch kein Training")
        self.messages = queue.Queue()
        self.activation_lock = threading.Lock()
        self.latest_activation = None
        self.last_activation = None
        self.visual_window = None
        self.visual_nodes = {}
        self._update_device_info()

        self._build_ui()
        self._load_saved_model()
        self.root.after(100, self._process_messages)

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(5, weight=1)
        frame.rowconfigure(8, weight=1)

        ttk.Label(
            frame,
            text="Eigener Python-Codegenerator",
            font=("Segoe UI", 18, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            frame,
            text=(
                "Dieses kleine Zeichenmodell wird lokal von Grund auf trainiert. "
                "Es ist ein Lernprototyp, kein leistungsfähiges Coding-Modell."
            ),
            wraplength=760,
        ).grid(row=1, column=0, sticky="w", pady=(4, 14))

        self._build_training_controls(frame)
        ttk.Label(frame, text="Anforderung").grid(row=4, column=0, sticky="w")
        self.prompt = ScrolledText(frame, height=5, wrap="word")
        self.prompt.grid(row=5, column=0, sticky="nsew", pady=(4, 12))

        self._build_output_controls(frame)
        ttk.Label(frame, text="Generierter Code (vor dem Ausführen prüfen):").grid(
            row=7, column=0, sticky="w", pady=(12, 0)
        )
        self.output = ScrolledText(frame, height=10, wrap="none")
        self.output.grid(row=8, column=0, sticky="nsew", pady=(4, 8))
        ttk.Label(frame, textvariable=self.status).grid(row=9, column=0, sticky="w")

    def _build_training_controls(self, parent):
        data_frame = ttk.LabelFrame(
            parent, text="Trainingsdaten (JSONL oder Parquet)", padding=10
        )
        data_frame.grid(row=2, column=0, sticky="ew")
        data_frame.columnconfigure(0, weight=1)
        ttk.Entry(data_frame, textvariable=self.training_path).grid(
            row=0, column=0, sticky="ew", padx=(0, 8)
        )
        ttk.Button(data_frame, text="Datei wählen…", command=self._choose_data).grid(
            row=0, column=1
        )
        ttk.Label(
            data_frame,
            text=(
                'Jede JSONL-Zeile: {"prompt": "Anforderung", '
                '"code": "passender Python-Code"}.'
            ),
            wraplength=740,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(7, 0))
        ttk.Label(data_frame, text="Modell auswählen:").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        self.model_selector = ttk.Combobox(
            data_frame,
            textvariable=self.selected_model,
            state="readonly",
            width=42,
        )
        self.model_selector.grid(
            row=3, column=0, sticky="ew", pady=(4, 0)
        )
        self.model_selector.bind("<<ComboboxSelected>>", self._on_model_selected)
        ttk.Button(
            data_frame, text="Aktualisieren", command=self._refresh_model_choices
        ).grid(row=3, column=1, sticky="e", pady=(4, 0))

        train_frame = ttk.Frame(parent)
        train_frame.grid(row=3, column=0, sticky="ew", pady=12)
        ttk.Label(train_frame, text="Trainingsstärke (Epochen):").grid(
            row=0, column=0, sticky="w"
        )
        self.epoch_scale = ttk.Scale(
            train_frame,
            from_=1,
            to=50,
            variable=self.epochs,
            command=self._update_epoch_label,
        )
        self.epoch_scale.grid(row=0, column=1, sticky="ew", padx=8)
        self.epoch_label = ttk.Label(train_frame, text="5")
        self.epoch_label.grid(row=0, column=2, sticky="w")
        ttk.Label(train_frame, text="Max. Beispiele:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Combobox(
            train_frame,
            textvariable=self.max_examples,
            values=("250", "500", "1,000", "2,500", "5,000", "10,000", "Alle"),
            state="readonly",
            width=12,
        ).grid(row=1, column=1, sticky="w", padx=8, pady=(8, 0))
        ttk.Label(train_frame, text="Weniger Beispiele = schnellerer Probelauf").grid(
            row=1, column=2, sticky="w", pady=(8, 0)
        )
        ttk.Label(train_frame, text="Rechengerät:").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Combobox(
            train_frame,
            textvariable=self.device,
            values=("Automatisch", "CPU", "GPU"),
            state="readonly",
            width=12,
        ).grid(row=2, column=1, sticky="w", padx=8, pady=(8, 0))
        ttk.Label(train_frame, textvariable=self.device_info, wraplength=480).grid(
            row=2, column=2, sticky="w", pady=(8, 0)
        )
        train_frame.columnconfigure(1, weight=1)
        self.train_button = ttk.Button(
            train_frame, text="Von Grund auf trainieren", command=self._start_training
        )
        self.train_button.grid(row=0, column=3, rowspan=3, padx=(12, 0))
        ttk.Button(
            train_frame,
            text="Beispieldaten erstellen",
            command=self._write_demo_data,
        ).grid(row=0, column=4, rowspan=3, padx=8)
        self.settings_button = ttk.Button(
            train_frame,
            text="Modell-Settings…",
            command=self._open_model_settings,
        )
        self.settings_button.grid(
            row=3, column=3, columnspan=2, sticky="e", padx=8, pady=(8, 0)
        )
        self.progress_bar = ttk.Progressbar(
            train_frame,
            variable=self.training_progress,
            maximum=100,
            mode="determinate",
        )
        self.progress_bar.grid(
            row=4, column=0, columnspan=4, sticky="ew", padx=(0, 8), pady=(12, 0)
        )
        ttk.Label(train_frame, textvariable=self.training_time).grid(
            row=4, column=4, sticky="e", pady=(12, 0)
        )

    def _update_epoch_label(self, _value=None):
        if hasattr(self, "epoch_label"):
            self.epoch_label.configure(text=str(round(self.epochs.get())))

    def _update_device_info(self):
        if torch.cuda.is_available():
            self.device_info.set(f"GPU verfügbar: {torch.cuda.get_device_name(0)}")
        else:
            self.device_info.set(
                "Keine CUDA-GPU verfügbar; PyTorch trainiert derzeit auf der CPU."
            )

    def _open_model_settings(self):
        window = tk.Toplevel(self.root)
        window.title("Modell-Settings")
        window.resizable(False, False)
        window.transient(self.root)
        window.grab_set()

        hidden_size = tk.IntVar(value=self.hidden_size.get())
        num_layers = tk.IntVar(value=self.num_layers.get())
        learning_rate = tk.StringVar(value=self.learning_rate.get())

        ttk.Label(window, text="Hidden-Größe (Einheiten pro Layer):").grid(
            row=0, column=0, sticky="w", padx=12, pady=(12, 6)
        )
        ttk.Spinbox(
            window,
            from_=32,
            to=512,
            increment=32,
            textvariable=hidden_size,
            width=10,
        ).grid(row=0, column=1, padx=12, pady=(12, 6))
        ttk.Label(window, text="LSTM-Layer:").grid(
            row=1, column=0, sticky="w", padx=12, pady=6
        )
        ttk.Spinbox(
            window,
            from_=1,
            to=6,
            increment=1,
            textvariable=num_layers,
            width=10,
        ).grid(row=1, column=1, padx=12, pady=6)
        ttk.Label(window, text="Lernrate:").grid(
            row=2, column=0, sticky="w", padx=12, pady=6
        )
        ttk.Combobox(
            window,
            textvariable=learning_rate,
            values=("0.001", "0.003", "0.005", "0.01"),
            state="readonly",
            width=10,
        ).grid(row=2, column=1, padx=12, pady=6)
        ttk.Label(
            window,
            text=(
                "Größere Modelle brauchen mehr Zeit und Speicher. "
                "Jede Kombination erhält eine eigene Checkpoint-Datei."
            ),
            wraplength=360,
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=12, pady=8)

        def apply_settings():
            try:
                new_hidden_size = hidden_size.get()
                new_num_layers = num_layers.get()
                new_learning_rate = float(learning_rate.get())
                if new_hidden_size < 32 or new_hidden_size > 512:
                    raise ValueError("Hidden-Größe muss zwischen 32 und 512 liegen.")
                if new_num_layers < 1 or new_num_layers > 6:
                    raise ValueError("Layerzahl muss zwischen 1 und 6 liegen.")
            except (ValueError, tk.TclError) as error:
                messagebox.showerror(
                    "Ungültige Modell-Settings", str(error), parent=window
                )
                return

            self.hidden_size.set(new_hidden_size)
            self.num_layers.set(new_num_layers)
            self.learning_rate.set(format(new_learning_rate, "g"))
            self.selected_model.set("Neues Modell (aktuelle Settings)")
            self._load_selected_model()
            window.destroy()

        ttk.Button(window, text="Übernehmen", command=apply_settings).grid(
            row=4, column=0, padx=12, pady=(4, 12), sticky="w"
        )
        ttk.Button(window, text="Abbrechen", command=window.destroy).grid(
            row=4, column=1, padx=12, pady=(4, 12), sticky="e"
        )

    def _build_output_controls(self, parent):
        output_frame = ttk.Frame(parent)
        output_frame.grid(row=6, column=0, sticky="ew")
        output_frame.columnconfigure(0, weight=1)
        self.generate_button = ttk.Button(
            output_frame, text="Python-Code erzeugen", command=self._generate
        )
        self.generate_button.grid(row=0, column=0, sticky="w")
        ttk.Button(
            output_frame, text="Netzwerk ansehen", command=self._open_visualization
        ).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(output_frame, text="Code kopieren", command=self._copy_code).grid(
            row=0, column=2, padx=(8, 0)
        )

    def _load_saved_model(self):
        self._refresh_model_choices()
        self._load_selected_model()

    def _refresh_model_choices(self, selected_path=None):
        model_paths = available_models()
        options = ["Neues Modell (aktuelle Settings)"]
        options.extend(
            (
                f"{path.name} (altes Format)"
                if path == LEGACY_MODEL_PATH
                else path.name
            )
            for path in model_paths
        )
        if LEGACY_ROOT_MODEL_PATH.exists():
            options.append(f"{LEGACY_ROOT_MODEL_PATH.name} (altes Format)")
        self.model_selector.configure(values=options)

        if selected_path is not None:
            selected_name = Path(selected_path).name
        elif self.selected_model.get() in options:
            selected_name = self.selected_model.get()
        else:
            default_path = model_checkpoint_path(
                DEFAULT_HIDDEN_SIZE,
                DEFAULT_NUM_LAYERS,
                DEFAULT_LEARNING_RATE,
            )
            selected_name = (
                default_path.name
                if default_path.exists()
                else (
                    (
                        f"{LEGACY_MODEL_PATH.name} (altes Format)"
                        if LEGACY_MODEL_PATH.exists()
                        else f"{LEGACY_ROOT_MODEL_PATH.name} (altes Format)"
                    )
                    if LEGACY_MODEL_PATH.exists() or LEGACY_ROOT_MODEL_PATH.exists()
                    else options[0]
                )
            )
        if selected_name not in options:
            selected_name = options[0]
        self.selected_model.set(selected_name)

    def _on_model_selected(self, _event=None):
        selected = self.selected_model.get()
        if selected == "Neues Modell (aktuelle Settings)":
            self._load_selected_model()
            return
        if selected in (
            f"{LEGACY_MODEL_PATH.name} (altes Format)",
            f"{LEGACY_ROOT_MODEL_PATH.name} (altes Format)",
        ):
            self.hidden_size.set(DEFAULT_HIDDEN_SIZE)
            self.num_layers.set(DEFAULT_NUM_LAYERS)
            self.learning_rate.set(format(DEFAULT_LEARNING_RATE, "g"))
            self._load_selected_model()
            return

        path = (
            LEGACY_MODEL_PATH
            if selected == f"{LEGACY_MODEL_PATH.name} (altes Format)"
            else LEGACY_ROOT_MODEL_PATH
            if selected == f"{LEGACY_ROOT_MODEL_PATH.name} (altes Format)"
            else MODEL_DIR / selected
        )
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            hidden_size = checkpoint["hidden_size"]
            num_layers = checkpoint["num_layers"]
            learning_rate = checkpoint["learning_rate"]
            if (
                checkpoint.get("version") != 2
                or not isinstance(hidden_size, int)
                or not isinstance(num_layers, int)
                or not isinstance(learning_rate, (float, int))
            ):
                raise ValueError("Die Datei enthält keine gültigen Modell-Settings.")
        except (OSError, RuntimeError, KeyError, ValueError) as error:
            messagebox.showerror(
                "Modell konnte nicht ausgewählt werden",
                f"{selected}\n{error}",
                parent=self.root,
            )
            return

        self.hidden_size.set(hidden_size)
        self.num_layers.set(num_layers)
        self.learning_rate.set(format(float(learning_rate), "g"))
        self._load_selected_model()

    def _load_selected_model(self):
        hidden_size = self.hidden_size.get()
        num_layers = self.num_layers.get()
        learning_rate = float(self.learning_rate.get())
        checkpoint_path = model_checkpoint_path(
            hidden_size, num_layers, learning_rate
        )
        selected = self.selected_model.get()
        if selected == "Neues Modell (aktuelle Settings)":
            self.model = None
            self.generate_button.configure(state="disabled")
            self.status.set(f"Neues Modell: {checkpoint_path.name}")
            return

        if selected == f"{LEGACY_MODEL_PATH.name} (altes Format)":
            path = LEGACY_MODEL_PATH
        elif selected == f"{LEGACY_ROOT_MODEL_PATH.name} (altes Format)":
            path = LEGACY_ROOT_MODEL_PATH
        else:
            path = MODEL_DIR / selected
        if not path.exists():
            self.model = None
            self.generate_button.configure(state="disabled")
            self.status.set(f"Modell nicht gefunden: {path.name}")
            return

        self.model = None
        self.generate_button.configure(state="disabled")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint.get("version") not in (1, 2):
                raise ValueError("Unbekannte Modellversion.")
            if checkpoint.get("version") == 2 and (
                checkpoint.get("hidden_size") != hidden_size
                or checkpoint.get("num_layers") != num_layers
                or checkpoint.get("learning_rate") != learning_rate
            ):
                raise ValueError("Checkpoint und ausgewählte Modell-Settings weichen ab.")
            model = CodeModel(hidden_size, num_layers)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
        except (OSError, RuntimeError, KeyError, ValueError) as error:
            messagebox.showerror(
                "Modell konnte nicht geladen werden",
                f"Die gespeicherte Modelldatei ist ungültig:\n{error}",
                parent=self.root,
            )
            return
        self.model = model
        self.generate_button.configure(state="normal")
        self.status.set(f"Trainiertes Modell geladen: {path.name}")

    def _choose_data(self):
        selected = filedialog.askopenfilename(
            title="Trainingsdatei auswählen",
            filetypes=[
                ("Trainingsdaten", ("*.jsonl", "*.parquet")),
                ("JSONL-Dateien", "*.jsonl"),
                ("Parquet-Dateien", "*.parquet"),
                ("Alle Dateien", "*.*"),
            ],
        )
        if selected:
            self.training_path.set(selected)

    def _write_demo_data(self):
        if DEMO_DATA_PATH.exists():
            messagebox.showinfo(
                "Beispieldaten vorhanden",
                f"Die Datei existiert bereits:\n{DEMO_DATA_PATH}",
                parent=self.root,
            )
            return
        try:
            with DEMO_DATA_PATH.open("x", encoding="utf-8") as data_file:
                for example in DEMO_EXAMPLES:
                    data_file.write(json.dumps(example, ensure_ascii=False) + "\n")
        except OSError as error:
            messagebox.showerror(
                "Beispieldaten konnten nicht gespeichert werden",
                str(error),
                parent=self.root,
            )
            return
        self.training_path.set(str(DEMO_DATA_PATH))
        self.status.set(f"Beispieldaten gespeichert: {DEMO_DATA_PATH.name}")

    def _start_training(self):
        try:
            epochs = round(self.epochs.get())
            hidden_size = self.hidden_size.get()
            num_layers = self.num_layers.get()
            learning_rate = float(self.learning_rate.get())
            max_examples = (
                None
                if self.max_examples.get() == "Alle"
                else int(self.max_examples.get().replace(",", ""))
            )
            device = self._selected_device()
            examples = load_training_examples(
                Path(self.training_path.get()), max_examples=max_examples
            )
        except (OSError, ValueError) as error:
            messagebox.showerror("Trainingsdaten prüfen", str(error), parent=self.root)
            return

        if not examples:
            messagebox.showerror(
                "Keine Trainingsdaten",
                "Die ausgewählte Datei enthält keine Trainingsbeispiele.",
                parent=self.root,
            )
            return

        if len(examples) > 1000 and not messagebox.askyesno(
            "Großer Trainingsdatensatz",
            (
                f"Es wurden {len(examples):,} Beispiele geladen. "
                "Dieses kleine Modell verarbeitet Trainingsbeispiele einzeln; "
                "das Training kann deshalb sehr lange dauern. Trotzdem starten?"
            ),
            parent=self.root,
        ):
            return

        self.train_button.configure(state="disabled")
        self.settings_button.configure(state="disabled")
        self.generate_button.configure(state="disabled")
        self.status.set(
            f"Training mit {len(examples):,} Beispielen, {epochs} Epochen "
            f"auf {device.upper()} · H{hidden_size}/L{num_layers}/LR{learning_rate:g}…"
        )
        self.training_progress.set(0)
        self.training_time.set("Berechne Restzeit…")
        threading.Thread(
            target=self._train_in_background,
            args=(examples, epochs, device, hidden_size, num_layers, learning_rate),
            daemon=True,
        ).start()

    def _selected_device(self):
        if self.device.get() == "GPU":
            if not torch.cuda.is_available():
                raise ValueError(
                    "GPU-Training ist nicht verfügbar. Installiere eine CUDA-fähige "
                    "PyTorch-Version und prüfe, ob dein Rechner eine unterstützte "
                    "NVIDIA-GPU mit aktuellem Treiber hat."
                )
            return "cuda"
        if self.device.get() == "CPU":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _train_in_background(
        self, examples, epochs, device, hidden_size, num_layers, learning_rate
    ):
        try:
            model = train_model(
                examples,
                epochs,
                lambda completed, total, loss, remaining: self.messages.put(
                    ("progress", completed, total, loss, remaining)
                ),
                device=device,
                hidden_size=hidden_size,
                num_layers=num_layers,
                learning_rate=learning_rate,
            )
            checkpoint_path = model_checkpoint_path(
                hidden_size, num_layers, learning_rate
            )
            self.messages.put(("finished", model, str(checkpoint_path), None))
        except (OSError, RuntimeError, ValueError) as error:
            self.messages.put(("finished", None, None, str(error)))

    def _process_messages(self):
        try:
            while True:
                message = self.messages.get_nowait()
                self._handle_message(message)
        except queue.Empty:
            pass

        with self.activation_lock:
            activation = self.latest_activation
            self.latest_activation = None
        if activation is not None:
            self.last_activation = activation
            if self.visual_window is not None:
                self._update_visualization(activation)
        self.root.after(100, self._process_messages)

    def _handle_message(self, message):
        message_type, *details = message
        if message_type == "progress":
            completed, total, loss, remaining = details
            percentage = completed / total * 100
            self.training_progress.set(percentage)
            self.training_time.set(
                f"{percentage:.1f}% · Rest ca. {self._format_duration(remaining)} "
                f"· Fehler {loss:.4f}"
            )
        elif message_type == "finished":
            model, checkpoint_path, error = details
            self.train_button.configure(state="normal")
            self.settings_button.configure(state="normal")
            if error:
                self.training_time.set("Training fehlgeschlagen")
                self.generate_button.configure(
                    state="normal" if self.model is not None else "disabled"
                )
                self.status.set("Training fehlgeschlagen.")
                messagebox.showerror("Training fehlgeschlagen", error)
            else:
                self.model = model
                self.training_progress.set(100)
                self.training_time.set("100% · Training abgeschlossen")
                self.generate_button.configure(state="normal")
                self.status.set(
                    f"Training abgeschlossen und gespeichert: {Path(checkpoint_path).name}"
                )
                self._refresh_model_choices(checkpoint_path)
                self._load_selected_model()
        elif message_type == "generation_finished":
            (code,) = details
            self.generate_button.configure(state="normal")
            self.output.delete("1.0", "end")
            self.output.insert("1.0", code)
            self.status.set("Codegenerierung abgeschlossen.")
        elif message_type == "generation_failed":
            (error,) = details
            self.generate_button.configure(state="normal")
            self.status.set("Codegenerierung fehlgeschlagen.")
            messagebox.showerror("Codegenerierung fehlgeschlagen", error)

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, round(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours} Std. {minutes:02d} Min."
        if minutes:
            return f"{minutes} Min. {seconds:02d} Sek."
        return f"{seconds} Sek."

    def _open_visualization(self):
        if self.visual_window is not None and self.visual_window.winfo_exists():
            self.visual_window.lift()
            return

        window = tk.Toplevel(self.root)
        window.title("Cortex - Netzwerkaktivität")
        window.geometry("1080x640")
        window.minsize(1080, 620)
        window.protocol("WM_DELETE_WINDOW", self._close_visualization)
        self.visual_window = window

        ttk.Label(
            window,
            text=(
                "Aktivierungen des letzten Verarbeitungsschritts. "
                "Die Farbe zeigt die relative Stärke und das Vorzeichen."
            ),
            wraplength=1040,
        ).pack(anchor="w", padx=12, pady=(10, 4))

        canvas = tk.Canvas(window, background="#181a20", highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=10, pady=8)
        self.visual_canvas = canvas
        self.visual_nodes = {"embedding": [], "recurrent": [], "logits": []}

        canvas.create_text(
            130,
            26,
            text=f"Byte-Embedding · {self.hidden_size.get()}",
            fill="white",
            font=("Segoe UI", 11),
        )
        canvas.create_text(
            500,
            26,
            text=f"LSTM-Zustand · {self.num_layers.get()} Layer · {self.hidden_size.get()}",
            fill="white",
            font=("Segoe UI", 11),
        )
        canvas.create_text(
            875,
            26,
            text=f"Ausgabe-Logits · {VOCAB_SIZE}",
            fill="white",
            font=("Segoe UI", 11),
        )
        canvas.create_line(250, 230, 365, 230, fill="#7f8794", arrow=tk.LAST, width=2)
        canvas.create_line(620, 230, 735, 230, fill="#7f8794", arrow=tk.LAST, width=2)
        self.visual_status = canvas.create_text(
            540,
            525,
            text="Warte auf Codegenerierung…",
            fill="#d0d4dc",
            font=("Segoe UI", 10),
        )
        self._draw_node_grid("embedding", 55, 55, 8, 20)
        self._draw_node_grid("recurrent", 425, 55, 8, 20)
        self._draw_node_grid("logits", 735, 55, 17, 17)
        if self.last_activation is not None:
            self._update_visualization(self.last_activation)

    def _draw_node_grid(self, group, start_x, start_y, columns, spacing):
        radius = 6 if group == "logits" else 7
        node_count = VOCAB_SIZE if group == "logits" else self.hidden_size.get()
        for index in range(node_count):
            x = start_x + (index % columns) * spacing
            y = start_y + (index // columns) * 28
            node = self.visual_canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="#343943",
                outline="#686f7b",
            )
            self.visual_nodes[group].append(node)

    def _update_visualization(self, activation):
        embedding, recurrent, logits, token, step = activation
        self._color_nodes("embedding", embedding)
        self._color_nodes("recurrent", recurrent)
        self._color_nodes("logits", logits)
        for index, node in enumerate(self.visual_nodes["logits"]):
            self.visual_canvas.itemconfigure(
                node,
                outline="#ffe066" if index == token else "#686f7b",
                width=2 if index == token else 1,
            )
        token_text = "Ende" if token == EOS_TOKEN else f"{token} ({chr(token)!r})"
        self.visual_canvas.itemconfigure(
            self.visual_status,
            text=f"Generierungsschritt {step} · nächstes Byte: {token_text}",
        )

    def _color_nodes(self, group, values):
        maximum = max((abs(value) for value in values), default=0.0) or 1.0
        neutral = (52, 57, 67)
        for node, value in zip(self.visual_nodes[group], values):
            strength = min(abs(value) / maximum, 1.0)
            target = (242, 142, 43) if value >= 0 else (55, 126, 184)
            color = tuple(
                round(base + (end - base) * strength)
                for base, end in zip(neutral, target)
            )
            self.visual_canvas.itemconfigure(
                node,
                fill="#{:02x}{:02x}{:02x}".format(*color),
            )

    def _close_visualization(self):
        if self.visual_window is not None:
            self.visual_window.destroy()
        self.visual_window = None
        self.visual_nodes = {}

    def _generate(self):
        if self.model is None:
            messagebox.showwarning(
                "Modell noch nicht trainiert",
                "Trainiere zuerst das Modell mit passenden prompt/code-Beispielen.",
                parent=self.root,
            )
            return
        prompt = self.prompt.get("1.0", "end-1c")
        if not prompt.strip():
            messagebox.showerror(
                "Anforderung prüfen",
                "Gib zuerst eine Anforderung ein.",
                parent=self.root,
            )
            return
        if len(prompt) > 2000:
            messagebox.showerror(
                "Anforderung prüfen",
                "Die Anforderung darf höchstens 2000 Zeichen lang sein.",
                parent=self.root,
            )
            return

        self.generate_button.configure(state="disabled")
        self.status.set("Codegenerierung läuft…")
        with self.activation_lock:
            self.latest_activation = None
        threading.Thread(
            target=self._generate_in_background,
            args=(self.model, prompt),
            daemon=True,
        ).start()

    def _generate_in_background(self, model, prompt):
        try:
            code = generate_code(
                model,
                prompt,
                activation_callback=self._publish_activation,
            )
        except (RuntimeError, ValueError) as error:
            self.messages.put(("generation_failed", str(error)))
            return
        self.messages.put(("generation_finished", code))

    def _publish_activation(self, embedding, recurrent, logits, token, step):
        with self.activation_lock:
            self.latest_activation = (embedding, recurrent, logits, token, step)

    def _copy_code(self):
        code = self.output.get("1.0", "end-1c")
        if not code.strip():
            messagebox.showinfo("Kein Code", "Es gibt noch keinen Code zum Kopieren.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(code)
        self.status.set("Code in die Zwischenablage kopiert.")
