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
        self.visual_node_indices = {}
        self.visual_weights = None
        self.visual_weights_model_id = None
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
        window.geometry("1120x700")
        window.minsize(780, 520)
        window.protocol("WM_DELETE_WINDOW", self._close_visualization)
        self.visual_window = window

        canvas = tk.Canvas(window, background="#f5f5f7", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        self.visual_canvas = canvas
        self.visual_nodes = {}
        self.visual_node_indices = {}
        canvas.bind("<Configure>", self._render_visualization)
        self._render_visualization()

    def _render_visualization(self, event=None):
        if self.visual_window is None or not self.visual_window.winfo_exists():
            return

        canvas = self.visual_canvas
        width = event.width if event is not None else canvas.winfo_width()
        height = event.height if event is not None else canvas.winfo_height()
        if width < 200 or height < 200:
            return

        if self.last_activation is None:
            hidden_size = self.model.hidden_size if self.model else self.hidden_size.get()
            layer_count = self.model.num_layers if self.model else self.num_layers.get()
            embedding = [0.0] * hidden_size
            layers = [[0.0] * hidden_size for _ in range(layer_count)]
            logits = [0.0] * VOCAB_SIZE
            token = None
            step = None
        else:
            embedding, layers, logits, token, step, _previous_layers = (
                self.last_activation
            )
            hidden_size = len(embedding)
            layer_count = len(layers)

        canvas.delete("all")
        self.visual_nodes = {}
        self.visual_node_indices = {}
        canvas.create_text(
            24,
            22,
            text="Netzwerkaktivität",
            anchor="w",
            fill="#1d1d1f",
            font=("Segoe UI", 16, "bold"),
        )
        canvas.create_text(
            24,
            48,
            text="Aktivierungen für das jeweils nächste erzeugte Byte",
            anchor="w",
            fill="#6e6e73",
            font=("Segoe UI", 10),
        )

        columns = [("embedding", "Embedding", embedding)]
        columns.extend(
            (f"layer_{index}", f"LSTM {index + 1}", values)
            for index, values in enumerate(layers)
        )
        columns.append(("logits", "Byte-Ausgabe", logits))

        margin = 30
        column_width = (width - 2 * margin) / len(columns)
        node_top = 118
        node_bottom = height - 76
        max_visible = max(6, min(24, int((width - 2 * margin) / len(columns) / 4)))
        points = {}
        for column_index, (group, title, values) in enumerate(columns):
            center_x = margin + (column_index + 0.5) * column_width
            canvas.create_text(
                center_x,
                82,
                text=title,
                fill="#1d1d1f",
                font=("Segoe UI", 10, "bold"),
            )
            canvas.create_line(
                center_x,
                96,
                center_x,
                height - 54,
                fill="#e5e5ea",
                width=1,
            )
            required_indices = (
                [token] if group == "logits" and token is not None else []
            )
            neuron_indices = self._visible_neuron_indices(
                values, max_visible, required_indices
            )
            self.visual_node_indices[group] = neuron_indices
            self.visual_nodes[group] = []
            count = len(neuron_indices)
            for row, neuron_index in enumerate(neuron_indices):
                y = node_top + (row + 0.5) * (node_bottom - node_top) / count
                points[(group, neuron_index)] = (center_x, y)

        self._draw_active_connections(columns, points)
        for group, _title, values in columns:
            self._draw_neurons(group, values, points)

        self.visual_status = canvas.create_text(
            24,
            height - 25,
            text=(
                "Je Zielneuron: die stärksten Beiträge der angezeigten Neuronen. "
                "Orange/blau = positiver/negativer Beitrag."
            ),
            anchor="w",
            fill="#6e6e73",
            font=("Segoe UI", 9),
        )
        if step is not None:
            token_text = "Ende" if token == EOS_TOKEN else f"{token} ({chr(token)!r})"
            canvas.itemconfigure(
                self.visual_status,
                text=(
                    f"Schritt {step} · nächstes Byte: {token_text} · "
                    "Linien zeigen die stärksten gewichteten Verbindungen"
                ),
            )

    @staticmethod
    def _visible_neuron_indices(values, limit, required_indices=()):
        strongest = sorted(
            range(len(values)),
            key=lambda index: abs(values[index]),
            reverse=True,
        )[:limit]
        selected = sorted(strongest)
        for index in required_indices:
            if index not in selected:
                selected[-1:] = [index] if selected else [index]
        return sorted(selected)

    def _draw_active_connections(self, columns, points):
        if self.model is None:
            return
        self._prepare_visualization_weights()
        if self.visual_weights is None:
            return

        for source_column, target_column in zip(columns, columns[1:]):
            source_group, _source_title, source_values = source_column
            target_group = target_column[0]
            source_indices = self.visual_node_indices[source_group]
            target_indices = self.visual_node_indices[target_group]
            if target_group == "logits":
                weights = self.visual_weights["output"]
                source_tensor = torch.tensor(
                    [source_values[index] for index in source_indices],
                    dtype=weights.dtype,
                )
                contributions = (
                    weights.index_select(0, torch.tensor(target_indices))
                    .index_select(1, torch.tensor(source_indices))
                    .mul(source_tensor)
                    .tolist()
                )
                scores = [[abs(value) for value in row] for row in contributions]
            else:
                layer_index = int(target_group.split("_")[1])
                weights = self.visual_weights["layers"][layer_index]
                hidden_size = self.model.hidden_size
                gate_rows = torch.tensor(
                    [
                        gate * hidden_size + target_index
                        for gate in range(4)
                        for target_index in target_indices
                    ]
                )
                selected_weights = (
                    weights.index_select(0, gate_rows)
                    .index_select(1, torch.tensor(source_indices))
                    .reshape(4, len(target_indices), len(source_indices))
                )
                source_tensor = torch.tensor(
                    [source_values[index] for index in source_indices],
                    dtype=weights.dtype,
                )
                contributions = (
                    selected_weights.sum(dim=0).mul(source_tensor).tolist()
                )
                scores = (
                    selected_weights.abs()
                    .sum(dim=0)
                    .mul(source_tensor.abs())
                    .tolist()
                )

            strongest_scores = sorted(
                (score for row in scores for score in row), reverse=True
            )
            maximum = strongest_scores[0] if strongest_scores else 0
            if maximum == 0:
                continue
            for target_position, target_index in enumerate(target_indices):
                strongest = sorted(
                    range(len(source_indices)),
                    key=lambda position: scores[target_position][position],
                    reverse=True,
                )[: min(2, len(source_indices))]
                for source_position in strongest:
                    strength = scores[target_position][source_position] / maximum
                    if strength <= 0:
                        continue
                    contribution = contributions[target_position][source_position]
                    color = "#d17a32" if contribution >= 0 else "#3478b8"
                    start = points[(source_group, source_indices[source_position])]
                    end = points[(target_group, target_index)]
                    self.visual_canvas.create_line(
                        *start,
                        *end,
                        fill=color,
                        width=0.8 + 1.8 * strength,
                        arrow=tk.LAST,
                        arrowshape=(5, 7, 3),
                        tags=("active_connection",),
                    )

        previous_layers = (
            self.last_activation[5]
            if self.last_activation is not None
            else None
        )
        if previous_layers is None:
            return

        for layer_index, previous_values in enumerate(previous_layers):
            group = f"layer_{layer_index}"
            neuron_indices = self.visual_node_indices[group]
            weights = self.visual_weights["recurrent"][layer_index]
            hidden_size = self.model.hidden_size
            gate_rows = torch.tensor(
                [
                    gate * hidden_size + target_index
                    for gate in range(4)
                    for target_index in neuron_indices
                ]
            )
            selected_weights = (
                weights.index_select(0, gate_rows)
                .index_select(1, torch.tensor(neuron_indices))
                .reshape(4, len(neuron_indices), len(neuron_indices))
            )
            source_tensor = torch.tensor(
                [previous_values[index] for index in neuron_indices],
                dtype=weights.dtype,
            )
            contributions = (
                selected_weights.sum(dim=0).mul(source_tensor).tolist()
            )
            scores = (
                selected_weights.abs()
                .sum(dim=0)
                .mul(source_tensor.abs())
                .tolist()
            )
            strongest_scores = sorted(
                (score for row in scores for score in row), reverse=True
            )
            maximum = strongest_scores[0] if strongest_scores else 0
            if maximum == 0:
                continue
            center_x = points[(group, neuron_indices[0])][0]
            loop_x = center_x + 8
            for target_position, target_index in enumerate(neuron_indices):
                strongest = sorted(
                    range(len(neuron_indices)),
                    key=lambda position: scores[target_position][position],
                    reverse=True,
                )[: min(2, len(neuron_indices))]
                for source_position in strongest:
                    strength = scores[target_position][source_position] / maximum
                    if strength <= 0:
                        continue
                    source_index = neuron_indices[source_position]
                    contribution = contributions[target_position][source_position]
                    color = "#d17a32" if contribution >= 0 else "#3478b8"
                    start_y = points[(group, source_index)][1]
                    end_y = points[(group, target_index)][1]
                    self.visual_canvas.create_line(
                        center_x + 4,
                        start_y,
                        loop_x,
                        start_y,
                        loop_x,
                        end_y,
                        center_x + 4,
                        end_y,
                        smooth=True,
                        splinesteps=12,
                        fill=color,
                        width=0.8 + 1.8 * strength,
                        arrow=tk.LAST,
                        arrowshape=(5, 7, 3),
                        tags=("active_connection", "recurrent_connection"),
                    )

    def _prepare_visualization_weights(self):
        if self.visual_weights_model_id == id(self.model):
            return
        self.visual_weights_model_id = id(self.model)
        self.visual_weights = None
        if self.model is None:
            return
        self.visual_weights = {
            "layers": [
                getattr(self.model.lstm, f"weight_ih_l{index}")
                .detach()
                .cpu()
                for index in range(self.model.num_layers)
            ],
            "recurrent": [
                getattr(self.model.lstm, f"weight_hh_l{index}")
                .detach()
                .cpu()
                for index in range(self.model.num_layers)
            ],
            "output": self.model.output.weight.detach().cpu(),
        }

    def _draw_neurons(self, group, values, points):
        radius = 4
        for neuron_index in self.visual_node_indices[group]:
            x, y = points[(group, neuron_index)]
            is_selected_output = (
                group == "logits"
                and self.last_activation is not None
                and neuron_index == self.last_activation[3]
            )
            node = self.visual_canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="#e5e5ea",
                outline="#1d1d1f" if is_selected_output else "#ffffff",
                width=2 if is_selected_output else 1,
            )
            self.visual_nodes[group].append(node)
        self._color_nodes(group, values)

    def _update_visualization(self, activation):
        self.last_activation = activation
        self._render_visualization()

    def _color_nodes(self, group, values):
        maximum = max((abs(value) for value in values), default=0.0) or 1.0
        neutral = (229, 229, 234)
        for node, index in zip(
            self.visual_nodes[group], self.visual_node_indices[group]
        ):
            value = values[index]
            strength = min(abs(value) / maximum, 1.0)
            target = (229, 138, 53) if value >= 0 else (52, 120, 184)
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
        self.visual_node_indices = {}
        self.visual_weights = None
        self.visual_weights_model_id = None

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
                network_callback=self._publish_network_activation,
            )
        except (RuntimeError, ValueError) as error:
            self.messages.put(("generation_failed", str(error)))
            return
        self.messages.put(("generation_finished", code))

    def _publish_network_activation(
        self, embedding, layers, logits, token, step, previous_layers
    ):
        with self.activation_lock:
            self.latest_activation = (
                embedding,
                layers,
                logits,
                token,
                step,
                previous_layers,
            )

    def _copy_code(self):
        code = self.output.get("1.0", "end-1c")
        if not code.strip():
            messagebox.showinfo("Kein Code", "Es gibt noch keinen Code zum Kopieren.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(code)
        self.status.set("Code in die Zwischenablage kopiert.")
