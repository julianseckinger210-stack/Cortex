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
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_GRADIENT_CLIP,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_NEW_BYTES,
    DEFAULT_NUM_LAYERS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TARGET_LABEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_VALIDATION_SPLIT,
    EOS_TOKEN,
    LEGACY_MODEL_PATH,
    LEGACY_ROOT_MODEL_PATH,
    MODEL_DIR,
    VOCAB_SIZE,
    CodeModel,
    available_models,
    checkpoint_settings,
    generate_code,
    load_training_dataset,
    model_checkpoint_path,
    train_model,
)


def _decode_escapes(text):
    result = []
    mapping = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\"}
    index = 0
    while index < len(text):
        character = text[index]
        if character == "\\" and index + 1 < len(text):
            following = text[index + 1]
            if following in mapping:
                result.append(mapping[following])
                index += 2
                continue
        result.append(character)
        index += 1
    return "".join(result)


def _encode_escapes(text):
    return (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )


def _parse_stop_sequences(text):
    sequences = []
    for part in text.split(";"):
        value = _decode_escapes(part.strip())
        if value:
            sequences.append(value)
    return sequences


def _parse_seed(text):
    value = text.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            "Der Seed muss eine ganze Zahl sein (oder leer bleiben)."
        ) from error


class CodeGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cortex - Sprachmodell")
        self.root.geometry("900x780")
        self.model = None
        self.target_label = DEFAULT_TARGET_LABEL
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
        self.dropout = tk.DoubleVar(value=DEFAULT_DROPOUT)
        self.batch_size = tk.IntVar(value=DEFAULT_BATCH_SIZE)
        self.shuffle = tk.BooleanVar(value=True)
        self.gradient_clip = tk.DoubleVar(value=DEFAULT_GRADIENT_CLIP)
        self.validation_split = tk.DoubleVar(value=DEFAULT_VALIDATION_SPLIT * 100)
        self.training_seed = tk.StringVar(value="")
        self.generation_mode = tk.StringVar(value="Sampling")
        self.temperature = tk.DoubleVar(value=DEFAULT_TEMPERATURE)
        self.top_k = tk.IntVar(value=DEFAULT_TOP_K)
        self.top_p = tk.DoubleVar(value=DEFAULT_TOP_P)
        self.repetition_penalty = tk.DoubleVar(value=DEFAULT_REPETITION_PENALTY)
        self.max_new_bytes = tk.IntVar(value=DEFAULT_MAX_NEW_BYTES)
        self.stop_sequences = tk.StringVar(value="")
        self.generation_seed = tk.StringVar(value="")
        self.generation_summary = tk.StringVar(value="")
        self.status = tk.StringVar(value="Kein trainiertes Modell geladen.")
        self.training_progress = tk.DoubleVar(value=0)
        self.training_time = tk.StringVar(value="Noch kein Training")
        self.training_metrics = tk.StringVar(value="")
        self.messages = queue.Queue()
        self.activation_lock = threading.Lock()
        self.latest_activation = None
        self.last_activation = None
        self.visual_window = None
        self.visual_nodes = {}
        self.visual_node_indices = {}
        self.visual_weights = None
        self.visual_weights_model_id = None
        for variable in (
            self.generation_mode,
            self.temperature,
            self.top_k,
            self.top_p,
            self.repetition_penalty,
            self.max_new_bytes,
            self.stop_sequences,
            self.generation_seed,
        ):
            variable.trace_add(
                "write", lambda *_args: self._update_generation_summary()
            )
        self._update_generation_summary()
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
            text="Eigenes Sprachmodell",
            font=("Segoe UI", 18, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            frame,
            text=(
                "Dieses kleine Zeichenmodell wird lokal von Grund auf trainiert. "
                "Es ist ein Lernprototyp, kein leistungsfähiges Sprachmodell."
            ),
            wraplength=760,
        ).grid(row=1, column=0, sticky="w", pady=(4, 14))

        self._build_training_controls(frame)
        ttk.Label(frame, text="Anforderung / Frage").grid(
            row=4, column=0, sticky="w"
        )
        self.prompt = ScrolledText(frame, height=5, wrap="word")
        self.prompt.grid(row=5, column=0, sticky="nsew", pady=(4, 12))

        self._build_output_controls(frame)
        ttk.Label(frame, text="Ausgabe (vor der Verwendung prüfen):").grid(
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
                'Jede JSONL-Zeile: {"prompt": "Frage", "response": "Antwort"} '
                'oder {"prompt": "Anforderung", "code": "Python-Code"}.'
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
            to=300,
            variable=self.epochs,
            command=self._update_epoch_label,
        )
        self.epoch_scale.grid(row=0, column=1, sticky="ew", padx=8)
        self.epoch_label = ttk.Label(train_frame, text="5")
        self.epoch_label.grid(row=0, column=2, sticky="w")
        ttk.Label(
            train_frame,
            text="Wie oft das Modell alle Beispiele sieht. Mehr = lernt stärker, "
            "kann aber auswendig lernen.",
            wraplength=420,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
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
            train_frame, text="Training starten", command=self._start_training
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
        ttk.Label(train_frame, textvariable=self.training_metrics).grid(
            row=5, column=0, columnspan=5, sticky="w", pady=(4, 0)
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
        window.geometry("760x700")
        window.transient(self.root)
        window.grab_set()

        hidden_size = tk.IntVar(value=self.hidden_size.get())
        num_layers = tk.IntVar(value=self.num_layers.get())
        learning_rate = tk.StringVar(value=self.learning_rate.get())
        dropout = tk.DoubleVar(value=self.dropout.get())
        batch_size = tk.IntVar(value=self.batch_size.get())
        shuffle = tk.BooleanVar(value=self.shuffle.get())
        gradient_clip = tk.DoubleVar(value=self.gradient_clip.get())
        validation_split = tk.DoubleVar(value=self.validation_split.get())
        training_seed = tk.StringVar(value=self.training_seed.get())
        generation_mode = tk.StringVar(value=self.generation_mode.get())
        temperature = tk.DoubleVar(value=self.temperature.get())
        top_k = tk.IntVar(value=self.top_k.get())
        top_p = tk.DoubleVar(value=self.top_p.get())
        repetition_penalty = tk.DoubleVar(value=self.repetition_penalty.get())
        max_new_bytes = tk.IntVar(value=self.max_new_bytes.get())
        stop_sequences = tk.StringVar(value=self.stop_sequences.get())
        generation_seed = tk.StringVar(value=self.generation_seed.get())

        original_architecture = (
            self.hidden_size.get(),
            self.num_layers.get(),
            float(self.learning_rate.get()),
        )

        notebook = ttk.Notebook(window)
        notebook.pack(fill="both", expand=True, padx=12, pady=(12, 0))
        architecture_tab = ttk.Frame(notebook, padding=14)
        training_tab = ttk.Frame(notebook, padding=14)
        generation_tab = ttk.Frame(notebook, padding=14)
        notebook.add(architecture_tab, text="Architektur")
        notebook.add(training_tab, text="Training")
        notebook.add(generation_tab, text="Generierung")

        def add_field(tab, row, label, widget, description):
            ttk.Label(tab, text=label).grid(
                row=row * 2, column=0, sticky="w", pady=(10, 0)
            )
            widget.grid(
                row=row * 2, column=1, sticky="w", padx=(14, 0), pady=(10, 0)
            )
            ttk.Label(
                tab,
                text=description,
                wraplength=540,
                foreground="#555555",
            ).grid(
                row=row * 2 + 1,
                column=0,
                columnspan=2,
                sticky="w",
                pady=(2, 0),
            )

        add_field(
            architecture_tab,
            0,
            "Hidden-Größe:",
            ttk.Spinbox(
                architecture_tab,
                from_=32,
                to=512,
                increment=32,
                textvariable=hidden_size,
                width=10,
            ),
            "Anzahl der Neuronen pro LSTM-Layer. Mehr Neuronen können mehr Muster "
            "lernen, machen das Training aber langsamer und die Checkpoints größer.",
        )
        add_field(
            architecture_tab,
            1,
            "LSTM-Layer:",
            ttk.Spinbox(
                architecture_tab,
                from_=1,
                to=6,
                increment=1,
                textvariable=num_layers,
                width=10,
            ),
            "Anzahl der übereinanderliegenden LSTM-Schichten. Mehr Layer können "
            "komplexere Zusammenhänge lernen, brauchen aber deutlich mehr Zeit.",
        )
        add_field(
            architecture_tab,
            2,
            "Lernrate:",
            ttk.Combobox(
                architecture_tab,
                textvariable=learning_rate,
                values=("0.001", "0.003", "0.005", "0.01"),
                state="readonly",
                width=10,
            ),
            "Schrittweite der Gewichtsänderung beim Training. Zu groß = instabil, "
            "zu klein = lernt kaum. Sinnvoll sind 0.001 bis 0.01.",
        )
        add_field(
            architecture_tab,
            3,
            "Dropout:",
            ttk.Spinbox(
                architecture_tab,
                from_=0.0,
                to=0.5,
                increment=0.05,
                textvariable=dropout,
                width=10,
            ),
            "Anteil der Verbindungen, der beim Training zufällig abgeschaltet wird "
            "(0 bis 0.5). Wirkt Überanpassung entgegen; 0 = aus. Der Wert steckt "
            "im Dateinamen des Checkpoints.",
        )

        add_field(
            training_tab,
            0,
            "Beispiele pro Update:",
            ttk.Spinbox(
                training_tab,
                from_=1,
                to=64,
                increment=1,
                textvariable=batch_size,
                width=10,
            ),
            "Wie viele Trainings-Chunks gesammelt werden, bevor die Gewichte "
            "aktualisiert werden (Gradient Accumulation). Höher = stabilere "
            "Updates, aber weniger Updates pro Epoche.",
        )
        add_field(
            training_tab,
            1,
            "Reihenfolge mischen:",
            ttk.Checkbutton(training_tab, variable=shuffle, text="Aktiv"),
            "Mischt die Beispiele vor jeder Epoche neu. Verhindert, dass das Modell "
            "sich die Reihenfolge der Datei einprägt.",
        )
        add_field(
            training_tab,
            2,
            "Gradient-Clip:",
            ttk.Spinbox(
                training_tab,
                from_=0.0,
                to=10.0,
                increment=0.5,
                textvariable=gradient_clip,
                width=10,
            ),
            "Begrenzt die Länge der Gradienten pro Update. Verhindert, dass einzelne "
            "Beispiele das Training überschreiben. 0 = aus, 1.0 ist ein guter Start.",
        )
        add_field(
            training_tab,
            3,
            "Validierung:",
            ttk.Spinbox(
                training_tab,
                from_=0,
                to=30,
                increment=5,
                textvariable=validation_split,
                width=10,
            ),
            "Anteil der Beispiele in Prozent, der nicht trainiert, sondern nur zur "
            "Kontrolle genutzt wird (0 bis 30). Der Validierungsfehler zeigt, ob "
            "das Modell auswendig lernt.",
        )
        add_field(
            training_tab,
            4,
            "Seed (Training):",
            ttk.Entry(training_tab, textvariable=training_seed, width=14),
            "Startwert für Zufall (Split, Mischen, Initialisierung). Leer = jedes "
            "Mal anders; gleicher Wert = reproduzierbares Training.",
        )

        add_field(
            generation_tab,
            0,
            "Modus:",
            ttk.Combobox(
                generation_tab,
                textvariable=generation_mode,
                values=("Sampling", "Greedy"),
                state="readonly",
                width=12,
            ),
            "Greedy nimmt immer das wahrscheinlichste Byte (deterministisch, neigt zu "
            "Schleifen). Sampling zieht zufällig aus den wahrscheinlichsten Bytes.",
        )
        add_field(
            generation_tab,
            1,
            "Temperature:",
            ttk.Spinbox(
                generation_tab,
                from_=0.05,
                to=2.0,
                increment=0.05,
                textvariable=temperature,
                width=10,
            ),
            "Streuung beim Sampling. Kleine Werte (0.2) sind vorhersehbar, große "
            "Werte (1.2+) wirken kreativer, machen aber mehr Fehler.",
        )
        add_field(
            generation_tab,
            2,
            "Top-k:",
            ttk.Spinbox(
                generation_tab,
                from_=0,
                to=257,
                increment=5,
                textvariable=top_k,
                width=10,
            ),
            "Nur die k wahrscheinlichsten Bytes kommen infrage. 0 = aus "
            "(alle Bytes erlaubt).",
        )
        add_field(
            generation_tab,
            3,
            "Top-p:",
            ttk.Spinbox(
                generation_tab,
                from_=0.05,
                to=1.0,
                increment=0.05,
                textvariable=top_p,
                width=10,
            ),
            "Nur Bytes bis zu dieser kumulierten Wahrscheinlichkeit kommen infrage "
            "(Nucleus Sampling). 1.0 = aus. Nicht gleichzeitig mit Top-k stark "
            "einschränken.",
        )
        add_field(
            generation_tab,
            4,
            "Repetition-Penalty:",
            ttk.Spinbox(
                generation_tab,
                from_=1.0,
                to=2.0,
                increment=0.05,
                textvariable=repetition_penalty,
                width=10,
            ),
            "Bestraft bereits erzeugte Bytes, damit Code nicht in Schleifen "
            "wiederholt wird. 1.0 = aus, 1.1 bis 1.3 wirken deutlich. Leerzeichen "
            "und Zeilenumbrüche bleiben unbestraft.",
        )
        add_field(
            generation_tab,
            5,
            "Max. Bytes:",
            ttk.Spinbox(
                generation_tab,
                from_=16,
                to=4000,
                increment=32,
                textvariable=max_new_bytes,
                width=10,
            ),
            "Maximale Länge der Ausgabe in Bytes. 1 Byte entspricht ungefähr einem "
            "Zeichen. Kürzere Grenzen machen die Generierung schneller.",
        )
        add_field(
            generation_tab,
            6,
            "Stop-Sequenzen:",
            ttk.Entry(generation_tab, textvariable=stop_sequences, width=42),
            "Textstellen, bei denen die Generierung endet, z. B. \\n### . Mehrere "
            "mit ; trennen. \\n und \\t werden als Zeilenumbruch bzw. Tab gelesen.",
        )
        add_field(
            generation_tab,
            7,
            "Seed (Generierung):",
            ttk.Entry(generation_tab, textvariable=generation_seed, width=14),
            "Startwert für das Sampling. Gleicher Wert und gleiche Einstellungen "
            "ergeben dieselbe Ausgabe; leer = jedes Mal anders.",
        )

        def apply_settings():
            try:
                new_hidden_size = int(hidden_size.get())
                new_num_layers = int(num_layers.get())
                new_learning_rate = float(learning_rate.get())
                new_dropout = float(dropout.get())
                new_batch_size = int(batch_size.get())
                new_gradient_clip = float(gradient_clip.get())
                new_validation_split = float(validation_split.get())
                new_temperature = float(temperature.get())
                new_top_k = int(top_k.get())
                new_top_p = float(top_p.get())
                new_repetition_penalty = float(repetition_penalty.get())
                new_max_new_bytes = int(max_new_bytes.get())
                new_training_seed = _parse_seed(training_seed.get())
                new_generation_seed = _parse_seed(generation_seed.get())
                new_stop_sequences = _parse_stop_sequences(stop_sequences.get())

                if not 32 <= new_hidden_size <= 512:
                    raise ValueError("Hidden-Größe muss zwischen 32 und 512 liegen.")
                if not 1 <= new_num_layers <= 6:
                    raise ValueError("Layerzahl muss zwischen 1 und 6 liegen.")
                if not 0.0001 <= new_learning_rate <= 0.1:
                    raise ValueError("Lernrate muss zwischen 0.0001 und 0.1 liegen.")
                if not 0.0 <= new_dropout <= 0.5:
                    raise ValueError("Dropout muss zwischen 0 und 0.5 liegen.")
                if not 1 <= new_batch_size <= 64:
                    raise ValueError(
                        "Beispiele pro Update müssen zwischen 1 und 64 liegen."
                    )
                if not 0.0 <= new_gradient_clip <= 10.0:
                    raise ValueError(
                        "Gradient-Clip muss zwischen 0 und 10 liegen (0 = aus)."
                    )
                if not 0.0 <= new_validation_split <= 30.0:
                    raise ValueError(
                        "Der Validierungsanteil muss zwischen 0 und 30 % liegen."
                    )
                if not 0.05 <= new_temperature <= 2.0:
                    raise ValueError("Temperature muss zwischen 0.05 und 2.0 liegen.")
                if not 0 <= new_top_k <= 257:
                    raise ValueError("Top-k muss zwischen 0 (aus) und 257 liegen.")
                if not 0.05 <= new_top_p <= 1.0:
                    raise ValueError("Top-p muss zwischen 0.05 und 1.0 liegen.")
                if not 1.0 <= new_repetition_penalty <= 2.0:
                    raise ValueError(
                        "Repetition-Penalty muss zwischen 1.0 und 2.0 liegen."
                    )
                if not 16 <= new_max_new_bytes <= 4000:
                    raise ValueError(
                        "Max. Bytes müssen zwischen 16 und 4000 liegen."
                    )
                if len(new_stop_sequences) > 8:
                    raise ValueError("Es sind höchstens 8 Stop-Sequenzen erlaubt.")
                if any(len(stop) > 64 for stop in new_stop_sequences):
                    raise ValueError(
                        "Stop-Sequenzen dürfen höchstens 64 Zeichen lang sein."
                    )
            except (ValueError, tk.TclError) as error:
                messagebox.showerror(
                    "Ungültige Modell-Settings", str(error), parent=window
                )
                return

            self.hidden_size.set(new_hidden_size)
            self.num_layers.set(new_num_layers)
            self.learning_rate.set(format(new_learning_rate, "g"))
            self.dropout.set(round(new_dropout, 3))
            self.batch_size.set(new_batch_size)
            self.shuffle.set(bool(shuffle.get()))
            self.gradient_clip.set(new_gradient_clip)
            self.validation_split.set(new_validation_split)
            self.training_seed.set(
                "" if new_training_seed is None else str(new_training_seed)
            )
            self.generation_mode.set(generation_mode.get())
            self.temperature.set(new_temperature)
            self.top_k.set(new_top_k)
            self.top_p.set(new_top_p)
            self.repetition_penalty.set(new_repetition_penalty)
            self.max_new_bytes.set(new_max_new_bytes)
            self.stop_sequences.set(
                "; ".join(_encode_escapes(stop) for stop in new_stop_sequences)
            )
            self.generation_seed.set(
                "" if new_generation_seed is None else str(new_generation_seed)
            )

            architecture_changed = (
                new_hidden_size,
                new_num_layers,
                new_learning_rate,
            ) != original_architecture
            if architecture_changed:
                self.selected_model.set("Neues Modell (aktuelle Settings)")
                self._load_selected_model()
            window.destroy()

        buttons = ttk.Frame(window)
        buttons.pack(fill="x", padx=12, pady=12)
        ttk.Button(buttons, text="Übernehmen", command=apply_settings).pack(
            side="left"
        )
        ttk.Button(buttons, text="Abbrechen", command=window.destroy).pack(
            side="right"
        )

    def _build_output_controls(self, parent):
        output_frame = ttk.Frame(parent)
        output_frame.grid(row=6, column=0, sticky="ew")
        output_frame.columnconfigure(0, weight=1)
        self.generate_button = ttk.Button(
            output_frame, text="Antwort erzeugen", command=self._generate
        )
        self.generate_button.grid(row=0, column=0, sticky="w")
        ttk.Button(
            output_frame, text="Netzwerk ansehen", command=self._open_visualization
        ).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(output_frame, text="Ausgabe kopieren", command=self._copy_code).grid(
            row=0, column=2, padx=(8, 0)
        )
        ttk.Label(
            output_frame,
            textvariable=self.generation_summary,
            foreground="#555555",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

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
                DEFAULT_DROPOUT,
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
            self.dropout.set(DEFAULT_DROPOUT)
            self._load_selected_model()
            return

        path = MODEL_DIR / selected
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint.get("version") not in (2, 3):
                raise ValueError("Die Datei enthält keine gültigen Modell-Settings.")
            settings = checkpoint_settings(checkpoint)
            hidden_size = settings["hidden_size"]
            num_layers = settings["num_layers"]
            learning_rate = settings["learning_rate"]
            if (
                not isinstance(hidden_size, int)
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
        self.dropout.set(settings["dropout"])
        self._apply_training_settings(settings["training"])
        self._apply_generation_settings(settings["generation"])
        self._load_selected_model()

    def _apply_training_settings(self, settings):
        self.batch_size.set(int(settings["batch_size"]))
        self.shuffle.set(bool(settings["shuffle"]))
        self.gradient_clip.set(float(settings["gradient_clip"]))
        self.validation_split.set(round(float(settings["validation_split"]) * 100))
        seed = settings["seed"]
        self.training_seed.set("" if seed is None else str(seed))

    def _apply_generation_settings(self, settings):
        self.generation_mode.set(
            "Greedy" if settings["mode"] == "greedy" else "Sampling"
        )
        self.temperature.set(float(settings["temperature"]))
        self.top_k.set(int(settings["top_k"]))
        self.top_p.set(float(settings["top_p"]))
        self.repetition_penalty.set(float(settings["repetition_penalty"]))
        self.max_new_bytes.set(int(settings["max_new_bytes"]))
        self.stop_sequences.set(
            "; ".join(_encode_escapes(stop) for stop in settings["stop_sequences"])
        )
        seed = settings["seed"]
        self.generation_seed.set("" if seed is None else str(seed))

    def _generation_settings(self):
        return {
            "mode": self.generation_mode.get(),
            "temperature": float(self.temperature.get()),
            "top_k": int(self.top_k.get()),
            "top_p": float(self.top_p.get()),
            "repetition_penalty": float(self.repetition_penalty.get()),
            "max_new_bytes": int(self.max_new_bytes.get()),
            "stop_sequences": _parse_stop_sequences(self.stop_sequences.get()),
            "seed": _parse_seed(self.generation_seed.get()),
        }

    def _update_generation_summary(self):
        try:
            if self.generation_mode.get() == "Greedy":
                summary = "Greedy (immer das wahrscheinlichste Byte)"
            else:
                summary = (
                    f"Sampling · Temperature {self.temperature.get():.2f} · "
                    f"Top-k {self.top_k.get()} · Top-p {self.top_p.get():.2f}"
                )
            summary += (
                f" · Repetition-Penalty {self.repetition_penalty.get():.2f} · "
                f"max. {self.max_new_bytes.get()} Bytes"
            )
            if _parse_stop_sequences(self.stop_sequences.get()):
                summary += " · Stop-Sequenzen aktiv"
            if _parse_seed(self.generation_seed.get()) is not None:
                summary += " · Seed gesetzt"
        except (tk.TclError, ValueError):
            summary = ""
        self.generation_summary.set(summary)

    def _load_selected_model(self):
        hidden_size = self.hidden_size.get()
        num_layers = self.num_layers.get()
        learning_rate = float(self.learning_rate.get())
        dropout = float(self.dropout.get())
        checkpoint_path = model_checkpoint_path(
            hidden_size, num_layers, learning_rate, dropout
        )
        selected = self.selected_model.get()
        if selected == "Neues Modell (aktuelle Settings)":
            self.model = None
            self.target_label = DEFAULT_TARGET_LABEL
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
            version = checkpoint.get("version")
            if version not in (1, 2, 3):
                raise ValueError("Unbekannte Modellversion.")
            if version in (2, 3):
                stored_dropout = round(float(checkpoint.get("dropout") or 0.0), 3)
                if (
                    checkpoint.get("hidden_size") != hidden_size
                    or checkpoint.get("num_layers") != num_layers
                    or checkpoint.get("learning_rate") != learning_rate
                    or stored_dropout != round(dropout, 3)
                ):
                    raise ValueError(
                        "Checkpoint und ausgewählte Modell-Settings weichen ab."
                    )
            model = CodeModel(hidden_size, num_layers, dropout)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
        except (OSError, RuntimeError, KeyError, ValueError) as error:
            messagebox.showerror(
                "Modell konnte nicht geladen werden",
                f"Die gespeicherte Modelldatei ist ungültig:\n{error}",
                parent=self.root,
            )
            return
        if version in (2, 3):
            try:
                settings = checkpoint_settings(checkpoint)
            except (TypeError, ValueError):
                settings = None
            if settings is not None:
                self._apply_training_settings(settings["training"])
                self._apply_generation_settings(settings["generation"])
                self.target_label = settings.get(
                    "target_label", DEFAULT_TARGET_LABEL
                )
        self.model = model
        self.generate_button.configure(state="normal")
        self.status.set(
            f"Trainiertes Modell geladen: {path.name} "
            f"(Format: {self.target_label})"
        )

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
            config = {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "learning_rate": learning_rate,
                "dropout": float(self.dropout.get()),
                "batch_size": int(self.batch_size.get()),
                "shuffle": bool(self.shuffle.get()),
                "gradient_clip": float(self.gradient_clip.get()),
                "validation_split": float(self.validation_split.get()) / 100,
                "seed": _parse_seed(self.training_seed.get()),
                "generation_settings": self._generation_settings(),
            }
            device = self._selected_device()
            examples, target_label = load_training_dataset(
                Path(self.training_path.get()), max_examples=max_examples
            )
            config["target_label"] = target_label
            self.target_label = target_label
        except (OSError, ValueError, tk.TclError) as error:
            messagebox.showerror(
                "Einstellungen prüfen", str(error), parent=self.root
            )
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
        checkpoint_path = model_checkpoint_path(
            hidden_size,
            num_layers,
            learning_rate,
            config["dropout"],
        )
        resume_hint = (
            " · vorhandener Checkpoint wird fortgesetzt"
            if checkpoint_path.exists()
            else ""
        )
        self.status.set(
            f"Training mit {len(examples):,} Beispielen, {epochs} Epochen "
            f"auf {device.upper()} · H{hidden_size}/L{num_layers}"
            f"/LR{learning_rate:g}{resume_hint}…"
        )
        self.training_progress.set(0)
        self.training_time.set("Berechne Restzeit…")
        self.training_metrics.set("")
        threading.Thread(
            target=self._train_in_background,
            args=(examples, epochs, device, config),
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

    def _train_in_background(self, examples, epochs, device, config):
        try:
            model = train_model(
                examples,
                epochs,
                lambda completed, total, loss, remaining: self.messages.put(
                    ("progress", completed, total, loss, remaining)
                ),
                device=device,
                eval_callback=(
                    lambda epoch, total, train_loss, validation_loss: self.messages.put(
                        ("epoch", epoch, total, train_loss, validation_loss)
                    )
                ),
                **config,
            )
            checkpoint_path = model_checkpoint_path(
                config["hidden_size"],
                config["num_layers"],
                config["learning_rate"],
                config["dropout"],
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
        elif message_type == "epoch":
            epoch, epoch_total, train_loss, validation_loss = details
            if validation_loss is None:
                self.training_metrics.set(
                    f"Epoche {epoch}/{epoch_total} · Trainingsfehler {train_loss:.4f}"
                )
            else:
                self.training_metrics.set(
                    f"Epoche {epoch}/{epoch_total} · Training {train_loss:.4f} "
                    f"· Validierung {validation_loss:.4f}"
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
                self.model.eval()
                self.training_progress.set(100)
                self.training_time.set("100% · Training abgeschlossen")
                self.generate_button.configure(state="normal")
                self.status.set(
                    "Training abgeschlossen und gespeichert: "
                    f"{Path(checkpoint_path).name}"
                )
                self._refresh_model_choices(checkpoint_path)
                try:
                    checkpoint = torch.load(
                        checkpoint_path, map_location="cpu", weights_only=True
                    )
                    settings = checkpoint_settings(checkpoint)
                except (OSError, RuntimeError, KeyError, ValueError):
                    settings = None
                if settings is not None:
                    self.hidden_size.set(settings["hidden_size"])
                    self.num_layers.set(settings["num_layers"])
                    self.learning_rate.set(
                        format(float(settings["learning_rate"]), "g")
                    )
                    self.dropout.set(settings["dropout"])
                    self.target_label = settings.get(
                        "target_label", DEFAULT_TARGET_LABEL
                    )
                    self._apply_training_settings(settings["training"])
                    self._apply_generation_settings(settings["generation"])
        elif message_type == "generation_finished":
            (code,) = details
            self.generate_button.configure(state="normal")
            self.output.delete("1.0", "end")
            self.output.insert("1.0", code)
            self.status.set("Generierung abgeschlossen.")
        elif message_type == "generation_failed":
            (error,) = details
            self.generate_button.configure(state="normal")
            self.status.set("Generierung fehlgeschlagen.")
            messagebox.showerror("Generierung fehlgeschlagen", error)

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
                "Trainiere zuerst das Modell mit passenden Trainingsdaten.",
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

        try:
            settings = self._generation_settings()
        except (ValueError, tk.TclError) as error:
            messagebox.showerror(
                "Generierungs-Settings prüfen", str(error), parent=self.root
            )
            return

        self.generate_button.configure(state="disabled")
        self.status.set("Generierung läuft…")
        with self.activation_lock:
            self.latest_activation = None
        threading.Thread(
            target=self._generate_in_background,
            args=(self.model, prompt, settings, self.target_label),
            daemon=True,
        ).start()

    def _generate_in_background(self, model, prompt, settings, target_label):
        try:
            code = generate_code(
                model,
                prompt,
                network_callback=self._publish_network_activation,
                settings=settings,
                target_label=target_label,
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
            messagebox.showinfo(
                "Keine Ausgabe",
                "Es gibt noch keine Ausgabe zum Kopieren.",
            )
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(code)
        self.status.set("Code in die Zwischenablage kopiert.")
