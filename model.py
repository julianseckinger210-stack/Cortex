import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn


MODEL_DIR = Path(__file__).with_name("models")
LEGACY_MODEL_PATH = MODEL_DIR / "code_model.pt"
LEGACY_ROOT_MODEL_PATH = Path(__file__).with_name("code_model.pt")
DEMO_DATA_PATH = Path(__file__).with_name("training_data.example.jsonl")
DATA_DIR = Path(__file__).with_name("DATA")
CHECKPOINT_VERSION = 3
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_NUM_LAYERS = 1
DEFAULT_LEARNING_RATE = 0.003
DEFAULT_DROPOUT = 0.0
DEFAULT_BATCH_SIZE = 8
DEFAULT_SHUFFLE = True
DEFAULT_GRADIENT_CLIP = 1.0
DEFAULT_VALIDATION_SPLIT = 0.1
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 40
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_MAX_NEW_BYTES = 600
MAX_NEW_BYTES_LIMIT = 4000
DEFAULT_GENERATION_MODE = "sampling"
EOS_TOKEN = 256
VOCAB_SIZE = 257
TRAINING_CHUNK_SIZE = 256
REPETITION_WINDOW = 64
UNPENALIZED_BYTES = frozenset((9, 10, 13, 32))

INSTRUCTION_COLUMNS = ("instruction", "question", "request", "task")
FALLBACK_PROMPT_COLUMNS = ("prompt", "text")
CODE_COLUMNS = (
    "code",
    "output",
    "completion",
    "response",
    "answer",
    "solution",
)
DEFAULT_TARGET_LABEL = "Python-Code"
GENERAL_TARGET_LABEL = "Antwort"
TARGET_FIELDS = (
    ("code", DEFAULT_TARGET_LABEL),
    ("output", DEFAULT_TARGET_LABEL),
    ("solution", DEFAULT_TARGET_LABEL),
    ("response", GENERAL_TARGET_LABEL),
    ("answer", GENERAL_TARGET_LABEL),
    ("completion", GENERAL_TARGET_LABEL),
)
INSTRUCTION_START_MARKERS = ("### instruction:", "### anforderung:")
INPUT_START_MARKERS = ("### input:", "### eingabe:")
ANSWER_MARKERS = (
    "### output:",
    "### response:",
    "### antwort:",
    "### python-code:",
    "### code:",
)

DEMO_EXAMPLES = [
    {
        "prompt": "Schreibe eine Funktion, die zwei Zahlen addiert.",
        "code": "def addiere(a, b):\n    return a + b",
    },
    {
        "prompt": "Schreibe eine Funktion, die prüft, ob eine Zahl gerade ist.",
        "code": "def ist_gerade(zahl):\n    return zahl % 2 == 0",
    },
    {
        "prompt": "Schreibe eine Funktion, die eine Liste von Zahlen summiert.",
        "code": "def summiere_zahlen(zahlen):\n    return sum(zahlen)",
    },
    {
        "prompt": "Schreibe eine Funktion, die den größten Wert einer Liste zurückgibt.",
        "code": "def maximum(werte):\n    if not werte:\n        return None\n    return max(werte)",
    },
]


def _normalize_dropout(dropout):
    return round(min(0.5, max(0.0, float(dropout))), 3)


class CodeModel(nn.Module):
    """Ein kleines Byte-Zeichenmodell, das lokal von Grund auf trainiert wird."""

    def __init__(
        self,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        num_layers=DEFAULT_NUM_LAYERS,
        dropout=DEFAULT_DROPOUT,
    ):
        super().__init__()
        dropout = _normalize_dropout(dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.embedding = nn.Embedding(VOCAB_SIZE, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, VOCAB_SIZE)

    def forward(self, token_ids, hidden=None):
        logits, hidden, _, _ = self.forward_with_activations(token_ids, hidden)
        return logits, hidden

    def forward_with_activations(self, token_ids, hidden=None, include_layers=False):
        embedded = self.embedding(token_ids)
        recurrent, hidden = self.lstm(self.embedding_dropout(embedded), hidden)
        logits = self.output(recurrent)
        if include_layers:
            layer_activations = hidden[0][:, 0, :]
            return logits, hidden, embedded[0, -1], recurrent[0, -1], layer_activations
        return logits, hidden, embedded[0, -1], recurrent[0, -1]


def _normalize_target_label(target_label):
    label = str(target_label or DEFAULT_TARGET_LABEL).strip()
    return label or DEFAULT_TARGET_LABEL


def build_prompt_prefix(prompt, target_label=DEFAULT_TARGET_LABEL):
    label = _normalize_target_label(target_label)
    return f"### Anforderung:\n{prompt.strip()}\n### {label}:\n"


def build_training_text(prompt, code, target_label=DEFAULT_TARGET_LABEL):
    return f"{build_prompt_prefix(prompt, target_label)}{code.strip()}\n"


def _strip_leaked_answer(prompt, code):
    """Entfernt eine im Prompt enthaltene Lösung (Data-Leakage)."""
    text = prompt.strip()
    code_text = code.strip()
    if len(code_text) < 20 or code_text not in text:
        return text
    before_answer = text.split(code_text, 1)[0].strip()
    return before_answer or text


def _prompt_from_template(prompt):
    """Zieht Anforderung und Eingabe aus einem Trainings-Template heraus."""
    text = prompt.strip()
    lowered = text.casefold()

    instruction_start = 0
    for marker in INSTRUCTION_START_MARKERS:
        index = lowered.find(marker)
        if index != -1:
            instruction_start = index + len(marker)
            break

    answer_positions = [
        lowered.find(marker, instruction_start) for marker in ANSWER_MARKERS
    ]
    answer_positions = [index for index in answer_positions if index != -1]
    instruction_end = min(answer_positions) if answer_positions else len(text)
    instruction = text[instruction_start:instruction_end].strip()

    input_text = ""
    for marker in INPUT_START_MARKERS:
        index = lowered.find(marker, instruction_start)
        if index == -1:
            continue
        input_start = index + len(marker)
        input_positions = [
            lowered.find(answer_marker, input_start)
            for answer_marker in ANSWER_MARKERS
        ]
        input_positions = [pos for pos in input_positions if pos != -1]
        input_end = min(input_positions) if input_positions else len(text)
        candidate = text[input_start:input_end].strip()
        if candidate and candidate not in instruction:
            input_text = candidate
        break

    if input_text:
        return f"{instruction}\n\nEingabe:\n{input_text}"
    return instruction


def _sanitize_prompt(prompt, code):
    text = prompt.strip()
    if not text:
        return text
    template_markers = (
        INSTRUCTION_START_MARKERS + ANSWER_MARKERS + INPUT_START_MARKERS
    )
    lowered = text.casefold()
    if any(marker in lowered for marker in template_markers):
        text = _prompt_from_template(text)
    if not text:
        text = prompt.strip()
    return _strip_leaked_answer(text, code)


def normalize_training_settings(settings=None):
    values = {
        "batch_size": DEFAULT_BATCH_SIZE,
        "shuffle": DEFAULT_SHUFFLE,
        "gradient_clip": DEFAULT_GRADIENT_CLIP,
        "validation_split": DEFAULT_VALIDATION_SPLIT,
        "seed": None,
    }
    if isinstance(settings, dict):
        for key in values:
            if key in settings and settings[key] is not None:
                values[key] = settings[key]
    values["batch_size"] = max(1, min(64, int(values["batch_size"])))
    values["shuffle"] = bool(values["shuffle"])
    values["gradient_clip"] = min(10.0, max(0.0, float(values["gradient_clip"])))
    values["validation_split"] = min(
        0.3, max(0.0, float(values["validation_split"]))
    )
    seed = values["seed"]
    values["seed"] = None if seed in (None, "") else int(seed)
    return values


def normalize_generation_settings(settings=None):
    values = {
        "mode": DEFAULT_GENERATION_MODE,
        "temperature": DEFAULT_TEMPERATURE,
        "top_k": DEFAULT_TOP_K,
        "top_p": DEFAULT_TOP_P,
        "repetition_penalty": DEFAULT_REPETITION_PENALTY,
        "max_new_bytes": DEFAULT_MAX_NEW_BYTES,
        "stop_sequences": [],
        "seed": None,
    }
    if isinstance(settings, dict):
        for key in values:
            if key in settings and settings[key] is not None:
                values[key] = settings[key]

    mode = str(values["mode"]).casefold()
    values["mode"] = "greedy" if mode in ("greedy", "gierig") else "sampling"
    values["temperature"] = min(2.0, max(0.05, float(values["temperature"])))
    values["top_k"] = max(0, min(VOCAB_SIZE, int(values["top_k"])))
    values["top_p"] = min(1.0, max(0.05, float(values["top_p"])))
    values["repetition_penalty"] = min(
        2.0, max(1.0, float(values["repetition_penalty"]))
    )
    values["max_new_bytes"] = max(
        16, min(MAX_NEW_BYTES_LIMIT, int(values["max_new_bytes"]))
    )
    stop_sequences = values["stop_sequences"]
    if isinstance(stop_sequences, str):
        stop_sequences = [stop_sequences]
    values["stop_sequences"] = [
        str(stop) for stop in (stop_sequences or []) if str(stop)
    ][:8]
    seed = values["seed"]
    values["seed"] = None if seed in (None, "") else int(seed)
    return values


def checkpoint_settings(checkpoint):
    return {
        "hidden_size": checkpoint.get("hidden_size"),
        "num_layers": checkpoint.get("num_layers"),
        "learning_rate": checkpoint.get("learning_rate"),
        "dropout": _normalize_dropout(checkpoint.get("dropout") or 0.0),
        "target_label": _normalize_target_label(checkpoint.get("target_label")),
        "training": normalize_training_settings(checkpoint.get("training")),
        "generation": normalize_generation_settings(checkpoint.get("generation")),
    }


def load_training_dataset(path, max_examples=None):
    if path.suffix.casefold() == ".parquet":
        try:
            import pyarrow as arrow
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise ValueError(
                "PyArrow konnte nicht importiert werden. "
                f"Importfehler: {error}\n"
                "Installiere oder aktualisiere die Abhängigkeiten mit: "
                "python -m pip install -r requirements.txt"
            ) from error

        try:
            table = parquet.read_table(path)
        except (OSError, arrow.ArrowException) as error:
            raise ValueError(
                f"Parquet-Datei konnte nicht gelesen werden: {error}"
            ) from error

        column_names = {name.casefold(): name for name in table.column_names}
        prompt_name = next(
            (
                column_names[name]
                for name in INSTRUCTION_COLUMNS
                if name in column_names
            ),
            None,
        )
        if prompt_name is None:
            prompt_name = next(
                (
                    column_names[name]
                    for name in FALLBACK_PROMPT_COLUMNS
                    if name in column_names
                ),
                None,
            )
        target_name = None
        target_label = DEFAULT_TARGET_LABEL
        for field, label in TARGET_FIELDS:
            if field in column_names:
                target_name = column_names[field]
                target_label = label
                break
        if prompt_name is None or target_name is None:
            raise ValueError(
                "Die Parquet-Datei braucht eine Anforderungs-Spalte "
                "(z. B. prompt oder instruction) und eine Ziel-Spalte "
                "(z. B. code, output, response oder answer). "
                f"Gefundene Spalten: {table.column_names}"
            )

        input_name = column_names.get("input")
        selected_columns = [
            name for name in (prompt_name, target_name, input_name) if name
        ]
        examples = []
        selected_table = table.select(selected_columns)
        if max_examples is not None:
            selected_table = selected_table.slice(0, max_examples)
        rows = selected_table.to_pylist()
        for row_number, row in enumerate(rows, start=1):
            prompt = row.get(prompt_name)
            target_text = row.get(target_name)
            extra_input = row.get(input_name) if input_name else None
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"Parquet-Zeile {row_number}: Anforderung ist leer oder kein Text."
                )
            if not isinstance(target_text, str) or not target_text.strip():
                raise ValueError(
                    f"Parquet-Zeile {row_number}: Zieltext ist leer oder kein Text."
                )
            prompt = _sanitize_prompt(prompt, target_text)
            if (
                input_name
                and input_name != prompt_name
                and isinstance(extra_input, str)
                and extra_input.strip()
                and extra_input.strip() not in prompt
            ):
                prompt = f"{prompt}\n\nEingabe:\n{extra_input.strip()}"
            if not prompt.strip():
                raise ValueError(
                    f"Parquet-Zeile {row_number}: Anforderung ist nach dem "
                    "Entfernen der Lösung leer."
                )
            examples.append((prompt.strip(), target_text.strip()))
        if not examples:
            raise ValueError(
                "In der Parquet-Datei wurden keine Zeilen mit nicht-leerer "
                "Anforderung und nicht-leerem Zieltext gefunden."
            )
        return examples, target_label

    examples = []
    target_label = None
    with path.open("r", encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Zeile {line_number} ist kein gültiges JSON: {error.msg}"
                ) from error
            if not isinstance(example, dict):
                raise ValueError(f"Zeile {line_number} muss ein JSON-Objekt sein.")
            prompt = example.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                prompt = example.get("instruction")
            target_text = None
            example_label = None
            for field, label in TARGET_FIELDS:
                candidate = example.get(field)
                if isinstance(candidate, str) and candidate.strip():
                    target_text = candidate
                    example_label = label
                    break
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"Zeile {line_number} braucht ein nicht-leeres Textfeld "
                    "'prompt' oder 'instruction'."
                )
            if target_text is None:
                raise ValueError(
                    f"Zeile {line_number} braucht ein nicht-leeres Zielfeld "
                    "(code, output, response, answer oder completion)."
                )
            if target_label is None:
                target_label = example_label
            elif example_label != target_label:
                raise ValueError(
                    f"Zeile {line_number}: Gemischte Zielformate "
                    f"('{target_label}' und '{example_label}') sind nicht erlaubt."
                )
            prompt = _sanitize_prompt(prompt, target_text)
            if not prompt:
                raise ValueError(
                    f"Zeile {line_number}: Anforderung ist nach dem Entfernen "
                    "der Lösung leer."
                )
            examples.append((prompt, target_text.strip()))
            if max_examples is not None and len(examples) >= max_examples:
                break
    if not examples:
        raise ValueError("Die Trainingsdatei enthält keine Beispiele.")
    return examples, target_label or DEFAULT_TARGET_LABEL


def load_training_examples(path, max_examples=None):
    examples, _ = load_training_dataset(path, max_examples)
    return examples


def model_checkpoint_path(hidden_size, num_layers, learning_rate, dropout=0.0):
    learning_rate_name = format(learning_rate, "g")
    dropout = _normalize_dropout(dropout)
    dropout_suffix = f"_d{format(dropout, 'g')}" if dropout else ""
    return MODEL_DIR / (
        f"code_model_h{hidden_size}_l{num_layers}"
        f"_lr{learning_rate_name}{dropout_suffix}.pt"
    )


def available_models():
    MODEL_DIR.mkdir(exist_ok=True)
    return sorted(MODEL_DIR.glob("code_model*.pt"))


def _split_examples(examples, validation_split, seed):
    if validation_split <= 0 or len(examples) < 10:
        return list(examples), []
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_split))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def _encode_examples(examples, target_label=DEFAULT_TARGET_LABEL):
    sequences = []
    for prompt, code in examples:
        token_ids = list(
            build_training_text(prompt, code, target_label).encode("utf-8")
        )
        sequences.append(torch.tensor(token_ids + [EOS_TOKEN], dtype=torch.long))
    return sequences


def _apply_gradients(model, optimizer, gradient_clip, scale=1.0):
    if scale != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    if gradient_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _evaluate_loss(model, sequences, loss_function, device):
    if not sequences:
        return None
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for sequence in sequences:
            hidden = None
            for start in range(0, len(sequence) - 1, TRAINING_CHUNK_SIZE):
                chunk = sequence[start : start + TRAINING_CHUNK_SIZE + 1]
                if len(chunk) < 2:
                    continue
                inputs = chunk[:-1].unsqueeze(0).to(device)
                targets = chunk[1:].unsqueeze(0).to(device)
                logits, hidden = model(inputs, hidden)
                loss = loss_function(
                    logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
                )
                total_loss += loss.item()
                steps += 1
    model.train()
    return total_loss / steps if steps else None


def _save_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    hidden_size,
    num_layers,
    learning_rate,
    dropout,
    target_label,
    training,
    generation,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "target_label": target_label,
            "training": training,
            "generation": generation,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)


def train_model(
    examples,
    epochs,
    progress_callback,
    device="cpu",
    hidden_size=DEFAULT_HIDDEN_SIZE,
    num_layers=DEFAULT_NUM_LAYERS,
    learning_rate=DEFAULT_LEARNING_RATE,
    dropout=DEFAULT_DROPOUT,
    batch_size=DEFAULT_BATCH_SIZE,
    shuffle=DEFAULT_SHUFFLE,
    gradient_clip=DEFAULT_GRADIENT_CLIP,
    validation_split=DEFAULT_VALIDATION_SPLIT,
    seed=None,
    generation_settings=None,
    eval_callback=None,
    target_label=DEFAULT_TARGET_LABEL,
):
    target_label = _normalize_target_label(target_label)
    training = normalize_training_settings(
        {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "gradient_clip": gradient_clip,
            "validation_split": validation_split,
            "seed": seed,
        }
    )
    batch_size = training["batch_size"]
    shuffle = training["shuffle"]
    gradient_clip = training["gradient_clip"]
    validation_split = training["validation_split"]
    seed = training["seed"]
    dropout = _normalize_dropout(dropout)
    epochs = max(1, int(epochs))
    generation = normalize_generation_settings(generation_settings)

    if seed is not None:
        torch.manual_seed(seed)
    randomizer = random.Random(seed)

    checkpoint_path = model_checkpoint_path(
        hidden_size, num_layers, learning_rate, dropout
    )
    model = CodeModel(hidden_size, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        if checkpoint.get("version") not in (2, CHECKPOINT_VERSION):
            raise ValueError(f"Unbekannte Checkpoint-Version: {checkpoint_path.name}")
        if (
            checkpoint.get("hidden_size") != hidden_size
            or checkpoint.get("num_layers") != num_layers
            or checkpoint.get("learning_rate") != learning_rate
            or _normalize_dropout(checkpoint.get("dropout") or 0.0) != dropout
        ):
            raise ValueError(
                f"Modell-Settings stimmen nicht mit {checkpoint_path.name} überein."
            )
        stored_target_label = _normalize_target_label(
            checkpoint.get("target_label")
        )
        if stored_target_label != target_label:
            raise ValueError(
                f"{checkpoint_path.name} wurde für das Format "
                f"'{stored_target_label}' trainiert, nicht für "
                f"'{target_label}'. Wähle andere Modell-Settings oder "
                "lösche den Checkpoint."
            )
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    elif (
        target_label == DEFAULT_TARGET_LABEL
        and hidden_size == DEFAULT_HIDDEN_SIZE
        and num_layers == DEFAULT_NUM_LAYERS
        and learning_rate == DEFAULT_LEARNING_RATE
        and dropout == 0.0
    ):
        legacy_path = (
            LEGACY_MODEL_PATH
            if LEGACY_MODEL_PATH.exists()
            else LEGACY_ROOT_MODEL_PATH
        )
        if legacy_path.exists():
            checkpoint = torch.load(
                legacy_path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(checkpoint["state_dict"])

    train_examples, validation_examples = _split_examples(
        examples, validation_split, seed
    )
    sequences = _encode_examples(train_examples, target_label)
    validation_sequences = _encode_examples(validation_examples, target_label)
    loss_function = nn.CrossEntropyLoss()

    chunk_counts = [
        max(0, math.ceil((len(sequence) - 1) / TRAINING_CHUNK_SIZE))
        for sequence in sequences
    ]
    steps_per_epoch = sum(chunk_counts)
    total_steps = max(1, steps_per_epoch * epochs)
    completed_steps = 0
    progress_interval = max(1, total_steps // 100)
    training_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    pending_examples = 0

    for epoch in range(epochs):
        order = list(range(len(sequences)))
        if shuffle:
            randomizer.shuffle(order)
        epoch_loss = 0.0
        epoch_steps = 0
        for index in order:
            sequence = sequences[index]
            chunk_count = max(1, chunk_counts[index])
            hidden = None
            for start in range(0, len(sequence) - 1, TRAINING_CHUNK_SIZE):
                chunk = sequence[start : start + TRAINING_CHUNK_SIZE + 1]
                if len(chunk) < 2:
                    continue
                inputs = chunk[:-1].unsqueeze(0).to(device)
                targets = chunk[1:].unsqueeze(0).to(device)
                logits, hidden = model(inputs, hidden)
                loss = loss_function(
                    logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
                )
                (loss / (batch_size * chunk_count)).backward()
                hidden = tuple(state.detach() for state in hidden)
                completed_steps += 1
                epoch_loss += loss.item()
                epoch_steps += 1
                if (
                    completed_steps % progress_interval == 0
                    or completed_steps == total_steps
                ):
                    elapsed = time.perf_counter() - training_started
                    remaining = elapsed / completed_steps * (
                        total_steps - completed_steps
                    )
                    progress_callback(
                        completed_steps,
                        total_steps,
                        loss.item(),
                        remaining,
                    )
            pending_examples += 1
            if pending_examples >= batch_size:
                _apply_gradients(model, optimizer, gradient_clip)
                pending_examples = 0
        if pending_examples:
            _apply_gradients(
                model,
                optimizer,
                gradient_clip,
                scale=batch_size / pending_examples,
            )
            pending_examples = 0

        average_train_loss = epoch_loss / epoch_steps if epoch_steps else 0.0
        validation_loss = _evaluate_loss(
            model, validation_sequences, loss_function, device
        )
        if eval_callback is not None:
            eval_callback(epoch + 1, epochs, average_train_loss, validation_loss)
        _save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            hidden_size,
            num_layers,
            learning_rate,
            dropout,
            target_label,
            training,
            generation,
        )

    model = model.cpu()
    _save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        hidden_size,
        num_layers,
        learning_rate,
        dropout,
        target_label,
        training,
        generation,
    )
    return model


def _choose_next_token(logits, settings, generated, generator):
    if settings["mode"] == "greedy":
        return int(torch.argmax(logits))

    adjusted = logits.detach().to("cpu", dtype=torch.float32).clone()
    penalty = settings["repetition_penalty"]
    if penalty > 1.0 and generated:
        for token in set(generated[-REPETITION_WINDOW:]):
            if token in UNPENALIZED_BYTES:
                continue
            if adjusted[token] > 0:
                adjusted[token] /= penalty
            else:
                adjusted[token] *= penalty

    adjusted /= settings["temperature"]

    top_k = settings["top_k"]
    if top_k > 0:
        keep = min(top_k, adjusted.numel())
        threshold = torch.topk(adjusted, keep).values[-1]
        adjusted[adjusted < threshold] = float("-inf")

    top_p = settings["top_p"]
    if 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(adjusted, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        removed = cumulative - probabilities > top_p
        sorted_logits[removed] = float("-inf")
        adjusted = torch.full_like(adjusted, float("-inf")).scatter(
            0, sorted_indices, sorted_logits
        )

    probabilities = torch.softmax(adjusted, dim=-1)
    if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
        return int(torch.argmax(logits))
    return int(torch.multinomial(probabilities, 1, generator=generator))


def _matching_stop_sequence(generated, stop_sequences):
    for stop in stop_sequences:
        stop_bytes = stop.encode("utf-8")
        if stop_bytes and len(generated) >= len(stop_bytes):
            if bytes(generated[-len(stop_bytes) :]) == stop_bytes:
                return stop_bytes
    return None


def generate_code(
    model,
    prompt,
    activation_callback=None,
    network_callback=None,
    settings=None,
    target_label=DEFAULT_TARGET_LABEL,
):
    if not prompt.strip():
        raise ValueError("Gib zuerst eine Anforderung ein.")
    if len(prompt) > 2000:
        raise ValueError("Die Anforderung darf höchstens 2000 Zeichen lang sein.")

    options = normalize_generation_settings(settings)
    generator = None
    if options["seed"] is not None:
        generator = torch.Generator()
        generator.manual_seed(options["seed"])

    prefix = build_prompt_prefix(prompt, target_label)
    device = next(model.parameters()).device
    token_ids = torch.tensor(
        [list(prefix.encode("utf-8"))], dtype=torch.long, device=device
    )
    model.eval()
    with torch.no_grad():
        previous_layer_activations = None
        logits, hidden, embedding, recurrent, layer_activations = (
            model.forward_with_activations(token_ids, include_layers=True)
        )
        generated = bytearray()
        for step in range(options["max_new_bytes"]):
            token = _choose_next_token(
                logits[0, -1], options, generated, generator
            )
            if activation_callback is not None:
                activation_callback(
                    embedding.detach().cpu().tolist(),
                    recurrent.detach().cpu().tolist(),
                    logits[0, -1].detach().cpu().tolist(),
                    token,
                    step + 1,
                )
            if network_callback is not None:
                network_callback(
                    embedding.detach().cpu().tolist(),
                    layer_activations.detach().cpu().tolist(),
                    logits[0, -1].detach().cpu().tolist(),
                    token,
                    step + 1,
                    (
                        previous_layer_activations.detach().cpu().tolist()
                        if previous_layer_activations is not None
                        else None
                    ),
                )
            if token == EOS_TOKEN:
                break
            generated.append(token)
            stop_bytes = _matching_stop_sequence(
                generated, options["stop_sequences"]
            )
            if stop_bytes is not None:
                del generated[-len(stop_bytes) :]
                break
            previous_layer_activations = hidden[0][:, 0, :]
            next_token = torch.tensor([[token]], dtype=torch.long, device=device)
            logits, hidden, embedding, recurrent, layer_activations = (
                model.forward_with_activations(
                    next_token, hidden, include_layers=True
                )
            )
    return generated.decode("utf-8", errors="replace").strip()
