import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn


MODEL_DIR = Path(__file__).with_name("models")
LEGACY_MODEL_PATH = MODEL_DIR / "code_model.pt"
LEGACY_ROOT_MODEL_PATH = Path(__file__).with_name("code_model.pt")
DEMO_DATA_PATH = Path(__file__).with_name("training_data.example.jsonl")
DATA_DIR = Path(__file__).with_name("DATA")
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_NUM_LAYERS = 1
DEFAULT_LEARNING_RATE = 0.003
EOS_TOKEN = 256
VOCAB_SIZE = 257
TRAINING_CHUNK_SIZE = 256
MAX_GENERATED_BYTES = 1600

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


class CodeModel(nn.Module):
    """Ein kleines Byte-Zeichenmodell, das lokal von Grund auf trainiert wird."""

    def __init__(self, hidden_size=DEFAULT_HIDDEN_SIZE, num_layers=DEFAULT_NUM_LAYERS):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(VOCAB_SIZE, hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.output = nn.Linear(hidden_size, VOCAB_SIZE)

    def forward(self, token_ids, hidden=None):
        logits, hidden, _, _ = self.forward_with_activations(token_ids, hidden)
        return logits, hidden

    def forward_with_activations(self, token_ids, hidden=None, include_layers=False):
        embedded = self.embedding(token_ids)
        recurrent, hidden = self.lstm(embedded, hidden)
        logits = self.output(recurrent)
        if include_layers:
            layer_activations = hidden[0][:, 0, :]
            return logits, hidden, embedded[0, -1], recurrent[0, -1], layer_activations
        return logits, hidden, embedded[0, -1], recurrent[0, -1]


def build_prompt_prefix(prompt):
    return f"### Anforderung:\n{prompt.strip()}\n### Python-Code:\n"


def build_training_text(prompt, code):
    return f"{build_prompt_prefix(prompt)}{code.strip()}\n"


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


def load_training_examples(path, max_examples=None):
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
        code_name = next(
            (
                column_names[name]
                for name in CODE_COLUMNS
                if name in column_names
            ),
            None,
        )
        if prompt_name is None or code_name is None:
            raise ValueError(
                "Die Parquet-Datei braucht eine Anforderungs-Spalte "
                "(z. B. prompt oder instruction) und eine Code-Spalte "
                f"(z. B. code oder output). Gefundene Spalten: {table.column_names}"
            )

        input_name = column_names.get("input")
        selected_columns = [
            name for name in (prompt_name, code_name, input_name) if name
        ]
        examples = []
        selected_table = table.select(selected_columns)
        if max_examples is not None:
            selected_table = selected_table.slice(0, max_examples)
        rows = selected_table.to_pylist()
        for row_number, row in enumerate(rows, start=1):
            prompt = row.get(prompt_name)
            code = row.get(code_name)
            extra_input = row.get(input_name) if input_name else None
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"Parquet-Zeile {row_number}: Anforderung ist leer oder kein Text."
                )
            if not isinstance(code, str) or not code.strip():
                raise ValueError(
                    f"Parquet-Zeile {row_number}: Code ist leer oder kein Text."
                )
            prompt = _sanitize_prompt(prompt, code)
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
            examples.append((prompt.strip(), code.strip()))
        if not examples:
            raise ValueError(
                "In der Parquet-Datei wurden keine Zeilen mit nicht-leerer "
                "Anforderung und Code gefunden."
            )
        return examples

    examples = []
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
            code = example.get("code")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"Zeile {line_number} braucht ein nicht-leeres Textfeld 'prompt'."
                )
            if not isinstance(code, str) or not code.strip():
                raise ValueError(
                    f"Zeile {line_number} braucht ein nicht-leeres Textfeld 'code'."
                )
            prompt = _sanitize_prompt(prompt, code)
            if not prompt:
                raise ValueError(
                    f"Zeile {line_number}: Anforderung ist nach dem Entfernen "
                    "der Lösung leer."
                )
            examples.append((prompt, code.strip()))
            if max_examples is not None and len(examples) >= max_examples:
                break
    if not examples:
        raise ValueError("Die Trainingsdatei enthält keine Beispiele.")
    return examples


def model_checkpoint_path(hidden_size, num_layers, learning_rate):
    learning_rate_name = format(learning_rate, "g")
    return MODEL_DIR / (
        f"code_model_h{hidden_size}_l{num_layers}_lr{learning_rate_name}.pt"
    )


def available_models():
    MODEL_DIR.mkdir(exist_ok=True)
    return sorted(MODEL_DIR.glob("code_model*.pt"))


def train_model(
    examples,
    epochs,
    progress_callback,
    device="cpu",
    hidden_size=DEFAULT_HIDDEN_SIZE,
    num_layers=DEFAULT_NUM_LAYERS,
    learning_rate=DEFAULT_LEARNING_RATE,
):
    checkpoint_path = model_checkpoint_path(hidden_size, num_layers, learning_rate)
    model = CodeModel(hidden_size, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        if checkpoint.get("version") != 2:
            raise ValueError(f"Unbekannte Checkpoint-Version: {checkpoint_path.name}")
        if (
            checkpoint.get("hidden_size") != hidden_size
            or checkpoint.get("num_layers") != num_layers
            or checkpoint.get("learning_rate") != learning_rate
        ):
            raise ValueError(
                f"Modell-Settings stimmen nicht mit {checkpoint_path.name} überein."
            )
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    elif (
        hidden_size == DEFAULT_HIDDEN_SIZE
        and num_layers == DEFAULT_NUM_LAYERS
        and learning_rate == DEFAULT_LEARNING_RATE
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

    loss_function = nn.CrossEntropyLoss()
    sequences = []
    for prompt, code in examples:
        text = build_training_text(prompt, code)
        token_ids = list(text.encode("utf-8")) + [EOS_TOKEN]
        sequences.append(torch.tensor(token_ids, dtype=torch.long))

    model.train()
    steps_per_epoch = sum(
        math.ceil((len(sequence) - 1) / TRAINING_CHUNK_SIZE)
        for sequence in sequences
    )
    total_steps = steps_per_epoch * epochs
    completed_steps = 0
    progress_interval = max(1, total_steps // 100)
    training_started = time.perf_counter()
    for epoch in range(epochs):
        for sequence in sequences:
            hidden = None
            for start in range(0, len(sequence) - 1, TRAINING_CHUNK_SIZE):
                chunk = sequence[start : start + TRAINING_CHUNK_SIZE + 1]
                if len(chunk) < 2:
                    continue
                inputs = chunk[:-1].unsqueeze(0).to(device)
                targets = chunk[1:].unsqueeze(0).to(device)
                optimizer.zero_grad()
                logits, hidden = model(inputs, hidden)
                loss = loss_function(
                    logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
                )
                loss.backward()
                optimizer.step()
                hidden = tuple(state.detach() for state in hidden)
                completed_steps += 1
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
    model = model.cpu()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 2,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    return model


def generate_code(model, prompt, activation_callback=None, network_callback=None):
    if not prompt.strip():
        raise ValueError("Gib zuerst eine Anforderung ein.")
    if len(prompt) > 2000:
        raise ValueError("Die Anforderung darf höchstens 2000 Zeichen lang sein.")

    prefix = build_prompt_prefix(prompt)
    token_ids = torch.tensor([list(prefix.encode("utf-8"))], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        previous_layer_activations = None
        logits, hidden, embedding, recurrent, layer_activations = (
            model.forward_with_activations(token_ids, include_layers=True)
        )
        next_token = logits[0, -1].argmax().view(1, 1)
        generated = bytearray()
        for step in range(MAX_GENERATED_BYTES):
            token = int(next_token.item())
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
            previous_layer_activations = hidden[0][:, 0, :]
            logits, hidden, embedding, recurrent, layer_activations = (
                model.forward_with_activations(
                    next_token, hidden, include_layers=True
                )
            )
            next_token = logits[0, -1].argmax().view(1, 1)
    return generated.decode("utf-8", errors="replace").strip()
