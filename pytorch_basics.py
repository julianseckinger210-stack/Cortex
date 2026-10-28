"""
PyTorch-Basics: Übungsskript (separat mit `python pytorch_basics.py` starten)
============================
Installation (einmalig):
    pip install torch matplotlib
(Für GPU-Unterstützung die passende Installationsanleitung auf pytorch.org nutzen.)

Das Skript besteht aus vier Teilen. Lies jeden Teil, führe ihn aus und
verändere dann Werte, um zu sehen, was passiert. Am meisten lernst du,
wenn du Dinge absichtlich kaputt machst (z.B. die Lernrate auf 10 stellen).
"""

import torch
import torch.nn as nn

# Für reproduzierbare Ergebnisse: gleicher Zufalls-Startwert bei jedem Lauf
torch.manual_seed(42)

# Wenn eine NVIDIA-GPU verfügbar ist, nutzen wir sie, sonst die CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Verwendetes Gerät: {device}\n")


# ============================================================
# TEIL 1: TENSOREN
# ============================================================
# Ein Tensor ist ein mehrdimensionales Array (wie bei NumPy),
# kann aber zusätzlich auf der GPU liegen und Gradienten speichern.
print("=== TEIL 1: Tensoren ===")

a = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])           # 2x2-Matrix
b = torch.ones(2, 2)                      # 2x2-Matrix voller Einsen
zufall = torch.rand(2, 3)                 # 2x3, Zufallswerte zwischen 0 und 1

print("Form (shape) von a:", a.shape)     # -> torch.Size([2, 2])
print("a + b =\n", a + b)                 # elementweise Addition
print("a * b =\n", a * b)                 # elementweise Multiplikation
print("a @ b =\n", a @ b)                 # echte Matrixmultiplikation
print("Summe aller Werte in a:", a.sum().item())  # .item() holt eine Zahl heraus

# Die Form ändern: aus (2, 3) wird (3, 2) bzw. (6,)
print("Umgeformt:", zufall.reshape(3, 2).shape, zufall.reshape(-1).shape)

# Einen Tensor auf das gewählte Gerät schieben (CPU oder GPU)
a_geraet = a.to(device)
print("a liegt jetzt auf:", a_geraet.device, "\n")


# ============================================================
# TEIL 2: AUTOGRAD (automatische Ableitung)
# ============================================================
# Das ist das Herzstück von PyTorch: Wenn ein Tensor requires_grad=True hat,
# merkt sich PyTorch jede Rechnung damit. Mit .backward() berechnet es dann
# automatisch, wie stark sich das Ergebnis ändert, wenn man den Tensor ändert.
print("=== TEIL 2: Autograd ===")

w = torch.tensor(3.0, requires_grad=True)   # ein "lernbarer" Wert
loss = (w * 2 - 10) ** 2                    # Fehler: (2w - 10)^2, Minimum bei w = 5
loss.backward()                             # Gradient berechnen
# Mathematisch: d/dw (2w - 10)^2 = 2 * (2w - 10) * 2 = 4 * (2w - 10) = -16 bei w=3
print("Gradient von loss nach w:", w.grad.item(), "(erwartet: -16.0)")

# Ein minimales "Training" von Hand: wir gehen immer ein Stück entgegen dem Gradienten.
# So funktioniert Lernen im Kern: Gewicht anpassen, damit der Fehler kleiner wird.
w = torch.tensor(3.0, requires_grad=True)
lernrate = 0.05                             # wie groß jeder Schritt ist
for schritt in range(20):
    loss = (w * 2 - 10) ** 2
    loss.backward()
    with torch.no_grad():                   # beim Anpassen soll PyTorch nichts mitschreiben
        w -= lernrate * w.grad              # Schritt gegen den Gradienten
    w.grad.zero_()                          # Gradient zurücksetzen, sonst summiert er sich auf!
print(f"Nach 20 Schritten: w = {w.item():.4f} (Ziel: 5.0)\n")


# ============================================================
# TEIL 3: EIN NEURONALES NETZ LERNT y = x²
# ============================================================
# Jetzt benutzen wir PyTorchs fertige Bausteine statt alles von Hand zu machen.
print("=== TEIL 3: Netz lernt y = x^2 ===")

# Trainingsdaten: 200 Punkte zwischen -2 und 2, und ihre Quadrate.
# unsqueeze(1) macht aus Form (200,) die Form (200, 1): 200 Beispiele, je 1 Eingabewert.
x_train = torch.linspace(-2, 2, 200).unsqueeze(1).to(device)
y_train = (x_train ** 2)


# Ein Modell ist eine Klasse, die von nn.Module erbt.
class QuadratNetz(nn.Module):
    def __init__(self):
        super().__init__()
        # Hier werden die Schichten (Layers) definiert.
        # nn.Linear(a, b) ist eine Schicht mit a Eingängen und b Ausgängen;
        # sie enthält lernbare Gewichte und Bias-Werte.
        self.schicht1 = nn.Linear(1, 132)    # 1 Eingang -> 32 Neuronen (erste Hidden Layer)
        self.schicht2 = nn.Linear(132, 132)   # 32 -> 32 (zweite Hidden Layer)
        self.ausgabe = nn.Linear(132, 1)     # 32 -> 1 Ausgang

    def forward(self, x):
        # forward beschreibt, wie die Daten durch das Netz fließen.
        # ReLU ist die "Aktivierungsfunktion": ohne sie wäre das ganze Netz nur
        # eine einzige große lineare Funktion und könnte keine Kurve wie x² lernen.
        x = torch.relu(self.schicht1(x))
        x = torch.relu(self.schicht2(x))
        return self.ausgabe(x)


modell = QuadratNetz().to(device)
print(modell)

# Wie viele lernbare Zahlen hat das Netz? (Das sind die "Parameter")
anzahl_parameter = sum(p.numel() for p in modell.parameters())
print("Anzahl Parameter:", anzahl_parameter)

verlustfunktion = nn.MSELoss()              # misst den mittleren quadratischen Fehler
optimierer = torch.optim.Adam(modell.parameters(), lr=0.01)  # passt die Gewichte an

losses = []                                 # merken wir uns für den Plot
for epoche in range(1500):
    vorhersage = modell(x_train)                    # 1. Vorhersage berechnen
    loss = verlustfunktion(vorhersage, y_train)     # 2. Fehler messen
    optimierer.zero_grad()                          # 3. alte Gradienten löschen
    loss.backward()                                 # 4. neue Gradienten berechnen
    optimierer.step()                               # 5. Gewichte anpassen
    losses.append(loss.item())
    if epoche % 250 == 0:
        print(f"Epoche {epoche:4d} | Fehler: {loss.item():.6f}")

# Test: Was sagt das Netz für Werte voraus?
modell.eval()                               # Auswertungsmodus (wichtig bei manchen Schichttypen)
with torch.no_grad():                       # kein Gradient nötig, spart Speicher
    for testwert in [-1.5, 0.0, 1.0, 1.8]:
        t = torch.tensor([[testwert]]).to(device)
        print(f"x = {testwert:5.1f} -> Netz: {modell(t).item():.3f} | echt: {testwert**2:.3f}")
print()


# ============================================================
# TEIL 4: XOR (ein Klassiker)
# ============================================================
# XOR ist die einfachste Aufgabe, die ein einzelnes Neuron NICHT lösen kann.
# Man braucht mindestens eine Hidden Layer. Perfekt, um zu sehen, warum Tiefe nötig ist.
print("=== TEIL 4: XOR ===")

x_xor = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]).to(device)
y_xor = torch.tensor([[0.], [1.], [1.], [0.]]).to(device)   # XOR: 1 nur wenn Eingänge verschieden

# Zur Abwechslung bauen wir das Modell mit nn.Sequential: Schichten einfach hintereinander.
xor_modell = nn.Sequential(
    nn.Linear(2, 8),    # 2 Eingänge -> 8 Hidden-Neuronen
    nn.Tanh(),          # Aktivierungsfunktion
    nn.Linear(8, 1),    # 8 -> 1 Ausgang (noch ohne Sigmoid, siehe nächste Zeile)
).to(device)

# BCEWithLogitsLoss ist für Ja/Nein-Aufgaben und enthält die Sigmoid-Funktion schon.
xor_loss_fn = nn.BCEWithLogitsLoss()
xor_opt = torch.optim.Adam(xor_modell.parameters(), lr=0.05)

for epoche in range(1000):
    loss = xor_loss_fn(xor_modell(x_xor), y_xor)
    xor_opt.zero_grad()
    loss.backward()
    xor_opt.step()

with torch.no_grad():
    wahrscheinlichkeiten = torch.sigmoid(xor_modell(x_xor))   # Logits -> Wahrscheinlichkeiten
for eingabe, w_keit in zip(x_xor.tolist(), wahrscheinlichkeiten.tolist()):
    print(f"Eingabe {eingabe} -> Wahrscheinlichkeit für '1': {w_keit[0]:.3f}")
print()


# ============================================================
# OPTIONAL: PLOT (nur wenn matplotlib installiert ist)
# ============================================================
try:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Links: Fehlerverlauf beim Training. Er sollte stark fallen.
    ax1.plot(losses)
    ax1.set_yscale("log")
    ax1.set_title("Trainingsfehler (Teil 3)")
    ax1.set_xlabel("Epoche")
    ax1.set_ylabel("Fehler (MSE)")

    # Rechts: echte Kurve gegen die Vorhersage des Netzes
    with torch.no_grad():
        vorhersage = modell(x_train).cpu()
    ax2.plot(x_train.cpu(), y_train.cpu(), label="echt: x²")
    ax2.plot(x_train.cpu(), vorhersage, "--", label="Netz")
    ax2.set_title("Was das Netz gelernt hat")
    ax2.legend()

    plt.tight_layout()
    plt.show()
except ImportError:
    print("matplotlib ist nicht installiert: 'pip install matplotlib' für die Plots.")


# ============================================================
# AUFGABEN ZUM SELBST AUSPROBIEREN
# ============================================================
# 1. Setze in Teil 3 die Lernrate auf 1.0 und dann auf 0.00001. Was passiert?
# 2. Entferne in QuadratNetz.forward die beiden torch.relu(...). Kann das Netz
#    x² dann noch lernen? Warum nicht?
# 3. Ändere die Anzahl der Neuronen (32) und die Anzahl der Hidden Layers.
#    Ab wann wird das Ergebnis schlechter oder besser?
# 4. Lass das Netz statt x² die Funktion sin(x) lernen (Bereich -6 bis 6).
# 5. Bonus: Welche Rolle spielt optimierer.zero_grad()? Kommentiere die Zeile aus
#    und beobachte den Fehler.