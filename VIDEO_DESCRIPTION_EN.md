# LeLamp-FlyBrain Phase 1 — Simulation Demo Video Description (English)

> Purpose: GitHub repository README / demo video description / subtitle script.
> Recording command (run in the project root, PowerShell):
>
> ```powershell
> .\.venv312\python.exe main.py --mode demo --controller brain --ckpt data\w_dn_calibrated.npz --engine mujoco
> ```

---

## One-Line Summary (for the top of the video description)

**EN**: A 5-DOF desk lamp driven entirely by a fixed-topology, Drosophila-inspired spiking neural network — six emotional behaviors emerge from a 10-neuron descending readout, modulated online by a γ=[DA, OA, 5-HT] neuromodulatory vector. No backpropagation, no learned connectivity, no dataset.

---

## Video Timeline Narration (~38 seconds, segment by segment)

| Time | State | What You Should See | Neuromodulation Interpretation | Console Caption |
|---|---|---|---|---|
| 00:00–00:06 | **Vigilance_Defense** | Lamp stands upright; base slowly scans at 0.15 Hz (±0.8 rad); ice-blue head LED, steady | High 5-HT → larger membrane leak time constant; the network becomes slow and deliberate, like a vigilant fly sweeping its gaze | "Scanning surroundings..." |
| 00:06–00:12 | **Approach_Reward** | Lamp leans forward; cheerful 0.8 Hz head nodding (±0.2 rad); warm-yellow LED, steady | High DA → three-factor learning gate fully open; the network learns fastest in this state | "Glad to be with you." |
| 00:12–00:18 | **Fear_Arousal** | Lamp recoils and rises; 8 Hz full-body tremor; red LED strobing at 10 Hz | High OA → sensory injection gain amplified to 1.85× (maximally error-sensitive); low 5-HT → fast reactions | "Watch out! Obstacle ahead." |
| 00:18–00:24 | **Exploration_Seeking** | Head-down cruising; 0.4 Hz S-curve scanning; green LED alternating breath | All three modulation knobs centered — an unbiased, neutral exploratory state | "Exploring your workspace." |
| 00:24–00:32 | **Emotional_Comfort** | Forward-and-downward lean; 0.3 Hz breathing rhythm; warm-pink slow breath (3 s period) | Low OA → sensory input actively suppressed; insensitive to external disturbance, conveying "calm" | "I am here for you." |
| 00:32–00:38 | **Grooming_Rest** | Tucked and bowed, fully still; dim blue LED at 25% brightness | DA/OA both floored → learning frozen + sensory suppressed; the system sleeps | (no caption) |

---

## Technical Points Not Visible On-Screen (recommended for the description)

1. **Fixed connection topology**: a 500-node mock connectome (COO sparse) whose edge set **never changes** during the full 38 seconds — behavioral differences arise entirely from how the *same* topology is run (γ modulation). This is one of the three pillars of FlyDoom.
2. **Single readout pathway**: pose = `Δq = W · s_DN`, where W is a 5×10 matrix reading exclusively from 10 descending neurons (DNs). W is calibrated by ridge regression (not backpropagation); deterministic tracking error across the six states is **0.04 rad (≈2.3°)** — a 92.6% reduction versus the 0.542 rad stationary baseline.
3. **Closed-loop servoing**: every control step (50 Hz), joint errors are injected into the SNN (100 Hz) through ON/OFF-rectified sensory channels. The network servos the **motion reference** in closed loop — scanning, nodding, and breathing are not scripted playback but the result of real-time SNN servoing.
4. **Physical fidelity**: MuJoCo engine, kp=17.8 sts3215 servo model, 240 Hz physics stepping; per-step joint increment clamped to 0.06 rad (safety layer).
5. **Event stream source**: this demo uses a local Mock script (`utils/mock_scenarios.json`). The same pipeline supports live DeepSeek generation (`--mode online`), with automatic fallback Qwen → Mock on API failure, without interrupting the simulation.

---

## Honesty Statement (recommended to keep — reflects research rigor)

- The three γ pathways (5-HT→time constant, OA→sensory gain, DA→learning gate) are **engineering analogies**, not claims of biological equivalence to Drosophila neuromodulators.
- The **reference targets** of the emotional poses (pose primitives) are hand-designed behavioral priors; this phase validates whether "fixed topology + DN readout" can servo-reproduce those behaviors — not the emergence of the behaviors themselves.
- Pure three-factor biological learning (eligibility trace × dopamine) is not yet stable on the T1 task (the negative result is documented as-is); the demo uses W from the calibration upper bound. Improving the learning rule is a follow-up research item.

---

## Recording Suggestions

1. **Layout**: record the MuJoCo window (main view) and the PowerShell console (captions) side by side; captions appear only in the console.
2. **Resolution**: 1280×720 or higher; the head-LED color changes (especially the red strobe and warm-pink breathing) are visual highlights — do not crop the lamp head.
3. **Single take**: play the full 38 seconds, then hold the final frame for 3–5 seconds to show the "window kept open" prompt.
4. **Optional second segment**: append a recording of `--mode online --scenario "late night, tired user working overtime, a cup on the desk"` to show the LLM generating a live event stream (~10 s generation + playback), contrasting with the Mock segment.
5. **Publish format**: MP4 (H.264), ≤60 s; keep this file in the repository as the text version of the description.
