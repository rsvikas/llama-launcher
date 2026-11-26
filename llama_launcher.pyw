"""
Llama Launcher - start / stop / switch llama.cpp models with one click.
Double-click to run (Python 3 + tkinter). Edit PRESETS to add or change models.

The window is drawn in a Material Design look (flat cards, rounded buttons, one
accent colour).  No extra package is needed for that - it is all tkinter.

At startup the launcher asks whether to start normally or as administrator.
Only an administrator can flush the Windows file cache, so that one control is
shown in administrator mode and marked as unavailable otherwise.  Everything
else works the same in both modes, and the badge in the header always says
which mode the launcher is in.

The three action buttons only do what makes sense right now:
  START   - needs a preset and a stopped launcher
  STOP    - only alive while a model runs, while auto-NGL is tuning, or while
            a model is being started
  RESTART - only alive while a model runs; it restarts it with the settings
            that are on screen now (this is what applies a changed Sampling)

Every sampler below was cross-checked against the publisher's own documentation
(the SOURCES table right above PRESETS lists the links).  Where a publisher
publishes no fixed table, the preset says so instead of pretending one exists.
"""
import csv
import glob
import hashlib
import json
import urllib.error
import urllib.request
import ctypes
import math
import os
import queue
import re
import struct
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
import tkinter.font as tkfont
from ctypes import wintypes
from tkinter import ttk, messagebox

BASE_DIR = r"C:\prism-llama"
DEFAULT_PORT = "8080"
WIN_W, WIN_H = 1000, 840          # the launcher window size
NO_WINDOW = 0x08000000
# Bump this whenever the launcher changes, so the window title and the first log
# line always say which copy is running and where it looks for the models.
APP_VERSION = "2026-11-26.08"

# The combobox starts on this entry instead of on a preset: nothing is chosen
# until the user picks one, so START never starts a model by accident.
NO_PRESET = "— Select a preset to start —"

# A .pyw has no console, so an error that is not shown on screen is an error
# the user never sees.  Every failure is written here as well.
ERROR_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "launcher_error.log")


def write_error_log(detail):
    """Write a traceback to the error log next to the launcher."""
    try:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n{detail}\n")
    except Exception:
        pass
    return ERROR_LOG_FILE

# llama-server verbosity: 3 = info (default), 4 = trace, 5 = debug.
# Level 4 adds the server's "cache state: N prompts, X MiB" line, which is what
# the "Prompt cache" row of the memory tab reads.  Level 5 (debug) is far too
# noisy, so we stop at 4.  Set to [] to go back to the server default.
EXTRA_SERVER_ARGS = []   # not a publisher setting; ["--verbosity", "4"] adds the prompt-cache log line but is noisy

# llama.cpp reasoning switch.  "auto" is NOT neutral: server-context.cpp does
#   enable_thinking = (enable_reasoning != 0) && template_supports_thinking
# so auto turns thinking ON for every template that supports enable_thinking,
# even when the template itself defaults it to false (Gemma 4 E4B does).
# We therefore state the mode explicitly on every preset.
REASONING_ON = ["--reasoning", "on"]
REASONING_OFF = ["--reasoning", "off"]

def ctx(n=16384, b=512, ub=512, ngl=99):
    return ["-c", str(n), "-np", "1", "-ngl", str(ngl), "-b", str(b), "-ub", str(ub)]

KV = ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--cache-ram", "2048"]

# The 27B is the one model here that does not fit in a 16 GB laptop with room to
# spare: weights + chat memory are about 7.5 GB before a single layer is offloaded.
# Its prompt cache is capped at 512 MiB instead of the global 2048 MiB, so that
# 1.5 GB stays available for the weights and for the graphics chip.  The cap only
# costs anything when a new question repeats a long old one; see the README.
KV_27B = ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--cache-ram", "512"]

# --- PrismML "Ternary Bonsai 2" samplers ---------------------------------
# https://docs.prismml.com/models/bonsai-2-27b -> "Best practices / Generation parameters"
#   thinking : temperature 1.0 | top_p 0.95 | top_k 20 | min_p 0.05 | presence 0.0 | repetition 1.0
#   instruct : temperature 0.7 | top_p 0.80 | top_k 20 | min_p 0.0  | presence 1.5 | repetition 1.0
# https://docs.prismml.com/models/bonsai-2-9b -> thinking: min_p 0.0, presence 1.0 (same otherwise)
# Both pair presence_penalty with "--repeat-last-n 64" (llama.cpp's default window).
SAMPLING_27B = ["--jinja", "--temp", "1.0", "--top-p", "0.95", "--top-k", "20",
                "--min-p", "0.05", "--presence-penalty", "0.0",
                "--repeat-penalty", "1.0", "--repeat-last-n", "64"]
SAMPLING_9B = ["--jinja", "--temp", "1.0", "--top-p", "0.95", "--top-k", "20",
               "--min-p", "0.0", "--presence-penalty", "1.0",
               "--repeat-penalty", "1.0", "--repeat-last-n", "64"]
BONSAI_INSTRUCT_SAMPLING = ["--jinja", "--temp", "0.7", "--top-p", "0.80", "--top-k", "20",
                            "--min-p", "0", "--presence-penalty", "1.5",
                            "--repeat-penalty", "1.0", "--repeat-last-n", "64"]

def think(n):
    return ["--reasoning-budget", str(n)]

# --- Sampling modes -------------------------------------------------------
# "Publisher" keeps the values the model publisher documents. The other modes
# are for small models that blurt out confident nonsense. They do one thing:
# stop the model from picking tokens out of the thin tail of the distribution,
# where confident-looking but wrong words live.
#
#   Precise   : low temperature and a tight nucleus. The usual first thing to
#               try. Costs some variety, buys a lot of accuracy.
#   Mirostat  : keeps the perplexity in a fixed band instead of following the
#               model's own confidence. Note: Mirostat makes llama.cpp ignore
#               top_k, top_p and typical_p, so they are left out here.
#   + DRY     : "Don't Repeat Yourself" penalises the model for repeating a
#               phrase. Small models loop; loops read as hallucination.
#
# The flags are REPLACED, not appended, so the chosen mode always wins over the
# publisher's values.
SAMPLER_FLAGS = ("--temp", "--temperature", "--top-k", "--top-p", "--min-p",
                 "--typical", "--typical-p", "--top-n-sigma", "--mirostat",
                 "--mirostat-lr", "--mirostat-ent", "--dry-multiplier",
                 "--dry-base", "--dry-allowed-length", "--dry-penalty-last-n",
                 "--repeat-penalty", "--repeat-last-n", "--presence-penalty",
                 "--frequency-penalty", "--xtc-probability", "--xtc-threshold",
                 "--dynatemp-range", "--dynatemp-exp", "--samplers")

PRECISE_SAMPLING = ["--temp", "0.3", "--top-k", "20", "--top-p", "0.8", "--min-p", "0.05",
                    "--repeat-penalty", "1.05"]
DETERMINISTIC_SAMPLING = ["--temp", "0.0", "--repeat-penalty", "1.05"]
MIROSTAT_SAMPLING = ["--mirostat", "2", "--mirostat-lr", "0.1", "--mirostat-ent", "5.0",
                    "--repeat-penalty", "1.05"]
DRY_SAMPLING = ["--dry-multiplier", "0.8", "--dry-base", "1.75", "--dry-allowed-length", "2",
                "--dry-penalty-last-n", "128"]
# For code, "Precise" is half right: the low temperature helps, but it also lowers the
# repetition penalty to 1.05, while the code publishers ask for 1.1.  A lower penalty on
# code means the model repeats the same line instead of moving on.  So this mode keeps
# the publisher's 1.1 and adds the DRY sampler, which punishes the repeated phrase
# directly.  Use it with the code presets.
#
# dry-allowed-length is 3 here, not the 2 used for prose.  DRY punishes any repeat of
# an n-gram of that length or longer, and in code a two-word fragment repeats by
# nature ("self.", "return ", the name of a variable).  Penalising those pushes the
# model out of writing normal code.  Length 3 catches a repeated phrase instead.
CODE_SAMPLING = ["--temp", "0.3", "--top-k", "20", "--top-p", "0.8", "--min-p", "0.05",
                 "--repeat-penalty", "1.1",
                 "--dry-multiplier", "0.8", "--dry-base", "1.75",
                 "--dry-allowed-length", "3", "--dry-penalty-last-n", "128"]
SAMPLING_MODES = {
    "Normal (publisher settings)": None,
    "Precise (less random)": PRECISE_SAMPLING,
    "Deterministic (same answer)": DETERMINISTIC_SAMPLING,
    "Mirostat 2 (steady)": MIROSTAT_SAMPLING,
    "Precise + DRY (no repeats)": PRECISE_SAMPLING + DRY_SAMPLING,
    "Code (precise, no repeats)": CODE_SAMPLING,
}

def apply_sampling_mode(args, mode):
    """Preset args with the sampling flags replaced by the chosen mode."""
    flags = SAMPLING_MODES.get(mode)
    if not flags:
        return list(args)
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
            continue
        if a in SAMPLER_FLAGS:
            skip = True
            continue
        out.append(a)
    return out + list(flags)


_VERSION_CACHE = {}


def server_version():
    """The llama.cpp version, one short line, e.g. 'b5678 (2026-05-05)'. Read once."""
    exe = os.path.join(BASE_DIR, "llama-server.exe")
    if exe in _VERSION_CACHE:
        return _VERSION_CACHE[exe]
    line = ""
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True,
                             timeout=25, creationflags=NO_WINDOW)
        for raw in ((out.stdout or "") + (out.stderr or "")).splitlines():
            if raw.strip():
                line = raw.strip()
                break
    except Exception:
        line = ""
    _VERSION_CACHE[exe] = line
    return line


def resolve_file(name):
    """Return the real on-disk file name in BASE_DIR, matched case-insensitively.

    Publishers are not consistent about case ("Qwen2.5-Coder-7B-Instruct-Q6_K.gguf"
    vs the published "qwen2.5-coder-7b-instruct-q6_k.gguf").  Windows hides the
    difference, but the launcher should still report the name that is really
    there.  Returns `name` unchanged when nothing matches, so the "missing file"
    message keeps showing what was expected.
    """
    if not name:
        return name
    if os.path.isfile(os.path.join(BASE_DIR, name)):
        return name
    try:
        entries = os.listdir(BASE_DIR)
    except OSError:
        return name
    low = name.lower()
    for e in entries:
        if e.lower() == low and os.path.isfile(os.path.join(BASE_DIR, e)):
            return e
    return name

def preset(name, model, extra, exe="llama-server.exe", mmproj=None, coding=False):
    model = resolve_file(model)
    args = ["-m", os.path.join(".", model)]
    if mmproj:
        mmproj = resolve_file(mmproj)
        args += ["--mmproj", os.path.join(".", mmproj)]
    # "coding" marks the presets where the coding rules and the code sampler matter.
    return {"name": name, "exe": exe, "args": args + extra, "coding": coding}

# Projector preference: Q8_0 first (smallest), then f16, then bf16 (reference
# precision), matching what each publisher recommends.
_MM_RANK = ("q8_0", "q4_0", "f16", "bf16", "f32")

def find_mmproj(patterns):
    """Find a projector file.  `patterns` is a list of globs tried in order.

    Both naming styles are in the wild ("mmproj-<model>-f16.gguf" and
    "<model>-mmproj.gguf"), so every projector is looked up with both.
    """
    hits = []
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(BASE_DIR, pat)))
        if hits:
            break
    if not hits:
        return None
    def rank(path):
        low = os.path.basename(path).lower()
        for i, q in enumerate(_MM_RANK):
            if q in low:
                return i
        return len(_MM_RANK)
    hits.sort(key=lambda p: (rank(p), p))
    return os.path.basename(hits[0])

# --- Google Gemma 4 E4B --------------------------------------------------
# https://ai.google.dev/gemma/docs/core/model_card_4 -> "Sampling Parameters":
#   "Use the following standardized sampling configuration across all use cases:
#    temperature=1.0, top_p=0.95, top_k=64"
# google/gemma-4-E4B-it generation_config.json agrees (temp 1.0, top_k 64, top_p 0.95).
# The remaining flags are llama.cpp defaults, spelled out so the command is explicit.
GEMMA_SAMPLING = ["--jinja", "--temp", "1.0", "--top-p", "0.95", "--top-k", "64",
                  "--min-p", "0", "--presence-penalty", "0.0", "--repeat-penalty", "1.0"]
# --- Qwen3-8B -------------------------------------------------------------
# https://huggingface.co/Qwen/Qwen3-8B -> "Best Practices":
#   thinking mode    : Temperature 0.6, TopP 0.95, TopK 20, MinP 0
#   non-thinking mode: Temperature 0.7, TopP 0.8,  TopK 20, MinP 0
# The card fixes no repetition penalty; 0.0 / 1.0 are the llama.cpp defaults.
QWEN_SAMPLING = ["--jinja", "--temp", "0.6", "--top-p", "0.95", "--top-k", "20",
                 "--min-p", "0", "--presence-penalty", "0.0", "--repeat-penalty", "1.0"]
QWEN_NON_THINKING_SAMPLING = ["--jinja", "--temp", "0.7", "--top-p", "0.80", "--top-k", "20",
                              "--min-p", "0", "--presence-penalty", "0.0", "--repeat-penalty", "1.0"]
# --- Meta Llama 3.1 8B ---------------------------------------------------
# The Llama 3.1 model card has no sampling table, but Meta's shipped generation_config.json
# (meta-llama/Llama-3.1-8B-Instruct) sets: do_sample true, temperature 0.6, top_p 0.9.
# Nothing else is set there, so nothing else is passed except the explicit "disabled" values.
LLAMA_SAMPLING = ["--jinja", "--temp", "0.6", "--top-p", "0.9", "--min-p", "0",
                  "--presence-penalty", "0.0", "--repeat-penalty", "1.0"]
# --- Qwen2.5-VL 7B -------------------------------------------------------
# The model card has no sampling table. The shipped generation_config.json is the only official
# source: temperature ~0 (greedy) + repetition_penalty 1.05.
# NOTE: the "temp 0.7 / top_p 0.8 / top_k 20 / presence 1.5" block on the QwenLM GitHub pages
# belongs to Qwen3-VL, not Qwen2.5-VL.
QWENVL_GENCFG_SAMPLING = ["--jinja", "--temp", "0.0", "--top-p", "1.0", "--top-k", "0", "--min-p", "0",
                          "--presence-penalty", "0.0", "--repeat-penalty", "1.05"]
# --- Qwen2.5-Coder 7B ----------------------------------------------------
# Qwen/Qwen2.5-Coder-7B-Instruct generation_config.json (the card has no table):
#   temperature 0.7 | top_p 0.8 | top_k 20 | repetition_penalty 1.1
QWEN_CODER_SAMPLING = ["--jinja", "--temp", "0.7", "--top-p", "0.8", "--top-k", "20", "--min-p", "0",
                       "--presence-penalty", "0.0", "--repeat-penalty", "1.1"]
# --- SmolVLM2 500M Video -------------------------------------------------
# The model card's examples all use do_sample=False -> greedy decoding, and
# preprocessor_config.json has video_sampling.fps = 1 (hence --video-fps 1).
SMOL_SAMPLING = ["--jinja", "--temp", "0", "--top-p", "1.0", "--top-k", "0", "--min-p", "0",
                 "--presence-penalty", "0.0", "--repeat-penalty", "1.0"]
# --- Qwen3-ASR 1.7B ------------------------------------------------------
# Model card: "During evaluation ... Greedy search was used for all decoding."
# generation_config.json: do_sample false.
ASR_SAMPLING = ["--jinja", "--temp", "0.0", "--top-p", "1.0", "--top-k", "0", "--min-p", "0",
                "--presence-penalty", "0.0", "--repeat-penalty", "1.1", "--repeat-last-n", "64"]

# --- Next six models: publisher-documented values only -------------------
# No top_k / repeat penalty unless the publisher publishes it. "--jinja" is the chat template
# (needed for --reasoning), not a sampler; it is in every list to match the rest of this file.
MIMO_SAMPLING = ["--jinja", "--temp", "0.6", "--top-p", "0.95", "--top-k", "20"]      # Xiaomi generation_config.json
QWEN35_SAMPLING = ["--jinja", "--temp", "0.7", "--top-p", "0.8", "--top-k", "20"]    # Qwen3.5-9B card, NON-thinking mode
NEMOTRON_4B_SAMPLING = ["--jinja", "--temp", "1.0", "--top-p", "0.95"]              # NVIDIA Nemotron 3 Nano 4B, reasoning
DEEPSEEK_R1_SAMPLING = ["--jinja", "--temp", "0.6", "--top-p", "0.95"]              # DeepSeek-R1 README temp; top-p from distill config
MINISTRAL_SAMPLING = ["--jinja", "--temp", "0.15", "--top-p", "1.0"]                # Mistral Instruct guidance (0.15 or 0.1)

# --- OLMoE-1B-7B-0924-Instruct (Q6_K) ------------------------------------
# Config: max_position_embeddings = 4096, hidden_size = 2048, num_hidden_layers = 16,
#         64 experts, 8 active per token (1.3B active of 6.9B total).
# AllenAI publishes NO fixed sampling table: generation_config.json leaves the
# temperature unset and the demo just passes the user's own temperature.
# So this preset uses the standard chat values, and says so instead of pretending
# the publisher fixed them.
OLMOE_SAMPLING = ["--jinja", "--temp", "0.7", "--top-p", "0.9", "--top-k", "40",
                  "--min-p", "0", "--presence-penalty", "0.0", "--repeat-penalty", "1.05"]
# --- Qwen1.5-MoE-A2.7B-Chat (Q3_K_M) --------------------------------------
# Config: max_position_embeddings = 32768, 60 experts, 4 active per token
#         (14.3B total, 2.7B active).
# Qwen's published generation_config.json for the Qwen1.5-Chat family:
#   temperature 0.7 | top_p 0.8 | top_k 20 | repetition_penalty 1.05
# (Qwen's own advice: use these values if you hit code switching or other bad cases.)
QWEN15MOE_SAMPLING = ["--jinja", "--temp", "0.7", "--top-p", "0.8", "--top-k", "20",
                      "--min-p", "0", "--presence-penalty", "0.0", "--repeat-penalty", "1.05"]

VISION = dict(b=512, ub=512)

# Projector files: try "<name>-mmproj*.gguf" and "mmproj-<name>*.gguf".
_ASR_MM = find_mmproj(["mmproj*Qwen3-ASR*.gguf", "*Qwen3-ASR*mmproj*.gguf"])
_SMOL_MM = find_mmproj(["mmproj*SmolVLM2*.gguf", "*SmolVLM2*mmproj*.gguf"])
_BONSAI9_MM = find_mmproj(["*Bonsai-2-9B-mmproj*.gguf", "mmproj*Bonsai-2-9B*.gguf"])
# Gemma's projector is published BOTH ways: ggml-org ships
# "mmproj-gemma-4-E4B-it-Q8_0.gguf", google ships "gemma-4-E4B-it-mmproj.gguf".
_GEMMA_MM = find_mmproj(["mmproj*gemma*.gguf", "*gemma*mmproj*.gguf"])
_QWENVL_MM = find_mmproj(["mmproj*Qwen2.5-VL*.gguf", "*Qwen2.5-VL*mmproj*.gguf"])
_ULTRAVOX_MM = find_mmproj(["mmproj*ultravox*.gguf", "*ultravox*mmproj*.gguf"])

# Stock llama.cpp only - only Q2_0 pack runs here, no PrismML fork
BONSAI_27B_FILE = "Ternary-Bonsai-2-27B-Q2_0.gguf"
BONSAI_9B_FILE = "Ternary-Bonsai-2-9B-Q2_0.gguf"
STARTUP_NOTICES = []

# --- Where every preset value above comes from -----------------------------
# Bonsai 2 27B   https://docs.prismml.com/models/bonsai-2-27b   ("Best practices")
# Bonsai 2 9B    https://docs.prismml.com/models/bonsai-2-9b    ("Best practices")
# Gemma 4 E4B    https://ai.google.dev/gemma/docs/core/model_card_4  ("Sampling Parameters")
# Qwen3-8B       https://huggingface.co/Qwen/Qwen3-8B            ("Best Practices")
#                https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html (llama-cli cmd)
# Llama 3.1      meta-llama/Llama-3.1-8B-Instruct generation_config.json (temp 0.6, top_p 0.9)
# Qwen2.5-VL     https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct (generation_config.json)
#                https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html (Qwen3 flags)
# Qwen2.5-Coder  https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct (generation_config.json)
# Qwen3-ASR      https://huggingface.co/Qwen/Qwen3-ASR-1.7B      ("Greedy search was used")
# SmolVLM2       https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
# MiMo-V2.6      XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B generation_config.json (temp 0.6, top_p 0.95, top_k 20)
# Qwen3.5-9B     https://huggingface.co/Qwen/Qwen3.5-9B  (non-thinking 0.7/0.8/20; thinking 1.0/0.95/20)
# Nemotron 3 4B  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 (reasoning 1.0/0.95; tool calling 0.6/0.95)
# Nemotron 9B v2 https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2 (reasoning on: 0.6/0.95)
# DeepSeek-R1    https://github.com/deepseek-ai/DeepSeek-R1 (temp 0.5-0.7, recommended 0.6)
# Ministral 3 8B https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512 (Instruct: temp 0.15 or 0.1)
# OLMoE 1B-7B   https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct  (ctx 4096; no fixed sampling table)
# Qwen1.5-MoE  https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat        (generation_config.json: 0.7/0.8/20/1.05)
# llama.cpp CLI  https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
PRESETS = [
    preset("1. Bonsai 2 27B - thinking (8k ctx)", BONSAI_27B_FILE, ctx(8192) + KV_27B + SAMPLING_27B + REASONING_ON + think(-1)),
    preset("1b. Bonsai 2 27B - thinking (4k ctx, lighter)", BONSAI_27B_FILE, ctx(4096, b=128, ub=64) + KV_27B + SAMPLING_27B + REASONING_ON + think(-1)),
    preset("1c. Bonsai 2 27B - no thinking (8k ctx)", BONSAI_27B_FILE, ctx(8192) + KV_27B + BONSAI_INSTRUCT_SAMPLING + REASONING_OFF),
    preset("2. Bonsai 2 9B - thinking (16k ctx)", BONSAI_9B_FILE, ctx() + KV + SAMPLING_9B + REASONING_ON + think(-1)),
    preset("2b. Bonsai 2 9B - no thinking (16k ctx)", BONSAI_9B_FILE, ctx() + KV + BONSAI_INSTRUCT_SAMPLING + REASONING_OFF),
    preset("3. Bonsai 2 9B - images + thinking, chat only" + ("" if _BONSAI9_MM else " (text only)"), BONSAI_9B_FILE, ctx(**VISION) + KV + SAMPLING_9B + REASONING_ON + think(-1), mmproj=_BONSAI9_MM),
    preset("4. Gemma 4 E4B - no thinking (24k ctx)", "gemma-4-E4B_q4_0-it.gguf", ctx(24576) + KV + GEMMA_SAMPLING + REASONING_OFF),
    preset("4b. Gemma 4 E4B - thinking (24k ctx)", "gemma-4-E4B_q4_0-it.gguf", ctx(24576) + KV + GEMMA_SAMPLING + REASONING_ON),
    preset("4c. Gemma 4 12B - no thinking (16k ctx)", "gemma-4-12b-it-qat-q4_0.gguf", ctx(16384) + ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--cache-ram", "1024"] + GEMMA_SAMPLING + REASONING_OFF),
    preset("4d. Gemma 4 12B - thinking (8k ctx)", "gemma-4-12b-it-qat-q4_0.gguf", ctx(8192) + ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--cache-ram", "1024"] + GEMMA_SAMPLING + REASONING_ON),
    preset("5. Qwen3 8B - thinking (16k ctx)", "Qwen3-8B-Q4_K_M.gguf", ctx(16384) + KV + QWEN_SAMPLING + REASONING_ON + think(-1)),
    preset("5b. Qwen3 8B - no thinking (16k ctx)", "Qwen3-8B-Q4_K_M.gguf", ctx(16384) + KV + QWEN_NON_THINKING_SAMPLING + REASONING_OFF),
    preset("6. Llama 3.1 8B - general (8k ctx)", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", ctx(8192) + KV + LLAMA_SAMPLING),
    preset("7. Gemma 4 E4B - images (16k ctx)" + ("" if _GEMMA_MM else " (text only)"), "gemma-4-E4B_q4_0-it.gguf", ctx(16384, **VISION) + KV + GEMMA_SAMPLING + REASONING_OFF, mmproj=_GEMMA_MM),
    preset("8. Qwen2.5-VL 7B - images (8k ctx)" + ("" if _QWENVL_MM else " (text only)"), "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", ctx(8192, **VISION) + ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--cache-ram", "512"] + QWENVL_GENCFG_SAMPLING, mmproj=_QWENVL_MM),
    # ggml-org/ultravox-v0_5-llama-3_1-8b-GGUF ships its weights under the same file name as plain Llama 3.1;
    # the projector is what makes it Ultravox.
    preset("9. Ultravox audio - experimental (4k ctx)" + ("" if _ULTRAVOX_MM else " (text only)"), "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", ctx(4096, **VISION) + ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--cache-ram", "256"] + LLAMA_SAMPLING, mmproj=_ULTRAVOX_MM),
    preset("10. Qwen3-ASR 1.7B - speech to text (8k ctx)" + ("" if _ASR_MM else " (text only)"), "Qwen3-ASR-1.7B-Q8_0.gguf", ctx(8192, **VISION) + ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--cache-ram", "512"] + ASR_SAMPLING, mmproj=_ASR_MM),
    preset("11. SmolVLM2 500M - video (8k ctx)" + ("" if _SMOL_MM else " (text only)"), "SmolVLM2-500M-Video-Instruct-Q8_0.gguf", ctx(8192, **VISION) + KV + ["--video-fps", "1"] + SMOL_SAMPLING, mmproj=_SMOL_MM),
    # The code presets get more room than the others: a script of 200 lines is about
    # 2500 tokens, and it also has to fit the question, the file being worked on and
    # the answer. At 4096 the 14B ran out of context in the middle of a script.
    preset("12. Qwen2.5-Coder 7B - coding (16k ctx)", "Qwen2.5-Coder-7B-Instruct-Q6_K.gguf", ctx(16384) + KV + QWEN_CODER_SAMPLING, coding=True),
    preset("13. Qwen2.5-Coder 14B - coding (8k ctx)", "qwen2.5-coder-14b-instruct-q4_k_m.gguf", ctx(8192, b=128, ub=32, ngl=16) + ["-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0", "--cache-ram", "2048"] + QWEN_CODER_SAMPLING, coding=True),
    preset("13b. Qwen2.5-Coder 14B - coding, lighter (4k ctx)", "qwen2.5-coder-14b-instruct-q4_k_m.gguf", ctx(4096, b=128, ub=32, ngl=16) + ["-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0", "--cache-ram", "2048"] + QWEN_CODER_SAMPLING, coding=True),
    preset("14. MiMo-V2.6-Distill-Qwen-9B - agentic (8k ctx)", "MiMo-V2.6-Distill-Qwen-9B-Q4_K_M-OFFICIAL.gguf", ctx(8192) + KV + MIMO_SAMPLING + REASONING_ON),
    # Decode OOM on a 512-token prompt: GPU-side memory, so fewer GPU layers. KV stays q8_0 (this hybrid model
    # keeps KV in only 8 of 32 layers, so a q4_0 switch saves a few MiB). --cache-ram is host RAM, not GPU.
    preset("15. MiMo-V2.6-Distill-Qwen-9B Q8_0 - agentic (2k ctx)", "MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf", ctx(2048, b=64, ub=32, ngl=4) + ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--cache-ram", "1024"] + ["--jinja", "--temp", "0.6", "--top-p", "0.95", "--top-k", "20"]),
    preset("16. Qwen3.5-9B - no thinking (8k ctx)", "Qwen3.5-9B-Q4_K_M-OFFICIAL.gguf", ctx(8192) + KV + QWEN35_SAMPLING + REASONING_OFF),
    preset("17. Nemotron 3 Nano 4B - thinking (64k ctx)", "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf", ctx(65536) + KV + NEMOTRON_4B_SAMPLING + REASONING_ON),
    preset("18. DeepSeek-R1-Distill-Qwen-7B - thinking (8k ctx)", "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M-OFFICIAL.gguf", ctx(8192) + KV + DEEPSEEK_R1_SAMPLING + REASONING_ON),
    preset("19. Ministral 3 8B - general (8k ctx)", "Ministral-3-8B-Instruct-2512-BF16-Q4_K_M-OFFICIAL.gguf", ctx(8192) + KV + MINISTRAL_SAMPLING),
    preset("20. Nemotron-Nano 9B - thinking (4k ctx)", "Nemotron-Nano-9B-v2-Q4_K_M-OFFICIAL.gguf", ctx(4096, b=128, ub=32, ngl=16) + KV + ["--jinja", "--temp", "0.6", "--top-p", "0.95"]),
    # OLMoE: the official max context is 4096, so this preset uses exactly that (nothing to cut).
    preset("21. OLMoE 1B-7B - general (4k ctx)", "OLMoE-Q6_K.gguf", ctx(4096, b=512, ub=512, ngl=99) + KV + OLMOE_SAMPLING),
    # Qwen1.5-MoE: official max is 32768, but that OOMs at this file size on a 7-9 GB laptop.
    # 2048 is the context that actually runs; 4096 is kept as an "if you have the RAM" variant.
    preset("22. Qwen1.5-MoE-A2.7B-Chat - general (2k ctx)", "Qwen-MoE-Q3_K_M.gguf", ctx(2048, b=512, ub=512, ngl=99) + KV + QWEN15MOE_SAMPLING),
    preset("22b. Qwen1.5-MoE-A2.7B-Chat - general (4k ctx)", "Qwen-MoE-Q3_K_M.gguf", ctx(4096, b=512, ub=512, ngl=99) + KV + QWEN15MOE_SAMPLING),
    
]
def _preset_files(p):
    names = [p["exe"]]
    for flag in ("-m", "--mmproj"):
        if flag in p["args"]:
            f = p["args"][p["args"].index(flag) + 1]
            # FIX 4: use basename, not f[2:] slicing
            names.append(os.path.basename(f))
    return names

MISSING_FILES = {p["name"]: [n for n in _preset_files(p) if not os.path.isfile(os.path.join(BASE_DIR, n))]
                 for p in PRESETS}
# FIX 4: DO NOT hide missing presets. Keep them all visible so the user sees why one fails.
# The startup log lists what is missing for each of them instead.

PROCESS_NAMES = ["llama-server.exe", "llama-bench.exe"]
# llama.cpp prints one line per buffer, device name in the middle:
#   llama_model_load:          CPU model buffer size =  4096.00 MiB
#   llama_kv_cache_init:       CUDA0 KV buffer size =  1024.00 MiB
#   llama_memory_recurrent:    CUDA0 RS buffer size =    12.34 MiB
#   llama_context:             CPU compute buffer size =  512.00 MiB
#   llama_context:             CPU  output buffer size =     1.00 MiB
MEM_RE = re.compile(r"(\S+)\s+(model|KV|RS|compute|output)\s+buffer size\s*=\s*([\d.]+)\s*MiB")
# server trace line: " - cache state: 3 prompts, 12.345 MiB (limits: 2048.000 MiB, ...)"
# (trace level only - the launcher starts the server with --verbosity 4 for it)
CACHE_RE = re.compile(r"cache state:\s*(\d+)\s+prompts?,\s*([\d.]+)\s*MiB")
# "failed to allocate output buffer of size 1234.56 MiB"  (the size is already MiB)
FAIL_MIB_RE = re.compile(r"failed to allocate (\S+) buffer of size ([\d.]+)\s*MiB")
# "failed to allocate compute pp buffers" / "failed to allocate compute buffers"
FAIL_RAW_RE = re.compile(r"failed to allocate (\S+) buffer of size (\d+)\s*$")   # bytes, e.g. "...Vulkan0 buffer of size 176173060"
FAIL_BUF_RE = re.compile(r"failed to allocate (compute pp buffers|compute tg buffers|compute buffers)")
# "failed to allocate a 123456-byte staging buffer"
FAIL_BYTES_RE = re.compile(r"failed to allocate a (\d+)-byte")

COMPONENTS = [
    ("model", "Model weights", "#4c78a8", "Model file. Updated below."),
    ("kv", "KV cache", "#f58518", "Memory for this chat. Updated below."),
    ("rs", "Recurrent state", "#b279a2", "Fixed memory. Only some models need it."),
    ("compute", "Compute buffers", "#54a24b", "Rough work for reading your question. Updated below."),
    ("output", "Output buffer", "#e45756", "Marks for the next word. Updated below."),
    ("cache", "Prompt cache / growth", "#72b7b2", "Saved work from old questions. Updated below."),
    ("other", "Other host RAM", "#b0b0b0", "The app's own memory. Updated below."),
]
# Short names for the stacked bar. The long ones above are for the table under it.
BAR_NAMES = {"model": "Weights", "kv": "KV", "rs": "RS", "compute": "Compute",
             "output": "Output", "cache": "Cache", "other": "Other"}

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]

def ram_info():
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    gb = 1024 ** 3
    used = (st.ullTotalPhys - st.ullAvailPhys) / gb
    return used, st.ullTotalPhys / gb, st.dwMemoryLoad

def is_admin():
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False

def clear_standby_ram():
    if not is_admin():
        return False, "not admin - process memory freed, but Windows file cache not flushed"
    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]
    class LUID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [("Luid", LUID), ("Attributes", wintypes.DWORD)]
    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_ = [("PrivilegeCount", wintypes.DWORD), ("Privileges", LUID_AND_ATTRIBUTES * 1)]
    k32, adv, nt = ctypes.windll.kernel32, ctypes.windll.advapi32, ctypes.windll.ntdll
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    adv.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    adv.AdjustTokenPrivileges.argtypes = [wintypes.HANDLE, wintypes.BOOL, ctypes.POINTER(TOKEN_PRIVILEGES),
                                          wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p]
    adv.LookupPrivilegeValueW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.POINTER(LUID)]
    token = wintypes.HANDLE()
    if not adv.OpenProcessToken(k32.GetCurrentProcess(), 0x28, ctypes.byref(token)):
        return False, "could not open process token"
    try:
        luid = LUID()
        if not adv.LookupPrivilegeValueW(None, "SeProfileSingleProcessPrivilege", ctypes.byref(luid)):
            return False, f"could not look up memory privilege (Windows error {k32.GetLastError()})"
        tp = TOKEN_PRIVILEGES()
        tp.PrivilegeCount = 1
        tp.Privileges[0].Luid = luid
        tp.Privileges[0].Attributes = 2
        # AdjustTokenPrivileges may return success while not assigning the requested privilege.
        k32.SetLastError(0)
        if not adv.AdjustTokenPrivileges(token, False, ctypes.byref(tp), 0, None, None):
            return False, f"could not adjust memory privilege (Windows error {k32.GetLastError()})"
        if k32.GetLastError() == 1300:  # ERROR_NOT_ALL_ASSIGNED
            return False, "this token does not have the required memory privilege"
        for cmd in (3, 4):
            c = ctypes.c_int(cmd)
            status = nt.NtSetSystemInformation(80, ctypes.byref(c), ctypes.sizeof(c))
            if status != 0:
                return False, f"NtSetSystemInformation({cmd}) failed: 0x{status & 0xFFFFFFFF:08X}"
        return True, "standby/cached RAM flush requested"
    finally:
        k32.CloseHandle(token)

def kill_all_llama():
    # Kept for manual use only. NOT called automatically on Start/Stop anymore (FIX 5).
    for name in PROCESS_NAMES:
        subprocess.run(["taskkill", "/F", "/T", "/IM", name],
                       capture_output=True, creationflags=NO_WINDOW)

class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

_API = {}
def _api():
    if not _API:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ps = ctypes.WinDLL("psapi", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                   ctypes.POINTER(wintypes.DWORD)]
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        ps.EmptyWorkingSet.argtypes = [wintypes.HANDLE]
        ps.EnumProcesses.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD,
                                     ctypes.POINTER(wintypes.DWORD)]
        _API["k32"], _API["ps"] = k32, ps
    return _API["k32"], _API["ps"]

_ACCESS = 0x1000 | 0x0400 | 0x0010 | 0x0100

def _working_set(ps, h):
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    if ps.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
        return pmc.WorkingSetSize
    return 0

def list_processes():
    k32, ps = _api()
    arr = (wintypes.DWORD * 4096)()
    need = wintypes.DWORD()
    ps.EnumProcesses(arr, ctypes.sizeof(arr), ctypes.byref(need))
    n = need.value // ctypes.sizeof(wintypes.DWORD)
    groups = {}
    me = os.getpid()
    for pid in arr[:n]:
        if pid in (0, 4, me):
            continue
        h = k32.OpenProcess(_ACCESS, False, pid)
        if not h:
            continue
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            if not k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                continue
            name = os.path.basename(buf.value)
            g = groups.setdefault(name, {"pids": [], "ws": 0})
            g["pids"].append(pid)
            g["ws"] += _working_set(ps, h)
        finally:
            k32.CloseHandle(h)
    return groups

def trim_pids(pids):
    k32, ps = _api()
    released, failed = 0, 0
    for pid in pids:
        h = k32.OpenProcess(_ACCESS, False, pid)
        if not h:
            failed += 1
            continue
        try:
            before = _working_set(ps, h)
            if ps.EmptyWorkingSet(h):
                released += max(0, before - _working_set(ps, h))
            else:
                failed += 1
        finally:
            k32.CloseHandle(h)
    return released, failed

def process_memory(pid):
    k32, ps = _api()
    h = k32.OpenProcess(0x1000 | 0x0400 | 0x0010, False, pid)
    if not h:
        return None
    try:
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if ps.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize, pmc.PagefileUsage
    finally:
        k32.CloseHandle(h)
    return None

_GGUF_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_GGUF_CACHE = {}

# FIX 2: only read keys we need, no fragile early-break on tokenizer order.
_WANTED_SUFFIXES = (".block_count", ".embedding_length", ".attention.head_count",
                    ".attention.head_count_kv", ".attention.key_length",
                    ".attention.value_length", ".full_attention_interval",
                    ".context_length", ".ssm.inner_size")
_WANTED_EXACT = {"general.architecture", "general.size_label"}

def read_gguf_info(path):
    ck = (path, os.path.getmtime(path))
    if ck in _GGUF_CACHE:
        return _GGUF_CACHE[ck]
    info = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        f.read(4)
        _n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        def rstr():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")
        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                if et in _GGUF_SCALAR:
                    c = _GGUF_SCALAR[et]
                    data = f.read(struct.calcsize(c) * n)
                    return list(struct.unpack(f"<{n}{c}", data)) if n <= 1024 else None
                for _ in range(n):
                    rval(et)
                return None
            c = _GGUF_SCALAR[t]
            return struct.unpack("<" + c, f.read(struct.calcsize(c)))[0]
        for _ in range(n_kv):
            k = rstr()
            (t,) = struct.unpack("<I", f.read(4))
            # Still must consume the value to keep file offset correct, even if not wanted.
            v = rval(t)
            if k in _WANTED_EXACT or any(k.endswith(s) for s in _WANTED_SUFFIXES):
                info[k] = v
            # Also keep arch early so suffix matching works - arch is in _WANTED_EXACT
    info["arch"] = info.get("general.architecture")
    _GGUF_CACHE[ck] = info
    return info

_BYTES_PER_ELEM = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
                   "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}

def estimate_kv_mib(info, n_ctx, ctk, ctv):
    arch = info.get("arch")
    if not arch or not n_ctx:
        return None
    def g(k):
        return info.get(f"{arch}.{k}")
    n_layer, n_embd = g("block_count"), g("embedding_length")
    n_head, n_kv = g("attention.head_count"), g("attention.head_count_kv")
    if not n_layer or not n_embd:
        return None
    if n_kv is None:
        n_kv = n_head
    if n_kv is None:
        return None
    kv_heads = list(n_kv) if isinstance(n_kv, list) else [n_kv] * n_layer
    if len(kv_heads) != n_layer:
        return None
    interval = g("full_attention_interval")
    if isinstance(interval, int) and interval > 1:
        kv_heads = [h if (i + 1) % interval == 0 else 0 for i, h in enumerate(kv_heads)]
    head_ref = max(n_head) if isinstance(n_head, list) else n_head
    key_len = g("attention.key_length") or (n_embd // head_ref if head_ref else None)
    if not key_len:
        return None
    val_len = g("attention.value_length") or key_len
    bk, bv = _BYTES_PER_ELEM.get(ctk, 2.0), _BYTES_PER_ELEM.get(ctv, 2.0)
    return sum(kv_heads) * n_ctx * (key_len * bk + val_len * bv) / 1024**2

def fmt_mib(mib):
    return f"{mib:,.0f} MiB ({mib / 1024:.2f} GB)"

# ----------------------------------------------------------------------------
# AUTO NGL. START does the work:
#   no saved value    -> trial loop (real load + real generation with a long prompt), save, start
#   saved value       -> start at once with it
#   OOM in real use   -> drop to the next lower value, save, restart
# ----------------------------------------------------------------------------
AUTO_NGL_FILE = os.path.join(BASE_DIR, "auto_ngl.json")
AUTO_CANDIDATES = [32, 16, 8, 4, 0]     # descending; only values <= the real layer count are used
AUTO_OOM_WORDS = ("outofdevicememory", "outofhostmemory", "decode() failed", "failed to allocate", "ggml_assert")
AUTO_LOAD_TIMEOUT = 240                 # seconds for /health to answer
AUTO_TEST_TIMEOUT = 600                 # seconds for the generation test (slow at low ngl)
AUTO_FREE_RESERVE_MIB = 1024            # RAM kept free for the OS while the server loads
AUTO_FREE_RAM_RISE_MIB = 4096           # more free RAM than this since tuning = the saved value is probably too low
AUTO_RETUNE_COOLDOWN_S = 3600           # at most one automatic re-tune per hour, per preset
AUTO_STEPDOWN_COOLDOWN_S = 600          # after an out-of-memory step-down, no re-tune for 10 minutes
AUTO_RAM_SAMPLES = 3                    # free RAM is read this many times


def _base(path):
    """File name from a Windows or POSIX path, on any OS."""
    return re.split(r"[\\/]", path)[-1]


def _port_listening(port):
    """True if something accepts connections on 127.0.0.1:port. Works the same on Windows and Linux
    (a bind() test does not: SO_REUSEADDR and TIME_WAIT make it unreliable)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
        sk.settimeout(1)
        return sk.connect_ex(("127.0.0.1", port)) == 0


def _is_oom_line(line):
    low = line.lower()
    if "pinned memory" in low:          # a warning real loads print and survive, not a failure
        return False
    return any(w in low for w in AUTO_OOM_WORDS)


def _first_oom(lines):
    for line in lines:
        if _is_oom_line(line):
            return line.strip()
    return None


def _copy_version(path):
    """Read APP_VERSION out of another copy of the launcher, if it has one."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            m = re.search(r'APP_VERSION\s*=\s*"([^"]+)"', f.read())
        return m.group(1) if m else None
    except Exception:
        return None

def _other_launcher_copies():
    """Find other copies of this launcher that would fight over auto_ngl.json.

    Every copy with the same BASE_DIR shares the same auto_ngl.json. Copies of
    the same version compute the same keys, so they are harmless. A copy of a
    different version computes different keys, so the two keep re-tuning each
    other and overwriting the saved values. Only those are reported.
    """
    try:
        me = os.path.abspath(__file__)
    except Exception:
        return []
    found = []
    for folder in (BASE_DIR, os.path.dirname(me)):
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        for name in names:
            low = name.lower()
            if not (low.startswith("llama_launcher") and low.endswith(".pyw")):
                continue
            path = os.path.abspath(os.path.join(folder, name))
            if path == me or path in found:
                continue
            other = _copy_version(path)
            if other != APP_VERSION:
                found.append((path, other))
    return found

def _is_crash_exit(code):
    """True for a Windows crash exit code (0xC000.... = user-mode exception).

    The Vulkan driver gives up by killing the process, and it can do that
    without printing an out-of-memory line first. The launcher would then see
    a dead server with nothing to recognise, so the crash code itself is the
    signal that the GPU was asked for too much.
    """
    try:
        c = int(code)
    except Exception:
        return False
    return 0xC0000000 <= c <= 0xC000FFFF


def _sampling_mismatch(rec, mode):
    """True if the record was tuned with different sampling settings.

    The sampler is part of the memory footprint, so a value tuned with one
    sampler is not necessarily right for another. A record with no sampling
    field comes from an older launcher: keep it rather than re-tune everything.
    """
    if not rec or "sampling" not in rec:
        return False
    return rec.get("sampling") != mode


def _auto_applies(p):
    """All llama-server presets with -ngl. No model is excluded."""
    return p["exe"] == "llama-server.exe" and "-ngl" in p["args"] and "-m" in p["args"]


def _free_ram_mib():
    try:
        used, total, _ = ram_info()
        return int((total - used) * 1024)
    except Exception:
        return None

def _wait_for_ram_settle(seconds=25.0, poll=0.5, delta=48):
    """Block until free RAM stops moving.

    A killed llama-server gives its Vulkan shared memory back slowly. Starting the
    next server straight away puts it into a half-drained pool, and it then fails on
    allocations of a few KB even though it loaded fine a moment ago. Wait until the
    pool has stopped shrinking before starting the next one.
    """
    try:
        prev = _free_ram_mib()
        if prev is None:
            return
        deadline = time.time() + seconds
        stable = 0
        while time.time() < deadline:
            time.sleep(poll)
            cur = _free_ram_mib()
            if cur is None:
                return
            if abs(cur - prev) < delta:       # the pool stopped moving
                stable += 1
                if stable >= 5:               # 2.5 s of quiet, not 1.5 s
                    return
            else:
                stable = 0
            prev = cur
    except Exception:
        return

def _free_ram_reading(samples=AUTO_RAM_SAMPLES, gap=1.0, mode="min"):
    """Free RAM, read a few times.

    mode="min" gives the lowest reading.  That is the one to use when asking "how much
    RAM is free now?": the lowest reading is the one that is still true in the worst
    moment, and the Windows file cache alone moves this number by more than a gigabyte
    when a 7 GB model file is read or trimmed.

    mode="max" gives the highest reading.  That is the one to use for the baseline a
    saved value is compared against.  Right after a server was killed its memory is
    still coming back, so a reading taken then is far too low; a baseline that is too
    low makes the launcher think the machine got faster, and it starts a re-tune that
    was not needed.
    """
    try:
        n = max(1, int(samples))
    except Exception:
        n = 1
    readings = []
    for i in range(n):
        r = _free_ram_mib()
        if r is not None:
            readings.append(r)
        if i + 1 < n:
            time.sleep(gap)
    if not readings:
        return None
    return max(readings) if mode == "max" else min(readings)


def _auto_is_stale(p, rec):
    """True if free RAM dropped a lot since tuning. Saved value was tuned with more RAM free."""
    if not rec or "free_ram_mib_at_tune" not in rec:
        return False
    tuned = rec.get("free_ram_mib_at_tune")
    if not isinstance(tuned, (int, float)):   # tuning ran without a RAM reading: cannot judge
        return False
    cur = _free_ram_reading(mode="min")
    if cur is None:
        return False
    return cur < tuned - 1536


def _auto_is_pessimistic(p, rec):
    """True if a lot more RAM is free now than when the value was tuned.

    The saved number of GPU layers was measured with less RAM free, so it is probably
    too low.  Without this rule a record written by the out-of-memory step-down stays
    at its lowered value for ever: nothing ever raises it again, even after the user
    closes the apps that caused the out-of-memory in the first place.

    Three things keep this rule from running wild:
      * the margin is large, because the file cache alone moves the reading by gigabytes
      * the reading is the lowest of a few samples, not a single number
      * a value that was just tuned is locked for an hour, so a re-tune cannot
        immediately claim that there is even more RAM free and start another one
    """
    if not rec or "free_ram_mib_at_tune" not in rec:
        return False
    tuned = rec.get("free_ram_mib_at_tune")
    if not isinstance(tuned, (int, float)):
        return False
    lock = rec.get("retune_lock_until")
    if isinstance(lock, (int, float)) and time.time() < lock:
        return False               # tuned recently: leave it alone for now
    cur = _free_ram_reading(mode="min")
    if cur is None:
        return False
    layers = _auto_layers(p)
    if isinstance(layers, int) and rec.get("ngl", 0) >= layers:
        return False            # already fully offloaded: a re-tune cannot give more
    return cur > tuned + AUTO_FREE_RAM_RISE_MIB

def _auto_key(p):
    """Model file + size + mtime + ctx + KV type + batch sizes + cache-ram + mmproj.
    A reconverted file gets a new key. A changed --cache-ram also gets a new key.
    -ngl is not part of the key: it is the value being tuned."""
    if not _auto_applies(p):
        return None
    a = p["args"]

    def val(flag):
        return a[a.index(flag) + 1] if flag in a else None

    rel = val("-m")
    try:
        st = os.stat(os.path.join(BASE_DIR, rel))
    except OSError:
        return None
    mm = None
    if val("--mmproj"):
        try:
            ms = os.stat(os.path.join(BASE_DIR, val("--mmproj")))
            mm = [ms.st_size, int(ms.st_mtime)]
        except OSError:
            mm = ["missing"]
    blob = json.dumps({"model": _base(rel), "size": st.st_size, "mtime": int(st.st_mtime),
                       "mmproj": mm, "ctx": val("-c"), "ctk": val("-ctk"), "ctv": val("-ctv"),
                       "fa": val("-fa"), "b": val("-b"), "ub": val("-ub"),
                       "cache_ram": val("--cache-ram")}, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def load_auto_ngl():
    try:
        with open(AUTO_NGL_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        cleaned = {k: v for k, v in data.items() if isinstance(v, dict) and "cache_ram" in v and "ngl" in v and "ctx" in v and "mmproj" in v}
        if len(cleaned) != len(data):
            try:
                save_auto_ngl(cleaned)
            except Exception:
                pass
        return cleaned
    except Exception:
        return {}


def save_auto_ngl(data):
    tmp = AUTO_NGL_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, AUTO_NGL_FILE)


AUTO_CACHE = load_auto_ngl()


def _auto_record(p):
    key = _auto_key(p)
    return AUTO_CACHE.get(key) if key else None


def _auto_layers(p):
    path = os.path.join(BASE_DIR, p["args"][p["args"].index("-m") + 1])
    try:
        info = read_gguf_info(path)
    except Exception:
        return None
    arch = info.get("arch")
    n = info.get(f"{arch}.block_count") if arch else None
    if isinstance(n, int) and n > 0:
        return n
    for k, v in info.items():
        if k.endswith(".block_count") and isinstance(v, int) and v > 0:
            return v
    v = info.get("general.block_count")
    if isinstance(v, int) and v > 0:
        return v
    return None


def _auto_trials(layers):
    """Coarse list only, no duplicate 99 trial. Top is layers + 1 for full offload.
    36 layers -> [37, 36, 32, 16, 8, 4, 0]. 16 layers -> [17, 16, 8, 4, 0]. Unknown -> [99, 32, 16, 8, 4, 0].
    Fine values like 24, 20, 12 are found later by the climb, not by trying every value."""
    if layers is None:
        return [99] + AUTO_CANDIDATES
    top = layers + 1
    vals = sorted({v for v in AUTO_CANDIDATES if v <= layers} | {layers, top}, reverse=True)
    return vals


def _http_json(url, body=None, timeout=5):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8", "replace") or "{}")


def _test_prompt(ctx_tokens):
    """About 75% of the context (capped): the decode path is exercised, not just a short ping."""
    target = max(256, min(int(ctx_tokens * 0.75), 2500))
    return "The quick brown fox jumps over the lazy dog. " * (target // 10 + 1) + "Reply with the single word OK."


def _auto_try(p, port, ngl, cancel, args=None):
    """One trial. Returns (status, detail, meta); status is 'ok', 'oom', 'other' or 'cancelled'."""
    exe = os.path.join(BASE_DIR, p["exe"])
    args = list(args if args is not None else p["args"])
    args[args.index("-ngl") + 1] = str(ngl)
    ctx_tokens = int(args[args.index("-c") + 1]) if "-c" in args else 4096
    cmd = [exe] + args + ["--port", str(port)] + EXTRA_SERVER_ARGS
    lines = []

    def reader(proc):
        for line in proc.stdout:
            lines.append(line.rstrip("\n"))

    kill_all_llama()        # a trial must not share the port with another server
    t_free = time.time()
    while _port_listening(port):                 # give the previous trial's server time to let go
        if time.time() - t_free > 15:
            return "other", f"port {port} is already in use by another program; close it or change the port", {}
        if cancel.is_set():
            return "cancelled", "", {}
        time.sleep(0.5)
    # The port is free, but the previous trial's Vulkan memory may still be draining.
    # Starting this trial into a half-drained pool makes it fail below the real limit,
    # so the tuner would record a value that is too low.
    if not cancel.is_set():
        _wait_for_ram_settle()
    proc = subprocess.Popen(cmd, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
    threading.Thread(target=reader, args=(proc,), daemon=True).start()
    try:
        # 1) load: wait for /health; an OOM line or an early exit ends the trial at once
        t0 = time.time()
        while True:
            if cancel.is_set():
                return "cancelled", "", {}
            hit = _first_oom(lines)
            if hit:
                return "oom", hit, {}
            if proc.poll() is not None:
                return "other", f"server exited (code {proc.returncode}): {lines[-1] if lines else 'no output'}", {}
            try:
                st, _ = _http_json(f"http://127.0.0.1:{port}/health", timeout=2)
                if st == 200:
                    break
            except Exception:
                pass
            if time.time() - t0 > AUTO_LOAD_TIMEOUT:
                return "other", "timed out loading", {}
            time.sleep(0.5)
        load_s = round(time.time() - t0, 1)

        # 2) generation with a long prompt: this is where the decode OOM showed up
        mark = len(lines)
        body = {"messages": [{"role": "user", "content": _test_prompt(ctx_tokens)}],
                "max_tokens": 64, "temperature": 0, "stream": False}
        try:
            _, res = _http_json(f"http://127.0.0.1:{port}/v1/chat/completions", body, timeout=AUTO_TEST_TIMEOUT)
        except urllib.error.HTTPError as e:
            # The server prints the OOM line, then answers 500. The reader thread may not have read that
            # line yet, so give the log a few seconds to catch up, and also check the response body.
            body = e.read()[:400].decode("utf-8", "replace")
            hit = _first_oom(lines[mark:]) or (body if any(w in body.lower() for w in AUTO_OOM_WORDS) else None)
            t_wait = time.time()
            while not hit and time.time() - t_wait < 3:
                time.sleep(0.1)
                hit = _first_oom(lines[mark:])
            if hit:
                return "oom", hit, {}
            return "other", f"HTTP {e.code} during generation: {body[:160]!r}", {}
        except Exception as e:
            if cancel.is_set():
                return "cancelled", "", {}
            return "other", f"generation failed: {e}", {}
        hit = _first_oom(lines[mark:])
        if hit:
            return "oom", hit, {}
        usage = res.get("usage", {}) if isinstance(res, dict) else {}
        if usage.get("completion_tokens", 0) < 1:
            return "other", f"no tokens generated: {str(res)[:120]}", {}
        return "ok", "ok", {"prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"), "load_s": load_s}
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=10)
        except Exception:
            pass


def _auto_safe_start(p, layers, free_mib):
    """If file + KV + cache-ram + 1 GB reserve is bigger than free RAM, full offload will almost surely OOM.
    Return the highest coarse value to START from. Climb will still try higher if it actually works."""
    if layers is None or free_mib is None:
        return None
    try:
        rel = p["args"][p["args"].index("-m") + 1]
        fpath = os.path.join(BASE_DIR, rel)
        fsize = os.path.getsize(fpath) / 1024 ** 2
        a = p["args"]

        def val(flag, default=None):
            return a[a.index(flag) + 1] if flag in a else default

        ctx_tokens = int(val("-c", 4096))
        try:
            info = read_gguf_info(fpath)
            kv = estimate_kv_mib(info, ctx_tokens, val("-ctk", "f16"), val("-ctv", "f16")) or 0
        except Exception:
            kv = 0
        cache_ram = int(val("--cache-ram", 2048))
        need = fsize + kv + cache_ram + AUTO_FREE_RESERVE_MIB
        if need <= free_mib:
            return None
        half = layers // 2
        cands = [v for v in _auto_trials(layers) if v <= half]
        return cands[0] if cands else 0
    except Exception:
        return None


def auto_tune_ngl(p, port, out, cancel, args=None):
    """Worker thread (no Tk calls). Puts ('log', text), then exactly one of
    ('done', ngl, meta), ('failed', reason) or ('cancelled', None).
    Phase 1: coarse trials, full prompt. Phase 2: binary-search climb to the highest working value."""
    try:
        try:
            layers = _auto_layers(p)
            note = ""
        except Exception as e:
            layers, note = None, f" (could not read the layer count: {e})"
        trials = _auto_trials(layers)
        free = _free_ram_mib()
        cache_ram = None
        a = p["args"]
        if "--cache-ram" in a:
            cache_ram = a[a.index("--cache-ram") + 1]
        safe = _auto_safe_start(p, layers, free)
        skipped_top = None
        if safe is not None and trials and safe < trials[0]:
            skipped_top = trials[0]
            trials = [v for v in trials if v <= safe]
            out.put(("log", f"[auto ngl] file + KV + cache-ram is bigger than free RAM, skipping top trial {skipped_top}, starting from {trials[0]}. Climb will still try higher if it works."))
        out.put(("log", f"[auto ngl] {p['name']}: layers={layers}{note}, free RAM={free} MiB, cache-ram={cache_ram}, trying -ngl {trials}"))
        low_ok = None
        high_fail = skipped_top
        last_meta = None
        last_error = None
        for ngl in trials:
            if cancel.is_set():
                out.put(("cancelled", None))
                return
            out.put(("log", f"[auto ngl] trying -ngl {ngl} ..."))
            status, detail, meta = _auto_try(p, port, ngl, cancel, args)
            if status == "cancelled":
                out.put(("cancelled", None))
                return
            if status == "ok":
                low_ok = ngl
                last_meta = meta
                break
            if status == "other":
                if "already in use" in detail:
                    out.put(("failed", f"not an out-of-memory error, so a lower -ngl will not help: {detail}"))
                    return
                # The server died without printing an OOM line. A Vulkan allocation kill looks exactly
                # like this: the process is gone, so the client only sees the connection drop.
                # Treat it as too high and keep looking lower for a value that actually works.
                high_fail = ngl
                last_error = detail
                out.put(("log", f"[auto ngl]   -ngl {ngl}: failed with non-OOM error, treating as too high: {detail[:200]}"))
                continue
            high_fail = ngl
            out.put(("log", f"[auto ngl]   -ngl {ngl}: out of memory: {detail[:200]}"))
        if low_ok is None:
            if last_error:
                out.put(("failed", f"no working -ngl in {trials}: the server failed with a non-OOM error: {last_error}"))
            else:
                out.put(("failed", f"out of memory at every -ngl in {trials}: free RAM is too low for this model at this context"))
            return
        if high_fail is not None and high_fail - low_ok > 1:
            lo, hi = low_ok, high_fail
            best = low_ok
            best_meta = last_meta
            floor = low_ok
            out.put(("log", f"[auto ngl] climb: searching between working {lo} and OOM {hi} for the highest working value"))
            while True:
                while hi - lo > 1:
                    if cancel.is_set():
                        out.put(("cancelled", None))
                        return
                    mid = (lo + hi) // 2
                    out.put(("log", f"[auto ngl] climb: trying -ngl {mid} ..."))
                    status, detail, meta = _auto_try(p, port, mid, cancel, args)
                    if status == "cancelled":
                        out.put(("cancelled", None))
                        return
                    if status == "ok":
                        best = mid
                        lo = mid
                        best_meta = meta
                    elif status == "oom":
                        hi = mid
                        out.put(("log", f"[auto ngl]   -ngl {mid}: out of memory"))
                    else:
                        hi = mid
                        out.put(("log", f"[auto ngl]   -ngl {mid}: failed with non-OOM error, treating as too high, keeping best {best}: {detail[:200]}"))
                # The winner is tried once more.  A trial can pass in a lucky moment -
                # the file cache is in a different state, the machine is a little
                # quieter - and then the very same value fails in real use.  One extra
                # trial costs one model load and catches that before the user does.
                out.put(("log", f"[auto ngl] re-checking -ngl {best} ..."))
                status, detail, meta = _auto_try(p, port, best, cancel, args)
                if status == "cancelled":
                    out.put(("cancelled", None))
                    return
                if status == "ok":
                    best_meta = meta
                    break
                if best <= floor:
                    out.put(("log", f"[auto ngl]   -ngl {best} failed the re-check too ({detail[:160]}). "
                                    f"Keeping it: it worked once, and everything lower is slower."))
                    break
                out.put(("log", f"[auto ngl]   -ngl {best} failed the re-check ({detail[:160]}), looking lower"))
                hi = best
                lo = floor
                best = floor
                best_meta = last_meta
            # The climb's winner always sits directly under a value that failed, so by
            # definition it is at the edge of what the machine can hold.  A trial can
            # pass in a better moment than a real request finds (the file cache is in
            # another state, the machine is a little quieter), and the price of being
            # one layer too high is an out-of-memory error in the middle of an answer.
            # One layer of headroom is kept: the value below the winner is strictly
            # safer than the winner itself, and it never goes below the coarse floor.
            ceiling = best
            if best > floor:
                best = best - 1
                out.put(("log", f"[auto ngl] -ngl {ceiling} is the edge of what this machine holds "
                                f"(the value above it failed). Using -ngl {best} instead, so a real "
                                f"request does not run out of memory."))
            low_ok = best
            last_meta = best_meta
            last_meta["ceiling_ngl"] = ceiling
        last_meta["layers"] = layers
        last_meta["free_ram_mib_at_tune"] = free
        last_meta["cache_ram"] = cache_ram
        out.put(("done", low_ok, last_meta))
        return
    except Exception as e:
        out.put(("failed", f"internal error: {e}"))


def _lighter_preset_for(p):
    """A preset that runs the same model file with a smaller context, if there is one.

    Used by the "not enough RAM" warning: naming the lighter cut of the same model is
    more useful than only saying "close some apps".
    """
    a = p["args"]
    if "-m" not in a:
        return None
    model = _base(a[a.index("-m") + 1]).lower()
    try:
        my_ctx = int(a[a.index("-c") + 1]) if "-c" in a else 0
    except ValueError:
        my_ctx = 0
    best = None
    for q in PRESETS:
        b = q["args"]
        if "-m" not in b or q["name"] == p["name"]:
            continue
        if _base(b[b.index("-m") + 1]).lower() != model:
            continue
        try:
            q_ctx = int(b[b.index("-c") + 1]) if "-c" in b else 0
        except ValueError:
            q_ctx = 0
        if q_ctx and my_ctx and q_ctx < my_ctx and (best is None or q_ctx < best[1]):
            best = (q["name"], q_ctx)
    return best


# ---------------------------------------------------------------------------
# Material Design look
#
# tkinter ships no Material theme, so the parts that carry the design - the
# action buttons, the status chip, the cards - are drawn by hand on a canvas,
# and the remaining widgets use the "clam" theme with Material colours.
# Nothing here needs an extra package: it is all tkinter.
# ---------------------------------------------------------------------------
MD_BG = "#F4F5F7"          # window background
MD_CARD = "#FFFFFF"        # card surface
MD_SURFACE = "#FAFAFB"     # inset surface (command preview)
MD_OUTLINE = "#E1E3E8"     # 1 px card border
MD_TEXT = "#1F2328"
MD_TEXT2 = "#6A707A"
MD_PRIMARY = "#3F51B5"     # indigo
MD_PRIMARY_D = "#303F9F"
MD_PRIMARY_T = "#EEF0FB"   # indigo tint, used for hover on flat buttons
MD_TEAL = "#00897B"
MD_TEAL_D = "#00695C"
MD_TEAL_T = "#E4F2F0"
MD_ERROR = "#D32F2F"
MD_ERROR_D = "#B71C1C"
MD_ERROR_T = "#FDECEA"
MD_WARN = "#ED6C02"
MD_OK = "#2E7D32"
MD_OFF_BG = "#E4E6EA"      # disabled button fill
MD_OFF_FG = "#A5AAB2"      # disabled button text
MD_FONT = ("Segoe UI", 10)
MD_FONT_B = ("Segoe UI", 10, "bold")
MD_FONT_S = ("Segoe UI", 9)
MD_FONT_SB = ("Segoe UI", 9, "bold")
MD_FONT_MONO = ("Consolas", 9)


def _round_rect(x0, y0, x1, y1, r, steps=8):
    """Points of a rounded rectangle, ready for create_polygon."""
    r = max(0.0, min(r, (x1 - x0) / 2.0, (y1 - y0) / 2.0))
    pts = []
    corners = ((x1 - r, y0 + r, 270, 360),
               (x1 - r, y1 - r, 0, 90),
               (x0 + r, y1 - r, 90, 180),
               (x0 + r, y0 + r, 180, 270))
    for cx, cy, a0, a1 in corners:
        for i in range(steps + 1):
            a = math.radians(a0 + (a1 - a0) * i / steps)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


class Tooltip:
    """A small hint that appears when the mouse rests on a widget."""

    def __init__(self, widget, text, delay=600):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip = None
        self._after = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress-1>", self._hide, add="+")

    def _schedule(self, event=None):
        self._cancel()
        self._after = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after:
            try:
                self.widget.after_cancel(self._after)
            except Exception:
                pass
            self._after = None

    def _show(self):
        self._after = None
        if self.tip or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
            self.tip = tk.Toplevel(self.widget)
            self.tip.wm_overrideredirect(True)
            self.tip.configure(bg="#333333")
            tk.Label(self.tip, text=self.text, bg="#333333", fg="#FFFFFF",
                     font=MD_FONT_S, justify="left", padx=9, pady=6).pack()
            self.tip.wm_geometry(f"+{x}+{y}")
        except Exception:
            self.tip = None

    def _hide(self, event=None):
        self._cancel()
        if self.tip:
            try:
                self.tip.destroy()
            except Exception:
                pass
            self.tip = None


class MatButton(tk.Canvas):
    """A Material-style button drawn on a canvas.

    ttk buttons cannot do rounded corners or a flat filled look, so the action
    buttons are drawn here: rounded rectangle, flat fill, no border, a darker
    fill on hover and a pressed state.  The icon is a shape, not a character,
    so it cannot fall back to a box on a machine without the right font.
    """

    def __init__(self, parent, text, command=None, variant="filled", icon=None,
                 color=MD_PRIMARY, color_dark=MD_PRIMARY_D, tint=MD_PRIMARY_T,
                 bg=MD_CARD, font=MD_FONT_B, height=40, padx=18, radius=10,
                 tooltip=None):
        self._text = text
        self._cmd = command
        self._variant = variant
        self._icon = icon
        self._color = color
        self._dark = color_dark
        self._tint = tint
        self._bg = bg
        self._font = font
        self._enabled = True
        self._hover = False
        self._press = False
        self._radius = radius
        self._padx = padx
        self._iw = 22 if icon else 0
        self._tw = tkfont.Font(font=font).measure(text)
        super().__init__(parent, width=self._width(), height=height, bg=bg,
                         highlightthickness=0, bd=0, cursor="hand2")
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<ButtonPress-1>", self._down)
        self.bind("<ButtonRelease-1>", self._up)
        self.bind("<Configure>", lambda e: self._draw())
        self._draw()
        if tooltip:
            Tooltip(self, tooltip)

    # ---- appearance ----
    def _width(self):
        return self._tw + self._iw + self._padx * 2 + (10 if self._icon else 0)

    def _size(self):
        """The real size, or the requested one while the widget is not mapped yet.

        The requested width wins when it is the bigger of the two: a label that
        just grew must not be drawn into the old, narrower canvas.
        """
        w = int(self["width"])
        h = int(self["height"])
        rw, rh = self.winfo_width(), self.winfo_height()
        if rw > 1 and rw > w:
            w = rw
        if rh > 1 and rh > h:
            h = rh
        return w, h

    def _draw(self):
        self.delete("all")
        w, h = self._size()
        x0, y0, x1, y1 = 1, 1, w - 1, h - 1
        if self._variant == "filled":
            if self._enabled:
                fill = self._dark if (self._hover or self._press) else self._color
                fg = "#FFFFFF"
            else:
                fill, fg = MD_OFF_BG, MD_OFF_FG
            self.create_polygon(_round_rect(x0, y0, x1, y1, self._radius),
                                fill=fill, outline="", width=0)
        elif self._variant == "outlined":
            if self._enabled:
                fill = self._tint if self._hover else self._bg
                outline, fg = self._color, self._color
            else:
                fill, outline, fg = self._bg, "#DCDEE2", MD_OFF_FG
            self.create_polygon(_round_rect(x0, y0, x1, y1, self._radius),
                                fill=fill, outline=outline, width=1)
        else:                                    # flat / text button
            if self._enabled:
                fill = self._tint if self._hover else self._bg
                fg = self._color
            else:
                fill, fg = self._bg, MD_OFF_FG
            if fill != self._bg:
                self.create_polygon(_round_rect(x0, y0, x1, y1, self._radius),
                                    fill=fill, outline="", width=0)
        total = self._iw + (10 if self._icon else 0) + self._tw
        sx = (w - total) / 2.0
        if self._icon:
            self._draw_icon(sx + self._iw / 2.0, h / 2.0, fg)
            sx += self._iw + 10
        self.create_text(sx, h / 2.0, text=self._text, fill=fg,
                         font=self._font, anchor="w")

    def _draw_icon(self, cx, cy, color):
        if self._icon == "play":
            self.create_polygon(cx - 6, cy - 8, cx + 8, cy, cx - 6, cy + 8,
                                fill=color, outline="")
        elif self._icon == "stop":
            self.create_rectangle(cx - 7, cy - 7, cx + 7, cy + 7, fill=color, outline="")
        elif self._icon == "restart":
            r = 8
            self.create_arc(cx - r, cy - r, cx + r, cy + r, start=50, extent=280,
                            style=tk.ARC, outline=color, width=2)
            self.create_polygon(cx + 1, cy - 10, cx + 10, cy - 5, cx + 2, cy - 1,
                                fill=color, outline="")
        else:
            self.create_text(cx, cy, text=self._icon, fill=color, font=("Segoe UI", 11))

    # ---- state ----
    def set_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled == self._enabled:
            return
        self._enabled = enabled
        self.configure(cursor="hand2" if enabled else "arrow")
        self._draw()

    def set_text(self, text):
        self._text = text
        self._tw = tkfont.Font(font=self._font).measure(text)
        self.configure(width=self._width())
        self._draw()

    # ---- mouse ----
    def _enter(self, event=None):
        self._hover = True
        self._draw()

    def _leave(self, event=None):
        self._hover = False
        self._press = False
        self._draw()

    def _down(self, event=None):
        if not self._enabled:
            return
        self._press = True
        self._draw()

    def _up(self, event=None):
        if not self._enabled:
            return
        was = self._press
        self._press = False
        self._draw()
        if was and self._cmd:
            self._cmd()


class MatChip(tk.Canvas):
    """A rounded status pill (Material chip) that shows the launcher state."""

    def __init__(self, parent, text="", bg=MD_TEXT2, fg="#FFFFFF", font=MD_FONT_B,
                 padx=14, height=30, canvas_bg=MD_BG, command=None, tooltip=None):
        self._text = text
        self._bg = bg
        self._fg = fg
        self._font = font
        self._padx = padx
        self._cmd = command
        super().__init__(parent, width=tkfont.Font(font=font).measure(text) + padx * 2,
                         height=height, bg=canvas_bg, highlightthickness=0, bd=0,
                         cursor="hand2" if command else "arrow")
        if command:
            self.bind("<Button-1>", lambda e: self._cmd())
        if tooltip:
            Tooltip(self, tooltip)
        self._draw()

    def set(self, text, bg=None):
        self._text = text
        if bg:
            self._bg = bg
        self.configure(width=tkfont.Font(font=self._font).measure(text) + self._padx * 2)
        self._draw()

    def _draw(self):
        self.delete("all")
        # The canvas is packed without fill, so the requested size is the real
        # one.  Using it (and not winfo_width, which lags one redraw behind)
        # is what keeps a longer label from being cut off.
        w = int(self["width"])
        h = int(self["height"])
        self.create_polygon(_round_rect(1, 1, w - 1, h - 1, (h - 2) / 2),
                            fill=self._bg, outline="")
        self.create_text(w / 2, h / 2, text=self._text, fill=self._fg, font=self._font)


class MatTab(tk.Canvas):
    """One tab of a Material tab bar: a label, with an underline when it is active.

    The ttk notebook draws its own tabs and ignores most of the theme, so the
    tab bar is drawn here instead - then it looks the same as the rest of the
    window on every Windows version.
    """

    def __init__(self, parent, text, command, bg=MD_CARD, font=MD_FONT, height=44):
        self._text = text
        self._cmd = command
        self._bg = bg
        self._font = font
        self._active = False
        self._hover = False
        self._h = height
        width = tkfont.Font(font=font).measure(text) + 36
        super().__init__(parent, width=width, height=height, bg=bg,
                         highlightthickness=0, bd=0, cursor="hand2")
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<Button-1>", self._click)
        self._draw()

    def set_active(self, active):
        active = bool(active)
        if active != self._active:
            self._active = active
            self._draw()

    def _draw(self):
        self.delete("all")
        w = int(self["width"])
        h = int(self["height"])
        if self._active:
            color = MD_PRIMARY
        elif self._hover:
            color = MD_TEXT
        else:
            color = MD_TEXT2
        font = MD_FONT_B if self._active else self._font
        self.create_text(w / 2, h / 2 - (2 if self._active else 0), text=self._text,
                         fill=color, font=font)
        if self._active:
            self.create_rectangle(14, h - 3, w - 14, h - 1, fill=MD_PRIMARY, outline="")

    def _enter(self, event=None):
        self._hover = True
        self._draw()

    def _leave(self, event=None):
        self._hover = False
        self._draw()

    def _click(self, event=None):
        if self._cmd:
            self._cmd()


class ScrollArea(tk.Frame):
    """A scrolling area for content that can be taller than the window.

    A bare Tk canvas scrolls into empty space as soon as its scroll region is
    missing or smaller than the window, because there is then nothing that
    confines the view.  So the region is built from the real content height,
    is never smaller than the window, and is refreshed whenever the content
    or the window changes.  The scrollbar only shows when there is something
    to scroll.
    """

    def __init__(self, parent, bg=MD_CARD):
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview,
                                  style="Vertical.TScrollbar")
        self.canvas.configure(yscrollcommand=self._scroll_set)
        self._bar_on = True
        self.vbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.sync())
        self.canvas.bind("<Configure>", self._resized)

    # ---- geometry ------------------------------------------------------
    def sync(self):
        """Make the scroll region match the content, and never be smaller
        than the window - that is what stops the scrolling into blank space."""
        try:
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            if w <= 1 or h <= 1:
                return                       # not laid out yet: try again later
            ih = self.inner.winfo_height()
            if ih <= 1:
                ih = self.inner.winfo_reqheight()
            self.canvas.configure(scrollregion=(0, 0, w, max(h, ih)))
        except Exception:
            pass

    def _resized(self, event):
        try:
            self.canvas.itemconfig(self._win, width=event.width)
        except Exception:
            pass
        self.sync()

    def _scroll_set(self, first, last):
        # the scrollbar only appears when the content does not fit
        needed = float(last) - float(first) < 0.999
        if needed != self._bar_on:
            self._bar_on = needed
            if needed:
                self.vbar.pack(side="right", fill="y")
            else:
                self.vbar.pack_forget()
        self.vbar.set(first, last)

    # ---- input ---------------------------------------------------------
    def scroll(self, units):
        self.canvas.yview_scroll(units, "units")

    def pointer_inside(self, event):
        """True when the mouse is over the scrollable area."""
        w = self.canvas.winfo_containing(event.x_root, event.y_root)
        while w is not None:
            if w is self.canvas:
                return True
            w = w.master
        return False


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Llama Launcher {APP_VERSION}")
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.minsize(700, 480)
        self.proc = None
        self.gen = 0
        # Administrator mode is what the full file-cache flush needs, so the
        # mode is read once, before the window is built.
        self.admin = is_admin()
        self.q = queue.Queue()
        self._reset_mem()
        self.mem_est = {}
        self.gpu_mem = None
        self._gpu_warned = False
        self.tuning = False
        self.starting = False   # the launcher is bringing a model up
        self.auto_preset = None
        self.auto_cancel = None
        self.auto_q = None
        self.run_auto = False        # the running server is subject to auto-tuning (OOM -> drop)
        self.run_preset = None
        self.run_ngl = None
        self.oom_pending = False
        # Opt-in: broad cleanup can cause other applications to page data back in.
        self.auto_clear = tk.BooleanVar(value=False)
        self.sampling_mode = tk.StringVar(value="Normal (publisher settings)")
        self.run_mode = None          # sampling mode the running server was started with
        self.port = tk.StringVar(value=DEFAULT_PORT)
        self.choice = tk.StringVar(value=NO_PRESET)
        self._build_ui()
        self._set_status("Stopped", MD_TEXT2)
        # First line of every run: which copy is this, and where does it look.
        self._log(f"[launcher] version {APP_VERSION} | file: {os.path.abspath(__file__)} | "
                  f"BASE_DIR: {BASE_DIR} | {len(PRESETS)} presets")
        ver = server_version()
        if ver:
            self._log(f"[launcher] llama.cpp: {ver}")
        others = _other_launcher_copies()
        if others:
            listed = ", ".join(f"{p} (version {v})" if v else f"{p} (older, no version marker)"
                               for p, v in others)
            self._log(f"[launcher] WARNING: other copies of this launcher are on disk: {listed}. "
                      f"They share the same auto_ngl.json and, being a different version, they would "
                      f"re-tune each other. Keep only the copy you are running.")
        for notice in STARTUP_NOTICES:
            self._log(notice)
        if not os.path.isfile(os.path.join(BASE_DIR, "llama-server.exe")):
            self._log(f"[presets] llama-server.exe is missing from {BASE_DIR} - every preset will fail to start.")
        for name, miss in MISSING_FILES.items():
            others = [n for n in miss if n != "llama-server.exe"]
            if others:
                self._log(f"[presets] missing in {BASE_DIR}: {', '.join(others)}   ->   {name}")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        threading.Thread(target=self._gpu_loop, daemon=True).start()
        self.after(100, self._drain)
        self.after(1000, self._tick)
        # The start-mode question comes up over the window, not instead of it:
        # a dialog that belongs to a hidden window is not shown reliably on
        # every Windows version, and then the launcher would look like it
        # starts without asking.
        self.after(80, self._ask_start_mode)

    def report_callback_exception(self, exc, val, tb):
        """Show an error instead of writing it to a console nobody sees.

        A .pyw has no console window, so an error in a button handler would
        otherwise disappear without a trace.
        """
        detail = "".join(traceback.format_exception(type(val), val, tb))
        path = write_error_log(detail)
        try:
            messagebox.showerror(
                "Llama Launcher",
                f"Something went wrong:\n\n{val}\n\n"
                f"The details were written to:\n{path}")
        except Exception:
            pass

    def _card(self, parent=None, bg=MD_CARD):
        """A Material card: flat surface, 1 px outline, no shadow."""
        return tk.Frame(parent or self, bg=bg, highlightbackground=MD_OUTLINE,
                        highlightthickness=1, bd=0)

    def _show_main_window(self):
        """Make sure the launcher is on screen.

        Used when the elevation fails: the launcher has to stay usable, so
        the window is brought back and the buttons are put in the right state.
        """
        try:
            if not self.winfo_ismapped():
                self.deiconify()
        except Exception:
            pass
        self._sync_buttons()

    def _ask_start_mode(self):
        """At startup: start normally, or as administrator.

        Administrator mode is what the full file-cache flush needs.  Asking
        once, up front, beats finding out later that a button does nothing.
        A launcher that is already running as administrator skips the question.

        The question is shown over the main window on purpose.  A dialog that
        belongs to a hidden window is not shown reliably on every Windows
        version, which is why the window stays visible here.
        """
        self._center_on_screen()
        if self.admin:
            self._sync_buttons()
            return
        mode = None
        try:
            dlg = tk.Toplevel(self)
            dlg.withdraw()
            dlg.title("Llama Launcher")
            dlg.configure(bg=MD_BG)
            dlg.resizable(False, False)
            dlg.transient(self)
            inner = tk.Frame(dlg, bg=MD_BG)
            inner.pack(fill="both", expand=True, padx=28, pady=24)
            tk.Label(inner, text="Llama Launcher", bg=MD_BG, fg=MD_TEXT,
                     font=("Segoe UI", 15, "bold")).pack(anchor="w")
            tk.Label(inner, text="How do you want to start?", bg=MD_BG, fg=MD_TEXT,
                     font=MD_FONT_B).pack(anchor="w", pady=(12, 6))
            tk.Label(inner, text=("Administrator mode lets the launcher clear the Windows file cache "
                                   "and free up RAM. Everything else works the same in both modes."),
                     bg=MD_BG, fg=MD_TEXT2, font=MD_FONT_S, justify="left", wraplength=400,
                     anchor="w").pack(anchor="w")
            row = tk.Frame(inner, bg=MD_BG)
            row.pack(fill="x", pady=(20, 0))
            MatButton(row, "Start normally", command=lambda: self._start_choice(dlg, "normal"),
                      color=MD_PRIMARY, color_dark=MD_PRIMARY_D, bg=MD_BG,
                      tooltip="Start without administrator rights. The file-cache flush stays "
                              "unavailable.").pack(side="left", padx=(0, 10))
            MatButton(row, "Start as administrator", command=lambda: self._start_choice(dlg, "admin"),
                      variant="outlined", color=MD_PRIMARY, color_dark=MD_PRIMARY_D,
                      tint=MD_PRIMARY_T, bg=MD_BG,
                      tooltip="Windows will ask you to confirm. Then the file-cache flush "
                              "works too.").pack(side="left")
            dlg.protocol("WM_DELETE_WINDOW", lambda: self._start_choice(dlg, None))
            self._start_dialog = dlg
            dlg.update_idletasks()
            self._center_dialog(dlg)
            dlg.deiconify()
            try:
                dlg.grab_set()
                dlg.focus_force()
            except Exception:
                pass
            # Insurance: if the question does not really get on screen, it is
            # closed and the launcher carries on.  A question that is on screen
            # is never touched, however long the user takes to answer it.
            self.after(6000, self._dialog_safety)
            self.wait_window(dlg)
            mode = getattr(self, "_start_mode", None)
        except Exception as e:
            write_error_log(traceback.format_exc())
            self._log(f"[launcher] the start-mode question could not be shown ({e}); "
                      f"starting as a standard user.")
            mode = None
        if mode == "admin":
            self._relaunch_admin(confirm=False)      # starts the elevated copy
            return
        self._sync_buttons()

    def _dialog_safety(self):
        """Bring the launcher back if the start-mode question never appeared."""
        dlg = getattr(self, "_start_dialog", None)
        if dlg is None:
            return
        try:
            if not dlg.winfo_exists():
                return                      # the question was answered already
            if dlg.winfo_ismapped():
                return                      # the question is on screen: leave it
        except Exception:
            return
        try:
            dlg.destroy()
        except Exception:
            pass
        self._start_mode = None
        self._log("[launcher] the start-mode question did not come up; starting as a standard user.")
        self._show_main_window()

    def _start_choice(self, dlg, mode):
        self._start_mode = mode
        try:
            # Deferred: the click is still being handled by a widget inside
            # this dialog, so the window is closed one step later.
            self.after_idle(dlg.destroy)
        except Exception:
            pass

    def _center_dialog(self, dlg):
        """Put the question in the middle of the launcher window."""
        try:
            dlg.update_idletasks()
            # Before the dialog is on screen winfo_width() is still 1, so the
            # requested size is the one to use.
            w = dlg.winfo_reqwidth()
            h = dlg.winfo_reqheight()
            x = self.winfo_rootx() + (self.winfo_width() - w) // 2
            y = self.winfo_rooty() + (self.winfo_height() - h) // 3
            dlg.geometry(f"+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass

    def _center_on_screen(self):
        """Put the launcher in the middle of the screen - position only."""
        try:
            self.update_idletasks()
            w = self.winfo_width() if self.winfo_width() > 1 else WIN_W
            h = self.winfo_height() if self.winfo_height() > 1 else WIN_H
            x = (self.winfo_screenwidth() - w) // 2
            y = (self.winfo_screenheight() - h) // 3
            self.geometry(f"+{max(0, x)}+{max(0, y)}")
        except Exception:
            pass

    def _build_ui(self):
        self.configure(bg=MD_BG)
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", font=MD_FONT, background=MD_BG, foreground=MD_TEXT)
        style.configure("TLabel", background=MD_BG, foreground=MD_TEXT)
        style.configure("Card.TLabel", background=MD_CARD, foreground=MD_TEXT)
        style.configure("CardDim.TLabel", background=MD_CARD, foreground=MD_TEXT2)
        style.configure("Dim.TLabel", background=MD_BG, foreground=MD_TEXT2)
        style.configure("TCombobox", fieldbackground=MD_CARD, background=MD_CARD,
                        foreground=MD_TEXT, bordercolor=MD_OUTLINE, lightcolor=MD_OUTLINE,
                        darkcolor=MD_OUTLINE, arrowsize=14, padding=(8, 7), font=MD_FONT)
        style.map("TCombobox", fieldbackground=[("readonly", MD_CARD)],
                  foreground=[("readonly", MD_TEXT)])
        style.configure("TEntry", fieldbackground=MD_CARD, foreground=MD_TEXT,
                        bordercolor=MD_OUTLINE, lightcolor=MD_OUTLINE,
                        darkcolor=MD_OUTLINE, insertcolor=MD_TEXT, padding=6, font=MD_FONT)
        style.configure("TCheckbutton", background=MD_CARD, foreground=MD_TEXT)
        style.map("TCheckbutton", background=[("active", MD_CARD)])
        style.configure("TProgressbar", background=MD_PRIMARY, troughcolor=MD_SURFACE,
                        bordercolor=MD_SURFACE, lightcolor=MD_PRIMARY, darkcolor=MD_PRIMARY)
        style.configure("TButton", background=MD_CARD, foreground=MD_TEXT,
                        bordercolor=MD_OUTLINE, lightcolor=MD_CARD, darkcolor=MD_CARD,
                        focuscolor=MD_CARD, padding=(14, 8), font=MD_FONT)
        style.map("TButton", background=[("active", MD_SURFACE), ("pressed", MD_OUTLINE)])
        style.configure("Treeview", background=MD_CARD, fieldbackground=MD_CARD,
                        foreground=MD_TEXT, borderwidth=0, rowheight=26)
        style.map("Treeview", background=[("selected", MD_PRIMARY_T)],
                  foreground=[("selected", MD_TEXT)])
        style.configure("Treeview.Heading", background=MD_SURFACE, foreground=MD_TEXT2,
                        font=MD_FONT_SB, relief="flat", borderwidth=0, padding=(6, 6))
        style.map("Treeview.Heading", background=[("active", MD_SURFACE)])
        style.configure("Vertical.TScrollbar", background="#C9CED6", troughcolor="#F1F2F4",
                        bordercolor="#C9CED6", darkcolor="#C9CED6", lightcolor="#C9CED6",
                        arrowcolor=MD_TEXT2, borderwidth=0, arrowsize=11)

        # ---- header: title on the left, state on the right ----------------
        head = tk.Frame(self, bg=MD_BG)
        head.pack(fill="x", padx=20, pady=(18, 10))
        tk.Label(head, text="Llama Launcher", bg=MD_BG, fg=MD_TEXT,
                 font=("Segoe UI", 18, "bold")).pack(side="left")
        tk.Label(head, text=f"v{APP_VERSION}   ·   {len(PRESETS)} presets   ·   {BASE_DIR}",
                 bg=MD_BG, fg=MD_TEXT2, font=MD_FONT_S, anchor="w").pack(side="left", padx=(12, 0), pady=(9, 0))
        self.status = MatChip(head, "Stopped", bg=MD_TEXT2, canvas_bg=MD_BG)
        self.status.pack(side="right")
        # The mode badge says up front what this launcher can and cannot do.
        # It is a label, not a button: the way to change the mode is the
        # "START AS ADMINISTRATOR" button, so that a badge that says
        # "Standard user" never does something surprising when it is clicked.
        if self.admin:
            self.mode_chip = MatChip(
                head, "Administrator", bg=MD_TEAL, canvas_bg=MD_BG,
                tooltip="Running as administrator: the Windows file cache can be cleared "
                        "and RAM can be flushed.")
        else:
            self.mode_chip = MatChip(
                head, "Standard user", bg=MD_TEXT2, canvas_bg=MD_BG,
                tooltip="Running as a standard user: the Windows file cache cannot be "
                        "cleared. Use START AS ADMINISTRATOR for that.")
        self.mode_chip.pack(side="right", padx=(0, 8))

        # ---- card 1: which model, on which port ---------------------------
        mcard = self._card()
        mcard.pack(fill="x", padx=20, pady=(0, 10))
        m_in = tk.Frame(mcard, bg=MD_CARD)
        m_in.pack(fill="x", padx=18, pady=16)
        row = tk.Frame(m_in, bg=MD_CARD)
        row.pack(fill="x")
        tk.Label(row, text="Model preset", bg=MD_CARD, fg=MD_TEXT, font=MD_FONT_B).pack(side="left")
        self.combo = ttk.Combobox(row, textvariable=self.choice, state="readonly",
                                  values=[NO_PRESET] + [p["name"] for p in PRESETS],
                                  height=14)
        self.combo.pack(side="left", padx=(14, 16), fill="x", expand=True)
        self.combo.bind("<<ComboboxSelected>>", lambda e: self._preset_changed())
        tk.Label(row, text="Port", bg=MD_CARD, fg=MD_TEXT, font=MD_FONT_B).pack(side="left")
        port = ttk.Entry(row, textvariable=self.port, width=8)
        port.pack(side="left", padx=(10, 0))
        port.bind("<KeyRelease>", lambda e: self._update_cmd())

        cmd_box = tk.Frame(m_in, bg=MD_SURFACE, highlightbackground=MD_OUTLINE,
                           highlightthickness=1, bd=0)
        cmd_box.pack(fill="x", pady=(14, 0))
        self.cmd_label = tk.Label(cmd_box, text="", bg=MD_SURFACE, fg=MD_TEXT2,
                                  font=MD_FONT_MONO, justify="left", anchor="w",
                                  wraplength=880, padx=14, pady=11)
        self.cmd_label.pack(fill="x")
        self.cmd_label.bind("<Configure>", self._fit_cmd_wrap)

        # ---- card 2: what to do -------------------------------------------
        acard = self._card()
        acard.pack(fill="x", padx=20, pady=(0, 10))
        a_in = tk.Frame(acard, bg=MD_CARD)
        a_in.pack(fill="x", padx=18, pady=16)
        btns = tk.Frame(a_in, bg=MD_CARD)
        btns.pack(fill="x")
        self.start_btn = MatButton(btns, "START", command=self.start, icon="play",
                                   color=MD_PRIMARY, color_dark=MD_PRIMARY_D,
                                   bg=MD_CARD, tooltip="Start the selected model preset.")
        self.start_btn.pack(side="left", padx=(0, 10))
        self.stop_btn = MatButton(btns, "STOP", command=self.stop, icon="stop",
                                  variant="outlined", color=MD_ERROR, color_dark=MD_ERROR_D,
                                  tint=MD_ERROR_T, bg=MD_CARD,
                                  tooltip="Stop the running model. During auto-tuning it cancels the tuning.")
        self.stop_btn.pack(side="left", padx=(0, 10))
        self.restart_btn = MatButton(btns, "RESTART", command=self._restart, icon="restart",
                                     variant="outlined", color=MD_PRIMARY, color_dark=MD_PRIMARY_D,
                                     tint=MD_PRIMARY_T, bg=MD_CARD,
                                     tooltip="Restart the running model with the settings on screen - "
                                             "use it after changing Sampling.")
        self.restart_btn.pack(side="left")
        tk.Frame(btns, bg=MD_CARD).pack(side="left", fill="x", expand=True)
        if self.admin:
            self.cache_btn = MatButton(btns, "CLEAR FILE CACHE", command=self.clear_ram,
                                       variant="text", color=MD_TEAL, color_dark=MD_TEAL_D,
                                       tint=MD_TEAL_T, bg=MD_CARD, font=MD_FONT,
                                       tooltip="Ask Windows to drop the file cache, so the model "
                                               "files are read from disk again.")
            self.cache_btn.pack(side="left", padx=(0, 6))
        else:
            # The flush needs administrator rights, so the button is not shown
            # at all - a button that does nothing is worse than no button.
            tk.Label(btns, text="File-cache flush needs administrator mode",
                     bg=MD_CARD, fg=MD_TEXT2, font=MD_FONT_S).pack(side="left", padx=(0, 12))
        self.ram_btn = MatButton(btns, "FREE RAM", command=self.open_free_ram,
                                 variant="text", color=MD_TEAL, color_dark=MD_TEAL_D,
                                 tint=MD_TEAL_T, bg=MD_CARD, font=MD_FONT,
                                 tooltip="List the apps using memory and let them give some back. "
                                         "This works in both modes.")
        self.ram_btn.pack(side="left")

        opts = tk.Frame(a_in, bg=MD_CARD)
        opts.pack(fill="x", pady=(16, 0))
        tk.Label(opts, text="Sampling", bg=MD_CARD, fg=MD_TEXT, font=MD_FONT_B).pack(side="left")
        self.sampling_box = ttk.Combobox(opts, textvariable=self.sampling_mode, state="readonly",
                                         values=list(SAMPLING_MODES), width=28)
        self.sampling_box.pack(side="left", padx=(14, 10))
        self.sampling_box.bind("<<ComboboxSelected>>", lambda e: self._sampling_changed())
        tk.Label(opts, text="Changes how the model writes. Presets stay publisher-official.",
                 bg=MD_CARD, fg=MD_TEXT2, font=MD_FONT_S).pack(side="left")

        opts2 = tk.Frame(a_in, bg=MD_CARD)
        opts2.pack(fill="x", pady=(10, 0))
        if self.admin:
            ttk.Checkbutton(opts2, text="Clear the Windows file cache automatically after Stop",
                            variable=self.auto_clear).pack(side="left")
        else:
            tk.Label(opts2, text="Automatic file-cache flush is available in administrator mode",
                     bg=MD_CARD, fg=MD_TEXT2, font=MD_FONT_S).pack(side="left")
        if not self.admin:
            MatButton(opts2, "START AS ADMINISTRATOR", command=self._relaunch_admin,
                      variant="text", color=MD_PRIMARY, color_dark=MD_PRIMARY_D,
                      tint=MD_PRIMARY_T, bg=MD_CARD, height=28, font=MD_FONT_SB,
                      tooltip="Close this window and open the launcher as administrator. "
                              "A running model is stopped for that.").pack(side="right")
        else:
            tk.Label(opts2, text="Running as administrator", bg=MD_CARD, fg=MD_OK,
                     font=MD_FONT_SB).pack(side="right")

        # ---- status line + log tools --------------------------------------
        stat = tk.Frame(self, bg=MD_BG)
        stat.pack(fill="x", padx=20, pady=(0, 8))
        self.ram_label = tk.Label(stat, text="", bg=MD_BG, fg=MD_TEXT2, font=MD_FONT_S)
        self.ram_label.pack(side="left")
        self.vram_label = tk.Label(stat, text="", bg=MD_BG, fg=MD_TEXT2, font=MD_FONT_S)
        self.vram_label.pack(side="left", padx=(16, 0))
        self.copy_btn = MatButton(stat, "COPY LOG", command=self._copy_log,
                                  variant="text", color=MD_PRIMARY, color_dark=MD_PRIMARY_D,
                                  tint=MD_PRIMARY_T, bg=MD_BG, height=28, font=MD_FONT_SB)
        self.copy_btn.pack(side="right")
        self.clearlog_btn = MatButton(stat, "CLEAR LOG", command=self._clear_log,
                                      variant="text", color=MD_TEXT2, color_dark=MD_TEXT,
                                      tint="#EDEFF2", bg=MD_BG, height=28, font=MD_FONT_SB)
        self.clearlog_btn.pack(side="right", padx=(0, 4))

        # ---- tabs: Log / Memory breakdown ---------------------------------
        tab_card = self._card()
        tab_card.pack(fill="both", expand=True, padx=20, pady=(0, 18))
        bar = tk.Frame(tab_card, bg=MD_CARD)
        bar.pack(fill="x", side="top")
        tk.Frame(tab_card, bg=MD_OUTLINE, height=1).pack(fill="x", side="top")
        content = tk.Frame(tab_card, bg=MD_CARD)
        content.pack(fill="both", expand=True)
        self._pages, self._tabs = {}, {}
        for key, label in (("log", "Log"), ("mem", "Memory breakdown")):
            page = tk.Frame(content, bg=MD_CARD)
            self._pages[key] = page
            tab = MatTab(bar, label, command=lambda k=key: self._show_tab(k))
            tab.pack(side="left", padx=(4, 0))
            self._tabs[key] = tab

        log_page = self._pages["log"]
        log_box = tk.Frame(log_page, bg=MD_SURFACE, highlightbackground=MD_OUTLINE,
                           highlightthickness=1, bd=0)
        log_box.pack(fill="both", expand=True, padx=16, pady=(4, 16))
        self.log = tk.Text(log_box, bg=MD_SURFACE, fg="#2B313A", font=("Consolas", 10),
                           wrap="word", insertbackground=MD_TEXT, relief="flat", bd=0,
                           highlightthickness=0, padx=16, pady=14, spacing1=1, spacing3=3)
        sb = ttk.Scrollbar(log_box, command=self.log.yview, style="Vertical.TScrollbar")
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

        self._build_mem_tab(self._pages["mem"])
        self._show_tab("log")
        self._update_cmd()

    def _show_tab(self, key):
        """Show one tab and hide the other."""
        for k, page in self._pages.items():
            if k == key:
                page.pack(fill="both", expand=True)
            else:
                page.pack_forget()
        for k, tab in self._tabs.items():
            tab.set_active(k == key)
        area = getattr(self, "mem_area", None)
        if area is not None:
            # the tab was hidden, so the canvas had no size: set the scroll
            # region again now that it is on screen.
            self.after(50, area.sync)

    def _clear_log(self):
        self.log.delete("1.0", "end")

    def _preset_changed(self):
        """A preset was picked (or cleared) in the dropdown."""
        self._update_cmd()
        self._sync_buttons()

    def _copy_log(self):
        self.clipboard_clear()
        self.clipboard_append(self.log.get("1.0", "end-1c"))
        self.update()
        self.copy_btn.set_text("Copied")
        self.after(1500, lambda: self.copy_btn.set_text("COPY LOG"))

    def _coding_hint(self, p):
        """Say what the launcher can do for code output, once per preset change.

        The coding rules themselves are not the launcher's business: the llama.cpp
        web UI sends them with the chat.  The sampler is.
        """
        if getattr(self, "_hint_preset", None) == p["name"]:
            return
        self._hint_preset = p["name"]
        if not p.get("coding"):
            return
        self._log("[coding] code preset selected. For code, switch the Sampling dropdown to "
                  "'Code (precise, no repeats)': low temperature, the publisher's repetition "
                  "penalty, and the DRY sampler that stops the model repeating the same line.")

    def _reset_mem(self):
        self.mem = {k: {} for k in ("model", "kv", "rs", "compute", "output")}
        self.mem_cache = None
        self.mem_fail = ""
        self.mem_ready = False
        self.mem_busy = False
        self.mem_base_cpu = None
        self.mem_printed_buffers = False   # does this llama.cpp build print buffer sizes at all?

    def _parse_mem(self, line):
        m = MEM_RE.search(line)
        if m:
            self.mem_printed_buffers = True
            dev, key, mib = m.group(1), m.group(2).lower(), float(m.group(3))
            d = self.mem[key]
            if key in ("model", "kv", "rs"):
                d[dev] = d.get(dev, 0) + mib
            else:
                d[dev] = mib
            return
        c = CACHE_RE.search(line)
        if c:
            self.mem_cache = float(c.group(2))
            return
        f = FAIL_MIB_RE.search(line)
        if f:
            # llama.cpp already reports MiB here - do not divide again.
            self.mem_fail = (f"Last failed allocation: {float(f.group(2)):,.0f} MiB on {f.group(1)} - out of "
                             f"memory. Lower -c, lower -b/-ub, or offload fewer layers (-ngl).")
            return
        fr = FAIL_RAW_RE.search(line.rstrip())
        if fr:
            self.mem_fail = (f"Last failed allocation: {int(fr.group(2)) / 1024**2:,.0f} MiB on {fr.group(1)} - out of "
                             f"memory. Lower -c, lower -b/-ub, or offload fewer layers (-ngl).")
            return
        fb = FAIL_BUF_RE.search(line)
        if fb:
            self.mem_fail = (f"Last failed allocation: {fb.group(1)} - out of memory. "
                             f"Lower -c, lower -b/-ub, or offload fewer layers (-ngl).")
            return
        fy = FAIL_BYTES_RE.search(line)
        if fy:
            self.mem_fail = (f"Last failed allocation: {int(fy.group(1)) / 1024**2:,.0f} MiB staging buffer - out of "
                             f"memory. Lower -c, lower -b/-ub, or offload fewer layers (-ngl).")

    def _build_mem_tab(self, parent):
        try:
            self._bar_font = tkfont.Font(font=("Segoe UI", 8, "bold"))
        except Exception:
            self._bar_font = None
        wrap = tk.Frame(parent, bg=MD_CARD)
        wrap.pack(fill="both", expand=True, padx=16, pady=(4, 16))
        self.mem_area = ScrollArea(wrap, bg=MD_CARD)
        self.mem_area.pack(fill="both", expand=True)
        inner = self.mem_area.inner

        def _mem_wheel(event):
            # Mouse wheel scrolls this tab only when the pointer is over it (labels included).
            if self.mem_area.pointer_inside(event):
                self.mem_area.scroll(-1 if event.delta > 0 else 1)
        self._mem_wheel = _mem_wheel
        self.bind_all("<MouseWheel>", _mem_wheel, add="+")

        self.mem_info = tk.Label(inner, text="", bg=MD_CARD, fg=MD_TEXT,
                                 font=MD_FONT_B, wraplength=820, justify="left", anchor="w")
        self.mem_info.pack(fill="x", pady=(0, 12))
        self.mem_canvas = tk.Canvas(inner, height=30, bg=MD_SURFACE, highlightthickness=1,
                                    highlightbackground=MD_OUTLINE)
        self.mem_canvas.pack(fill="x", pady=(0, 12))
        self.mem_canvas.bind("<Configure>", lambda e: self._refresh_mem())
        self.mem_note = tk.Label(inner, text="", bg=MD_CARD, fg=MD_ERROR,
                                 wraplength=820, justify="left", anchor="w")
        self.mem_note.pack(fill="x")
        grid = tk.Frame(inner, bg=MD_CARD)
        grid.pack(fill="x", pady=(12, 4))
        grid.columnconfigure(1, weight=0, minsize=150)
        grid.columnconfigure(2, weight=0, minsize=200)
        grid.columnconfigure(3, weight=1)
        self.mem_rows = {}
        for i, (key, name, color, desc) in enumerate(COMPONENTS):
            tk.Label(grid, bg=color, width=2).grid(row=i, column=0, padx=(0, 10), pady=5, sticky="nw")
            tk.Label(grid, text=name, bg=MD_CARD, fg=MD_TEXT, font=MD_FONT_B,
                     wraplength=150, justify="left", anchor="w").grid(row=i, column=1, sticky="nw", padx=(0, 16))
            size = tk.Label(grid, text="—", bg=MD_CARD, fg=MD_TEXT, font=MD_FONT,
                            wraplength=200, justify="left", anchor="w")
            size.grid(row=i, column=2, sticky="nw", padx=(0, 16))
            desc_label = tk.Label(grid, text=desc, bg=MD_CARD, fg=MD_TEXT2, font=MD_FONT_S,
                                  wraplength=380, justify="left", anchor="w")
            desc_label.grid(row=i, column=3, sticky="nw")
            self.mem_rows[key] = size
            if not hasattr(self, "mem_desc"):
                self.mem_desc = {}
            self.mem_desc[key] = desc_label
        tk.Frame(inner, bg=MD_OUTLINE, height=1).pack(fill="x", pady=(8, 10))
        self.mem_totals = tk.Label(inner, text="", bg=MD_CARD, fg=MD_TEXT,
                                   font=("Consolas", 10), justify="left", anchor="w")
        self.mem_totals.pack(anchor="w")
        self.mem_hint = tk.Label(inner, text="", bg=MD_CARD, fg=MD_TEXT2, font=MD_FONT_S,
                                 wraplength=820, justify="left", anchor="w")
        self.mem_hint.pack(fill="x", pady=(8, 0))
        self._refresh_mem()
        inner.bind("<Configure>", self._fit_mem_wrap)

    def _fit_cmd_wrap(self, event):
        """Wrap the command line to the window width instead of a fixed number."""
        try:
            w = max(200, event.width - 28)
            if w != getattr(self, "_cmd_wrap_w", None):
                self._cmd_wrap_w = w
                self.cmd_label.configure(wraplength=w)
        except Exception:
            pass

    def _fit_mem_wrap(self, event):
        """Re-flow every wrapped text in the memory tab when the window is resized."""
        try:
            w = event.width
            if w == getattr(self, "_mem_wrap_w", None):
                return
            self._mem_wrap_w = w
            full = max(240, w - 10)      # labels that span the whole tab
            desc = max(240, w - 420)     # description column: tab minus the fixed columns
            for lbl in getattr(self, "mem_desc", {}).values():
                lbl.configure(wraplength=desc)
            self.mem_info.configure(wraplength=full)
            self.mem_note.configure(wraplength=full)
            self.mem_hint.configure(wraplength=full)
            self.mem_totals.configure(wraplength=full)
        except Exception:
            pass

    def _load_model_info(self):
        """Weights = file size (read first). KV and layer info = GGUF header (may fail on its own)."""
        est = {}
        try:
            cmd = self._build_cmd()

            def val(flag, default=None):
                return cmd[cmd.index(flag) + 1] if flag in cmd else default

            model_arg = val("-m")
            est["for"] = model_arg                      # which model this info belongs to
            est["file"] = _base(model_arg)
            mpath = os.path.join(BASE_DIR, est["file"])
            est["model_mib"] = os.path.getsize(mpath) / 1024**2
            mm = val("--mmproj")
            if mm:
                est["mmproj_mib"] = os.path.getsize(os.path.join(BASE_DIR, _base(mm))) / 1024**2
        except Exception as e:
            est["error"] = f"{type(e).__name__}: {e} (looking in {BASE_DIR})"
            self.mem_est = est
            return
        try:
            info = read_gguf_info(mpath)
            est["info"] = info
            n_ctx = int(val("-c", 0))
            if n_ctx:
                est["kv_mib"] = estimate_kv_mib(info, n_ctx, val("-ctk", "f16"), val("-ctv", "f16"))
        except Exception as e:
            est["header_error"] = f"{type(e).__name__}: {e}"
        self.mem_est = est

    def _gpu_loop(self):
        # FIX 1: guarded typeperf parsing, log once on failure, never crash silently.
        while True:
            res = None
            p = self.proc
            if p and p.poll() is None:
                try:
                    pid = p.pid
                    out = subprocess.run(
                        ["typeperf", "-sc", "1",
                         rf"\GPU Process Memory(pid_{pid}*)\Dedicated Usage",
                         rf"\GPU Process Memory(pid_{pid}*)\Shared Usage"],
                        capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW).stdout
                    rows = [r for r in csv.reader(out.splitlines()) if r]
                    hi = None
                    for i, r in enumerate(rows):
                        if r and r[0].startswith("(PDH-CSV"):
                            hi = i
                            break
                    if hi is None or hi + 1 >= len(rows):
                        raise ValueError("typeperf returned no data rows")
                    ded = sh = 0.0
                    for h, v in zip(rows[hi][1:], rows[hi + 1][1:]):
                        try:
                            x = float(v)
                        except ValueError:
                            continue
                        if "Dedicated Usage" in h:
                            ded += x
                        elif "Shared Usage" in h:
                            sh += x
                    res = (ded / 1024**2, sh / 1024**2)
                    self._gpu_warned = False
                except Exception as e:
                    res = None
                    if not self._gpu_warned:
                        self._gpu_warned = True
                        try:
                            self.q.put((self.gen, "line", f"[gpu] typeperf failed, GPU RAM will show blank: {e}"))
                        except Exception:
                            pass
            self.gpu_mem = res
            time.sleep(1)

    def _model_desc_text(self, est, cmd):
        file_part = f"{est['file']}: {est['model_mib'] / 1024:.2f} GB on disk"
        if "mmproj_mib" in est:
            file_part += f" + picture part {est['mmproj_mib'] / 1024:.2f} GB"
        ngl = None
        try:
            if cmd and "-ngl" in cmd:
                ngl = int(cmd[cmd.index("-ngl") + 1])
        except (ValueError, IndexError):
            ngl = None
        info = est.get("info", {}) or {}
        arch = info.get("arch") or ""
        layers = info.get(f"{arch}.block_count") if arch else None
        if not isinstance(layers, int):
            layers = None
        running = bool(getattr(self, "proc", None) and self.proc.poll() is None)
        split = dict(getattr(self, "mem", {}).get("model", {}) or {})
        total = sum(split.values()) if split else 0.0
        meas = "Measured" if running else "Measured (last run)"
        if total > 0:
            cpu = sum(v for k, v in split.items() if str(k).upper().startswith("CPU"))
            gpu = total - cpu
            gpu_pct = 100.0 * gpu / total
            if gpu_pct >= 99:
                return f"{file_part}. {meas}. All on graphics chip."
            if gpu_pct <= 0:
                return f"{file_part}. {meas}. All on processor. It will be slow."
            if ngl is not None and layers is not None and ngl >= layers + 1:
                return (f"{file_part}. {meas}. {gpu_pct:.0f}% on graphics, {100.0 - gpu_pct:.0f}% on processor. "
                        f"Preset asks for all parts on graphics. So this much on processor means graphics ran short.")
            if ngl is not None and layers is not None:
                return (f"{file_part}. {meas}. {gpu_pct:.0f}% on graphics, {100.0 - gpu_pct:.0f}% on processor. "
                        f"Preset puts -ngl {ngl} of {layers} parts on graphics. Rest on processor is normal.")
            return f"{file_part}. {meas}. {gpu_pct:.0f}% on graphics, {100.0 - gpu_pct:.0f}% on processor."
        if running:
            return f"{file_part}. Running. Waiting for real numbers."
        if ngl is not None and layers is not None:
            return f"{file_part}. Not started. Preset puts -ngl {ngl} of {layers} parts on graphics. Start to see real numbers."
        return f"{file_part}. Not started. Start to see real numbers."

    def _kv_desc_text(self, est, cmd):
        n_ctx = None
        try:
            if cmd and "-c" in cmd:
                n_ctx = int(cmd[cmd.index("-c") + 1])
        except (ValueError, IndexError):
            n_ctx = None
        actual = sum((self.mem.get("kv", {}) or {}).values())
        info = est.get("info", {}) or {}
        arch = info.get("arch") or ""
        is_hybrid = info.get(f"{arch}.full_attention_interval") is not None if arch else False
        base = "Memory for this chat. It remembers what you said."
        if n_ctx:
            base += f" Preset sets -c {n_ctx}, about {int(n_ctx * 0.75):,} words."
        if actual > 0:
            base += f" Measured {actual:,.0f} MB now. It grows as the chat grows."
        elif est.get("kv_mib"):
            base += f" About {est['kv_mib']:,.0f} MB when the chat is full. It grows as the chat grows."
        if is_hybrid:
            base += " This model uses it in only some parts, so this is at most the real size."
        return base

    def _rs_desc_text(self, est):
        info = est.get("info", {}) or {}
        arch = info.get("arch") or ""
        is_hybrid = False
        if arch:
            is_hybrid = any(str(k).startswith(f"{arch}.ssm.") or k == f"{arch}.full_attention_interval" for k in info)
        if not is_hybrid:
            return "Not used by this model. Only some models need it."
        actual = sum((self.mem.get("rs", {}) or {}).values())
        base = "Fixed amount. It does not grow when the chat grows."
        if actual > 0:
            base += f" Measured {actual:,.0f} MB."
        else:
            base += " Not shown separately. It is inside the graphics total."
        return base

    def _compute_desc_text(self, cmd):
        b = ub = None
        try:
            if cmd and "-b" in cmd:
                b = int(cmd[cmd.index("-b") + 1])
            if cmd and "-ub" in cmd:
                ub = int(cmd[cmd.index("-ub") + 1])
        except (ValueError, IndexError):
            pass
        actual = sum((self.mem.get("compute", {}) or {}).values())
        base = "Rough work for reading your question."
        if b is not None and ub is not None:
            base += f" Preset sets -b {b}: it reads {b} word pieces at one time."
            base += f" Preset sets -ub {ub}: it works in groups of {ub}."
        base += " Bigger is faster but needs more memory. It is biggest when the question is long."
        if actual > 0:
            base += f" Measured about {actual:,.0f} MB."
        return base

    def _output_desc_text(self):
        actual = sum((self.mem.get("output", {}) or {}).values())
        base = ("Marks for the next word. The model gives a mark to every word and picks the one with most marks. "
                "Very small, under 1 MB. It is automatic. You do not set it.")
        if actual > 0:
            base += f" Measured {actual:.0f} MB."
        return base

    def _prompt_cache_desc_text(self, cache_ram):
        return (f"Saved work from old questions. If you ask the same thing again, it uses the saved work, so it is faster. "
                f"Preset allows {cache_ram} MB for this. When it is full, old saved work is removed. "
                f"So memory grows while you chat.")

    def _other_desc_text(self):
        return "Memory the app needs for itself. It is not for the model and not for the chat. It stays the same after start."

    def _refresh_mem(self):
        if self._selected() is None:
            # No preset picked yet: nothing to measure.
            self.mem_info.config(text="Select a model preset to see its weights and memory estimate.")
            self.mem_note.config(text="")
            self.mem_totals.config(text="")
            self.mem_hint.config(text="")
            for key, label in self.mem_rows.items():
                label.config(text="—")
            c = self.mem_canvas
            c.delete("all")
            w = max(c.winfo_width(), 100)
            ch = c.winfo_height() if c.winfo_height() > 1 else 30
            c.create_text(w / 2, ch / 2, text="Pick a preset to see its weights and KV estimate",
                          fill=MD_TEXT2)
            return
        mem = self.mem
        try:
            cmd = self._build_cmd()
            cache_ram = int(cmd[cmd.index("--cache-ram") + 1]) if "--cache-ram" in cmd else 2048
        except Exception:
            cmd, cache_ram = [], 2048
        want = cmd[cmd.index("-m") + 1] if "-m" in cmd else None
        if want and self.mem_est.get("for") != want:    # info is missing or belongs to another model: reload
            self._load_model_info()
        est = self.mem_est
        if "model_mib" in est:
            if hasattr(self, "mem_desc") and "model" in self.mem_desc:
                self.mem_desc["model"].config(text=self._model_desc_text(est, cmd))
        if hasattr(self, "mem_desc"):
            wiring = {
                "kv": lambda: self._kv_desc_text(est, cmd),
                "rs": lambda: self._rs_desc_text(est),
                "compute": lambda: self._compute_desc_text(cmd),
                "output": lambda: self._output_desc_text(),
                "cache": lambda: self._prompt_cache_desc_text(cache_ram),
                "other": lambda: self._other_desc_text(),
            }
            for key, make in wiring.items():
                if key in self.mem_desc:
                    self.mem_desc[key].config(text=make())
        if "model_mib" not in est:
            sel = self.choice.get() if hasattr(self, "choice") else ""
            if MISSING_FILES.get(sel):
                self.mem_info.config(text=f"⚠ MISSING in {BASE_DIR}: {', '.join(MISSING_FILES[sel])} - "
                                          f"this preset will fail to start.")
            else:
                self.mem_info.config(text=f"Model file not readable: {est.get('error', 'unknown error')}")
        else:
            head = f"{est['file']}: {est['model_mib'] / 1024:.2f} GB on disk"
            if "mmproj_mib" in est:
                head += f"  + mmproj {est['mmproj_mib'] / 1024:.2f} GB"
            info = est.get("info", {})
            arch = info.get("arch")
            bits = []
            if est.get("header_error"):
                bits.append(f"GGUF header not readable: {est['header_error']}")
            if arch:
                bits.append(f"arch {arch}")
            if info.get("general.size_label"):
                bits.append(str(info["general.size_label"]))
            if info.get(f"{arch}.block_count"):
                bits.append(f"{info[f'{arch}.block_count']} layers")
            if isinstance(info.get(f"{arch}.context_length"), int):
                bits.append(f"trained ctx {info[f'{arch}.context_length']:,}")
            try:
                preset_ctx = int(cmd[cmd.index("-c") + 1]) if "-c" in cmd else None
                trained = info.get(f"{arch}.context_length") if arch else None
                if preset_ctx and isinstance(trained, int):
                    if preset_ctx > trained:
                        bits.append(f"preset -c {preset_ctx:,} is above the trained {trained:,}: quality may drop past it")
                    else:
                        bits.append(f"preset uses -c {preset_ctx:,} of the trained {trained:,}")
            except (ValueError, IndexError):
                pass
            # FIX 4: show missing-file warning in header, don't hide preset
            sel = self.choice.get() if hasattr(self, "choice") else ""
            if sel in MISSING_FILES and MISSING_FILES[sel]:
                head += f"\n⚠ MISSING in {BASE_DIR}: {', '.join(MISSING_FILES[sel])} - this preset will fail to start."
            self.mem_info.config(text=head + ("\n" + "  |  ".join(bits) if bits else ""))
        best, text = {}, {}
        for key in ("model", "kv", "rs", "compute", "output"):
            if mem[key]:
                best[key] = sum(mem[key].values())
                text[key] = "\n".join(f"{dev}: {fmt_mib(v)}" for dev, v in mem[key].items())
        if "model" not in best and "model_mib" in est:
            v = est["model_mib"] + est.get("mmproj_mib", 0)
            best["model"] = v
            text["model"] = fmt_mib(v) + "\n(file size" + (" incl. mmproj)" if "mmproj_mib" in est else ")")
        if "kv" not in best and est.get("kv_mib"):
            best["kv"] = est["kv_mib"]
            text["kv"] = fmt_mib(est["kv_mib"]) + "\n(estimate)"
        if self.mem_cache is not None:
            best["cache"] = self.mem_cache
            text["cache"] = fmt_mib(self.mem_cache)
        running = bool(self.proc and self.proc.poll() is None)
        pm = process_memory(self.proc.pid) if running else None
        ws = pm[0] / 1024**2 if pm else 0.0
        priv = pm[1] / 1024**2 if pm else 0.0
        gpu = self.gpu_mem if running else None
        gsh = gpu[1] if gpu and priv >= gpu[1] else 0.0
        gtot = gpu[0] + gpu[1] if gpu else None
        cpu_side = max(0.0, priv - gsh)
        if pm and gpu and self.mem_ready and not self.mem_busy and self.mem_base_cpu is None:
            self.mem_base_cpu = cpu_side                   # baseline: loaded, GPU counters known, no request yet
        notes = []
        if self.mem_fail:
            notes.append(self.mem_fail)
        if pm and gtot is not None and "compute" not in best and "model" in best:
            left = gtot - best.get("model", 0) - best.get("kv", 0)
            if left >= 0:
                best["compute"] = left
                text["compute"] = f"≈ {fmt_mib(left)}\n(GPU - weights - KV)"
            else:
                notes.append(f"GPU memory ({fmt_mib(gtot)}) is smaller than weights + KV: part of the model is "
                             f"probably running on the CPU.")
        if pm and "cache" not in best and self.mem_base_cpu is not None:
            g = max(0.0, cpu_side - self.mem_base_cpu)
            best["cache"] = g
            text["cache"] = f"{fmt_mib(g)}\n(RAM growth)"
        if pm:
            host = self.mem_base_cpu if self.mem_base_cpu is not None else cpu_side
            best["other"] = host
            text["other"] = fmt_mib(host) + ("\n(host RAM right after load)" if self.mem_base_cpu is not None else "")
        info = est.get("info", {})
        arch = info.get("arch")
        hybrid = any(str(k).startswith(f"{arch}.ssm.") or k == f"{arch}.full_attention_interval" for k in info) if arch else False
        dflt = "Not measured yet" if running else "Shows when the server is running"
        text.setdefault("rs", ("Included in GPU total" if hybrid else "Not used"))
        text.setdefault("output", "< 1 MiB, negligible")
        if running and not getattr(self, "mem_printed_buffers", False):
            # No buffer-size line at all: this build does not print them, so waiting
            # for them is pointless.  Say that instead of "not measured yet".
            text.setdefault("compute", "Not reported\n(by this build)")
        for k in ("compute", "cache", "other"):
            text.setdefault(k, dflt)
        for key, _, _, _ in COMPONENTS:
            self.mem_rows[key].config(text=text.get(key, "—"))
        cpu_w = sum(v for dev, v in mem["model"].items() if dev.upper().startswith("CPU"))
        if cpu_w > 500:
            notes.append(f"{fmt_mib(cpu_w)} of model weights are on the CPU - generation will be slow.")
        self.mem_note.config(text="  ".join(notes))
        def m(v):
            return f"{v:,.0f} MiB"
        lines = [f"Weights + conversation memory : {m(best.get('model', 0) + best.get('kv', 0))}"]
        if pm:
            if gpu:
                lines.append(f"GPU memory dedicated / shared: {m(gpu[0])} / {m(gpu[1])}  "
                             f"(on an integrated GPU both come from system RAM)")
            else:
                lines.append("GPU memory                 : not available")
            lines.append(f"Process RAM held by server : {m(priv)}   (working set {m(ws)})")
            lines.append(f"Process RAM not on GPU     : {m(cpu_side)}")
        else:
            lines.append("Server memory             : not running")
        self.mem_totals.config(text="\n".join(lines))
        if not running:
            hint = ("Server not running: weights show the file size, and conversation memory is an estimate "
                    "for the selected preset. Start the server to see real numbers.")
        elif not any(mem.values()):
            hint = ("This build did not print buffer sizes. Weights are the file size. Chat memory is an estimate. "
                    "Rough work is GPU memory minus weights and chat memory. Prompt cache is RAM growth since load.")
        else:
            hint = "Numbers from the log override the file-size and estimate values. Process and GPU numbers are live."
        self.mem_hint.config(text=hint)
        c = self.mem_canvas
        c.delete("all")
        w = max(c.winfo_width(), 100)
        ch = c.winfo_height() if c.winfo_height() > 1 else 30
        order = ("model", "kv", "rs", "compute", "output", "cache", "other")
        total = sum(best.get(k, 0) for k in order)
        if total <= 0:
            c.create_text(w / 2, ch / 2, text="Pick a preset to see its weights and KV estimate",
                          fill=MD_TEXT2)
            return
        colors = {k: col for k, _, col, _ in COMPONENTS}
        x = 1
        for k in order:
            v = best.get(k, 0)
            if v <= 0:
                continue
            seg = (w - 2) * v / total
            c.create_rectangle(x, 1, x + seg, ch - 1, fill=colors[k], outline="white")
            # Only write the name if it really fits. A long name in a narrow
            # segment used to be cut off mid-word ("Compute buffe").
            label = BAR_NAMES.get(k, k)
            if self._bar_fits(label, seg):
                c.create_text(x + seg / 2, ch / 2, text=label, fill="white",
                              font=("Segoe UI", 8, "bold"))
            x += seg

    def _bar_fits(self, label, seg):
        """True if the text fits inside a segment of that width."""
        try:
            need = self._bar_font.measure(label) + 8
        except Exception:
            need = len(label) * 7          # rough fallback
        return need <= seg

    # ---------------- AUTO NGL (driven by START) ----------------
    def _begin_tuning(self, p, reason="no saved value yet"):
        try:
            port = int(self.port.get().strip() or DEFAULT_PORT)
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid port", "Enter a TCP port number from 1 to 65535.")
            return
        self.tuning = True
        self.auto_preset = p
        self.auto_cancel = threading.Event()
        self.auto_q = queue.Queue()
        self._set_run_state("tuning")
        self._set_status("Auto-tuning NGL…", MD_WARN)
        self._log(f"\n=== Auto-tuning NGL for {p['name']} (port {port}): {reason} ===")
        self._log("[auto ngl] each trial loads the model; big models take several minutes")
        threading.Thread(target=auto_tune_ngl,
                         args=(p, port, self.auto_q, self.auto_cancel, self._sampler_args(p)),
                         daemon=True).start()
        self.after(200, self._poll_auto)

    def _poll_auto(self):
        finished = False
        try:
            while True:
                item = self.auto_q.get_nowait()
                if item[0] == "log":
                    self._log(item[1])
                    continue
                finished = True
                self._auto_finish(item)
                break
        except queue.Empty:
            pass
        if self.tuning and not finished:
            self.after(200, self._poll_auto)

    def _auto_finish(self, item):
        self.tuning = False
        kind = item[0]
        p = self.auto_preset
        if kind == "done":
            ngl, meta = item[1], item[2]
            saved = self._save_record(p, ngl, meta=meta)
            ceiling = (meta or {}).get("ceiling_ngl")
            spare = (f" (the machine could hold -ngl {ceiling}, one layer is kept in reserve)"
                     if isinstance(ceiling, int) and ceiling > ngl else "")
            self._log(f"[auto ngl] RESULT: -ngl {ngl} loads and answers a long prompt{spare} ({saved}). Starting…")
            self.choice.set(p["name"])          # START must use the preset that was tuned
            self._update_cmd()
            # Straight from "tuning" into "starting".  Going through "stopped"
            # here would put the buttons in the stopped look for the moment
            # the RAM wait repaints the window, and then flip them again when
            # the server appears.
            self.starting = True
            self._set_status("Starting…", MD_WARN)
            self._sync_buttons()
            self.start()                        # the saved value is used from here on
            return
        if kind == "failed":
            self._log(f"[auto ngl] FAILED: {item[1]}")
            messagebox.showerror("Auto-tuning NGL failed", item[1])
        else:
            self._log("[auto ngl] cancelled; nothing saved")
        self._set_status("Stopped", MD_TEXT2)
        self._set_run_state("stopped")
        self._update_cmd()

    def _save_record(self, p, ngl, meta=None, note=None):
        key = _auto_key(p)
        model_name = _base(p["args"][p["args"].index("-m") + 1])
        a = p["args"]
        ctx_value = a[a.index("-c") + 1] if "-c" in a else None
        mm_value = _base(a[a.index("--mmproj") + 1]) if "--mmproj" in a else None
        cache_value = a[a.index("--cache-ram") + 1] if "--cache-ram" in a else None
        live_keys = set()
        for q in PRESETS:
            qk = _auto_key(q)
            if qk:
                live_keys.add(qk)
        for k in list(AUTO_CACHE.keys()):
            if AUTO_CACHE[k].get("model") == model_name and AUTO_CACHE[k].get("ctx") == ctx_value \
                    and AUTO_CACHE[k].get("mmproj") == mm_value and k != key and k not in live_keys:
                del AUTO_CACHE[k]
        rec = {"model": model_name, "ngl": ngl, "ctx": ctx_value, "mmproj": mm_value,
               "sampling": self.sampling_mode.get(),
               "tested_at": time.strftime("%Y-%m-%d %H:%M"), **(meta or {})}
        if "cache_ram" not in rec:
            rec["cache_ram"] = cache_value
        # The baseline is the free RAM the saved value was measured with, with no model
        # running.  For a tuned record the reading from the start of the tune is kept,
        # but the reading at save time is the one that matches the next START, so the
        # larger of the two is used: a high baseline only means "do not re-tune yet",
        # while a low one would trigger a re-tune that is not needed.
        cur_free = _free_ram_reading(mode="max")
        tuned_free = rec.get("free_ram_mib_at_tune")
        if not isinstance(tuned_free, (int, float)):
            tuned_free = cur_free
        elif isinstance(cur_free, (int, float)):
            tuned_free = max(tuned_free, cur_free)
        rec["free_ram_mib_at_tune"] = tuned_free
        if "layers" not in rec:
            rec["layers"] = _auto_layers(p)
        # One automatic re-tune per hour at most.  Without this, a value that was just
        # measured can claim that there is even more RAM free and start a second, third
        # ... re-tune.  Each one loads the model several times and takes minutes.
        rec["retune_lock_until"] = time.time() + (AUTO_RETUNE_COOLDOWN_S if meta
                                                   else AUTO_STEPDOWN_COOLDOWN_S)
        if note:
            rec["note"] = note
        AUTO_CACHE[key] = rec
        try:
            save_auto_ngl(AUTO_CACHE)
            return f"saved to {AUTO_NGL_FILE}"
        except Exception as e:
            return f"NOT saved ({e}); used for this session only"

    def _schedule_oom(self, gen):
        if self.oom_pending:
            return
        self.oom_pending = True
        self.after(50, lambda: self._on_oom(gen))

    def _on_oom(self, gen):
        """An out-of-memory line came from the server that START launched: drop to the next lower value."""
        self.oom_pending = False
        if gen != self.gen or not self.run_auto or self.tuning:
            return
        p, cur = self.run_preset, self.run_ngl
        if cur is None or cur <= 0:
            lower = []
        else:
            lower = [cur - 1]
        self._stop_process()
        self.gen += 1
        self.run_auto = False
        if not lower:
            self._log(f"[auto ngl] out of memory at -ngl {cur}, and nothing lower is left. "
                      f"Lower the context or free RAM, then START again.")
            self._set_status("Stopped", MD_TEXT2)
            self._set_run_state("stopped")
            return
        new = lower[0]
        # Wait for the dead server's memory to come back before the free RAM is read.
        # Reading it too early gives a number that is far too low, and the next START
        # then reads a normal number and thinks the machine got a lot faster.
        self._wait_ram()
        saved = self._save_record(p, new, note=f"lowered from {cur} after OOM in real use")
        self._log(f"[auto ngl] OOM at -ngl {cur} in real use -> saved -ngl {new} ({saved}); restarting")
        self.choice.set(p["name"])
        self._update_cmd()
        self.start()

    def _selected(self):
        """The chosen preset, or None while the dropdown still shows the placeholder."""
        name = self.choice.get()
        for p in PRESETS:
            if p["name"] == name:
                return p
        return None

    def _sampler_args(self, p):
        """The preset's args with the chosen sampling mode applied."""
        return apply_sampling_mode(p["args"], self.sampling_mode.get())

    def _sampler_pending(self):
        """True when the running server was started with different sampling settings."""
        try:
            if not (self.proc and self.proc.poll() is None):
                return False
        except Exception:
            return False
        run_flags = SAMPLING_MODES.get(getattr(self, "run_mode", None))
        cur_flags = SAMPLING_MODES.get(self.sampling_mode.get())
        return run_flags != cur_flags

    def _sampling_changed(self):
        """The user changed the sampling mode.

        The running server keeps the settings it was started with, so ask
        before restarting it: a restart cancels an answer in progress.
        """
        pending = self._sampler_pending()
        if pending and not self.tuning:
            if messagebox.askyesno(
                    "Restart to apply the new settings?",
                    "The running server was started with the old settings.\n\n"
                    "Restart it now? Any answer in progress will be cancelled."):
                self._restart()          # restart with the new settings
            else:
                self._log("[sampling] changed. Press RESTART when you are ready to apply it.")
        self._update_cmd()

    def _build_cmd(self):
        p = self._selected()
        if p is None:
            return []
        cmd = [p["exe"]] + self._sampler_args(p)
        if p["exe"] == "llama-server.exe":
            cmd += ["--port", self.port.get().strip() or DEFAULT_PORT]
            cmd += EXTRA_SERVER_ARGS
            rec = _auto_record(p)               # a saved AUTO NGL value overrides the preset's ngl
            if rec and "-ngl" in cmd:
                cmd[cmd.index("-ngl") + 1] = str(rec["ngl"])
        return cmd

    def _update_cmd(self):
        p = self._selected()
        if p is None:
            self.cmd_label.config(text="Pick a model preset above, then press START.")
            self._refresh_mem()
            self._sync_buttons()
            return
        text = " ".join(self._build_cmd())
        rec = _auto_record(p)
        if rec:
            text += f"\n[auto ngl {rec['ngl']} (saved {rec.get('tested_at', '?')}); changes to the model file or ctx re-tune]"
            if rec.get("note"):
                text += f"\n[{rec['note']}]"
        elif _auto_applies(p) and _auto_key(p):
            text += "\n[auto ngl: no saved value yet; START will tune it first]"
        if self._sampler_pending():
            text += "\n[sampling changed - the running server still uses the old settings, press RESTART]"
        self.cmd_label.config(text=text)
        self._coding_hint(p)
        self._refresh_mem()         # Memory tab: loads the file size + GGUF header for the selected model

    def _set_status(self, text, color):
        self.status.set(text, color)

    def _sync_buttons(self):
        """The action buttons follow what the launcher is actually doing.

        START needs a chosen preset and nothing running.
        STOP is only alive while a model runs, while auto-tuning is in progress
        or while the launcher is bringing a model up - there is nothing to stop
        otherwise.
        RESTART only makes sense while a model runs: it stops it and starts it
        again with the settings that are on screen now.

        The "starting" flag is what keeps the buttons still while the launcher
        moves from auto-tuning to a running server: without it the buttons
        would go to the stopped look, and then flip over again when the
        server process appears.
        """
        if not hasattr(self, "start_btn"):
            return
        running = bool(self.proc and self.proc.poll() is None)
        tuning = bool(self.tuning)
        starting = bool(getattr(self, "starting", False))
        has_preset = self._selected() is not None
        self.start_btn.set_enabled(has_preset and not running and not tuning and not starting)
        self.stop_btn.set_enabled(running or tuning or starting)
        self.restart_btn.set_enabled(running and not tuning)

    def _set_run_state(self, state):
        # The state itself is shown by the status chip; the buttons are
        # switched on and off here.
        self._sync_buttons()

    def _log(self, text):
        self.log.insert("end", text if text.endswith("\n") else text + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 3000:
            self.log.delete("1.0", "500.0")
        self.log.see("end")

    def _restart(self):
        """Restart the running model with the settings that are on screen now.

        This is the button that makes a changed Sampling mode take effect:
        the running server keeps the settings it was started with, and the
        restart replaces them.  An answer in progress is cancelled.
        """
        if self._selected() is None:
            if self.run_preset is not None:
                self.choice.set(self.run_preset["name"])
                self._update_cmd()
            else:
                messagebox.showinfo("No preset selected",
                                    "Pick a model preset in the dropdown first.")
                return
        # The old server goes away first, so a restart that has to auto-tune
        # does not tune next to a model that is still running.
        self.run_auto = False
        self.gen += 1
        self.starting = True
        self._stop_process()
        self._set_status("Restarting…", MD_WARN)
        self._sync_buttons()
        self.start()

    def start(self):
        if self.tuning:
            messagebox.showinfo("Auto-tuning NGL", "Wait for auto-tuning to finish, or press STOP to cancel it.")
            return
        p = self._selected()
        if p is None:
            messagebox.showinfo("No preset selected",
                                "Pick a model preset in the dropdown first, then press START.")
            return
        rec = _auto_record(p)
        mismatch = _sampling_mismatch(rec, self.sampling_mode.get())
        if _auto_applies(p) and _auto_key(p) is not None:
            if rec is None:
                self._begin_tuning(p)               # first run for this model/ctx/KV: tune, then start
                return
            if mismatch:
                self._log(f"[auto ngl] saved -ngl {rec['ngl']} was tuned with different sampling "
                          f"settings, retuning for the new ones")
                self._begin_tuning(p, reason=f"saved -ngl {rec['ngl']} was tuned with other sampling, retuning")
                return
            if _auto_is_stale(p, rec):
                self._log(f"[auto ngl] saved -ngl {rec['ngl']} is stale: free RAM dropped since tuning, retuning")
                self._begin_tuning(p, reason=f"saved -ngl {rec['ngl']} is stale, retuning")
                return
            if _auto_is_pessimistic(p, rec):
                self._log(f"[auto ngl] saved -ngl {rec['ngl']} was tuned with less RAM free than now. "
                          f"More GPU layers may fit, retuning.")
                self._begin_tuning(p, reason=f"more free RAM than at tuning time, retuning for a higher -ngl")
                return
        cmd = self._build_cmd()
        # Every way out of this block has to undo the "starting" flag, or the
        # buttons would stay frozen in the starting look for ever.  The finally
        # clause does that for every early return.
        try:
            port_value = self.port.get().strip() or DEFAULT_PORT
            try:
                port_number = int(port_value)
                if not 1 <= port_number <= 65535:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid port", "Enter a TCP port number from 1 to 65535.")
                return
            exe_path = os.path.join(BASE_DIR, cmd[0])
            if not os.path.isfile(exe_path):
                messagebox.showerror("Not found", f"{exe_path} does not exist.\nCheck BASE_DIR in the script.")
                return
            # FIX 4: explicit missing-file check at Start, since we no longer hide presets
            sel = self.choice.get()
            if sel in MISSING_FILES and MISSING_FILES[sel]:
                messagebox.showerror("Missing file", f"Preset '{sel}' is missing in {BASE_DIR}:\n{', '.join(MISSING_FILES[sel])}")
                return
            model = cmd[cmd.index("-m") + 1]
            if not os.path.isfile(os.path.join(BASE_DIR, os.path.basename(model))):
                messagebox.showerror("Model not found", f"{os.path.join(BASE_DIR, model)} does not exist.")
                return
            # From here on the launcher is busy.  The flag keeps the buttons in
            # the "starting" look while the old server is stopped and the RAM of
            # the dead one comes back - that wait repaints the window, and without
            # the flag the buttons would flip to the stopped look and back again.
            self.starting = True
            self._sync_buttons()
            self._stop_process()
            self._wait_ram()
            # After the old server is gone and its memory has come back, so the
            # reading is what the new server actually gets.
            self._ram_warning(p)
            self.gen += 1
            gen = self.gen
            auto = _auto_applies(p) and _auto_key(p) is not None
            self.run_auto = auto
            self.run_preset = p
            self.run_ngl = int(cmd[cmd.index("-ngl") + 1]) if auto and "-ngl" in cmd else None
            self.oom_pending = False
            self._reset_mem()
            self._log(f"\n=== START: {' '.join(cmd)} ===")
            try:
                self.proc = subprocess.Popen(
                    [exe_path] + cmd[1:], cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
            except Exception as e:
                messagebox.showerror("Failed to start", str(e))
                self._set_status("Stopped", MD_TEXT2)
                self._set_run_state("stopped")
                return
            # Remember what the running server was started with, so a later change
            # to the sampling can be reported as "not applied yet".
            self.run_mode = self.sampling_mode.get()
            # The server process exists now, so the buttons follow the process.
            self.starting = False
            self._set_status("Starting…", MD_WARN)
            self._set_run_state("starting")
            threading.Thread(target=self._reader, args=(self.proc, gen), daemon=True).start()
        finally:
            # Nothing is running (we bailed out, or the server never came up):
            # the starting look has to go, or the buttons stay frozen.
            if not (self.proc and self.proc.poll() is None):
                self.starting = False
                self._sync_buttons()

    def _reader(self, proc, gen):
        for line in proc.stdout:
            self.q.put((gen, "line", line))
        proc.wait()
        self.q.put((gen, "exit", proc.returncode))

    def _ram_warning(self, p):
        """Warn only when the model cannot fit in free RAM at all.

        Auto NGL can lower the GPU layers, but the weights and the chat memory
        are always there, so those are the fixed cost. The prompt cache is a cap
        that fills up as you chat, not an upfront allocation, so it is not
        counted here. Being wrong in the direction of "warns anyway" makes the
        warning useless, so it stays quiet unless the model really cannot fit.
        """
        try:
            free = _free_ram_mib()
            if free is None:
                return
            a = p["args"]

            def val(flag, default=None):
                return a[a.index(flag) + 1] if flag in a else default

            rel = val("-m")
            if not rel:
                return
            path = os.path.join(BASE_DIR, os.path.basename(rel))
            if not os.path.isfile(path):
                return
            need = os.path.getsize(path) / 1024 ** 2          # the weights, always resident
            try:
                info = read_gguf_info(path)
                kv = estimate_kv_mib(info, int(val("-c", 4096) or 4096),
                                     val("-ctk", "f16"), val("-ctv", "f16")) or 0
            except Exception:
                kv = 0
            need += kv + 512                                  # chat memory + a little overhead
            if free < need:
                cache = int(val("--cache-ram", 2048) or 0)
                extra = (f" Its prompt cache can take up to {cache:,.0f} MiB more as you chat."
                         if cache else "")
                self._log(f"[start] only {free:,.0f} MiB of RAM is free, but this model needs about "
                          f"{need:,.0f} MiB for the weights and the chat memory alone, before any GPU "
                          f"layers.{extra}")
                self._log("[start] The server will still start, but most of the model will run on the "
                          "processor, so it will be slow. What actually helps:")
                hints = []
                lighter = _lighter_preset_for(p)
                if lighter:
                    hints.append(f"preset '{lighter[0]}' runs the same model with a smaller context "
                                 f"({lighter[1]:,} tokens)")
                if cache >= 1024:
                    hints.append(f"a lower --cache-ram ({cache} -> 512) frees up to {cache - 512:,.0f} MiB "
                                 f"of host RAM for the weights")
                hints.append("closing apps: the file cache flush on its own frees very little when the "
                             "RAM is held by open programs, not by the cache")
                for h in hints:
                    self._log(f"[start]   - {h}")
        except Exception:
            pass

    def _wait_ram(self):
        """Wait until the freed memory of the previous server is really back.

        The Vulkan shared-memory pool of a killed llama-server is released slowly.
        Starting the next one too early makes it fail on allocations of a few KB,
        which looks like a much lower limit than it really is.
        """
        try:
            self._set_status("Waiting for RAM…", MD_WARN)
            self.update_idletasks()
        except Exception:
            pass
        _wait_for_ram_settle()

    def _stop_process(self):
        # FIX 5: only kill OUR process, never any other llama-server.exe on the machine.
        if self.proc and self.proc.poll() is None:
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                               capture_output=True, creationflags=NO_WINDOW)
            except Exception:
                pass
        self.proc = None

    def stop(self):
        if self.tuning:
            self.auto_cancel.set()
            kill_all_llama()          # the trial server dies; the worker then reports "cancelled"
            self._log("[auto ngl] stop requested")
            return
        self.starting = False
        self.run_auto = False
        self.gen += 1
        self._stop_process()
        self._set_status("Stopped", MD_TEXT2)
        self._set_run_state("stopped")
        self._log("=== STOPPED ===")
        if self.auto_clear.get():
            self.after(600, self.clear_ram)

    def clear_ram(self):
        # Only request a standby/file-cache flush here. Other apps' working sets
        # can be trimmed individually in the Free RAM window after explicit
        # selection - that part does not need administrator rights.
        if not self.admin:
            self._log("[file cache] not administrator - Windows only lets an administrator "
                      "flush the file cache. Start the launcher as administrator for that.")
            return
        before = ram_info()[0]
        ok, msg = clear_standby_ram()
        time.sleep(0.3)
        after = ram_info()[0]
        self._log(f"[file cache] {msg}  (RAM used {before:.1f} GB -> {after:.1f} GB)")

    def open_free_ram(self):
        """The "Free RAM" window: which app holds the memory, and let it go.

        The old name for this was "Cleaner", which said nothing about what it
        cleaned.  What it does is trim the working set of the apps you pick, so
        they hand their memory back to Windows.
        """
        if getattr(self, "cleaner", None) and self.cleaner.winfo_exists():
            self.cleaner.lift()
            return
        w = tk.Toplevel(self)
        w.title("Free RAM")
        w.geometry("600x600")
        w.configure(bg=MD_BG)
        self.cleaner = w
        head = tk.Frame(w, bg=MD_BG)
        head.pack(fill="x", padx=18, pady=(16, 0))
        tk.Label(head, text="Free RAM", bg=MD_BG, fg=MD_TEXT,
                 font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(head, text="Pick the apps that are holding memory and give some back. "
                            "Windows puts the memory back when the app needs it again.",
                 bg=MD_BG, fg=MD_TEXT2, font=MD_FONT_S, justify="left", wraplength=560,
                 anchor="w").pack(anchor="w", pady=(4, 0))
        tk.Label(head, text="Apps that run as administrator cannot be trimmed from a standard "
                            "user session - those rows are skipped.",
                 bg=MD_BG, fg=MD_TEXT2, font=MD_FONT_S, justify="left", wraplength=560,
                 anchor="w").pack(anchor="w", pady=(4, 0))
        usage = tk.Label(w, text="", bg=MD_BG, fg=MD_TEXT, font=MD_FONT_B, anchor="w")
        usage.pack(fill="x", padx=18, pady=(14, 4))
        bar = ttk.Progressbar(w, maximum=100)
        bar.pack(fill="x", padx=18)
        frame = tk.Frame(w, bg=MD_CARD, highlightbackground=MD_OUTLINE, highlightthickness=1)
        frame.pack(fill="both", expand=True, padx=18, pady=12)
        tree = ttk.Treeview(frame, columns=("name", "mem", "n"), show="headings", selectmode="extended")
        tree.heading("name", text="Application")
        tree.heading("mem", text="Memory")
        tree.heading("n", text="Procs")
        tree.column("name", width=300)
        tree.column("mem", width=110, anchor="e")
        tree.column("n", width=60, anchor="e")
        sb = ttk.Scrollbar(frame, command=tree.yview, style="Vertical.TScrollbar")
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        tree.pack(side="left", fill="both", expand=True)
        info = tk.Label(w, text="Ctrl/Shift-click to select several apps.",
                        bg=MD_BG, fg=MD_TEXT2, font=MD_FONT_S, anchor="w")
        info.pack(anchor="w", padx=18)
        def refresh():
            used, total, pct = ram_info()
            usage.config(text=f"Memory usage {pct}%   ({used:.1f} GB / {total:.1f} GB)")
            bar["value"] = pct
            tree.delete(*tree.get_children())
            groups = list_processes()
            for name, g in sorted(groups.items(), key=lambda kv: kv[1]["ws"], reverse=True):
                tree.insert("", "end", iid=name, values=(name, f"{g['ws'] / 1024**2:,.0f} MB", len(g["pids"])))
            self._groups = groups
        def clean(names):
            if not names:
                messagebox.showinfo("Free RAM", "Select at least one app.", parent=w)
                return
            pids = [pid for n in names for pid in self._groups[n]["pids"]]
            released, failed = trim_pids(pids)
            note = f"\n({failed} processes were protected and skipped)" if failed else ""
            messagebox.showinfo("Memory freed", f"Released {released / 1024**2:,.0f} MB of memory.{note}", parent=w)
            refresh()
        def clean_selected():
            names = list(tree.selection())
            if any(n.lower() in PROCESS_NAMES for n in names) and self.proc and self.proc.poll() is None:
                if not messagebox.askyesno("Model is running", "Trimming a running llama-server forces its memory to be re-read and slows generation. Continue?", parent=w):
                    return
            clean(names)
        row = tk.Frame(w, bg=MD_BG)
        row.pack(fill="x", padx=18, pady=(0, 16))
        ttk.Button(row, text="Refresh", command=refresh).pack(side="left")
        ttk.Button(row, text="Select all", command=lambda: tree.selection_set(tree.get_children())).pack(side="left", padx=6)
        if self.admin:
            ttk.Button(row, text="Flush Windows cache",
                       command=lambda: (self._log("[cache] " + clear_standby_ram()[1]), refresh())).pack(side="left")
        else:
            ttk.Button(row, text="Flush Windows cache (needs administrator)",
                       state="disabled").pack(side="left")
        MatButton(row, "FREE SELECTED RAM", command=clean_selected,
                  color=MD_PRIMARY, color_dark=MD_PRIMARY_D, bg=MD_BG, font=MD_FONT,
                  tooltip="Trim the memory of the selected apps, so Windows can use it "
                          "for the model.").pack(side="right")
        refresh()

    def _relaunch_admin(self, confirm=True):
        """Close this window and open the launcher as administrator.

        The running model has to go down for that, so the user is told before
        it happens, and the current launcher closes only once the new one has
        actually been started.
        """
        if confirm:
            running = bool(self.proc and self.proc.poll() is None)
            if self.tuning:
                what = ("The auto-NGL tuning is still running. It will be cancelled and the "
                        "value found so far will not be saved.")
            elif running:
                name = self.run_preset["name"] if self.run_preset else "the model"
                what = (f"The running model ({name}) will be stopped, and any answer in "
                        f"progress is cancelled.")
            else:
                what = "No model is running, so nothing is interrupted."
            if not messagebox.askyesno(
                    "Start as administrator?",
                    f"This launcher will close and open again as administrator.\n\n{what}\n\n"
                    "Windows will ask you to confirm."):
                return
        # Nothing should be left running when the window goes away.
        if self.tuning and self.auto_cancel is not None:
            self.auto_cancel.set()
            kill_all_llama()
            self.tuning = False
        self.starting = False
        self.run_auto = False
        self.gen += 1
        self._stop_process()
        try:
            shell32 = ctypes.windll.shell32
            # The return value is an HINSTANCE, which is 64 bit on a 64 bit
            # Windows.  Without this, ctypes cuts it down to 32 bit and the
            # "greater than 32" test below gives the wrong answer.
            shell32.ShellExecuteW.restype = ctypes.c_void_p
            rc = shell32.ShellExecuteW(
                None, "runas", sys.executable, f'"{os.path.abspath(__file__)}"', None, 1)
        except Exception as e:
            write_error_log(traceback.format_exc())
            messagebox.showerror("Could not start as administrator", str(e))
            self._show_main_window()
            return
        if rc is None or rc <= 32:
            # ShellExecute returns <= 32 when the request was refused, which
            # is what happens when the UAC prompt is cancelled.
            messagebox.showerror("Could not start as administrator",
                                  "Windows did not start the elevated copy. The UAC prompt "
                                  "was probably cancelled.")
            self._show_main_window()
            return
        self.destroy()

    def _on_close(self):
        self._stop_process()
        self.destroy()

    def _drain(self):
        try:
            while True:
                gen, kind, data = self.q.get_nowait()
                if gen != self.gen:
                    continue
                if kind == "line":
                    self._log(data.rstrip("\n"))
                    self._parse_mem(data)
                    if self.run_auto and _is_oom_line(data):
                        self._schedule_oom(gen)
                    low = data.lower()
                    if "listening" in low or "main loop" in low:
                        self.mem_ready = True
                        # FIX 6: capture baseline HERE, at Ready, not in the 1s tick.
                        # If first request already started, baseline is still the current CPU-side RAM.
                        try:
                            if (self.proc and self.proc.poll() is None and self.mem_base_cpu is None
                                    and self.gpu_mem is not None):
                                pm = process_memory(self.proc.pid)
                                if pm:
                                    priv = pm[1] / 1024**2
                                    gpu = self.gpu_mem
                                    gsh = gpu[1] if gpu and priv >= gpu[1] else 0.0
                                    self.mem_base_cpu = max(0.0, priv - gsh)
                        except Exception:
                            pass
                        self._set_status(f"Ready on port {self.port.get()}", MD_OK)
                        self._set_run_state("running")
                    if "processing task" in low:
                        self.mem_busy = True
                elif kind == "exit":
                    if (self.run_auto and not self.tuning and _is_crash_exit(data)):
                        # The server was killed by the graphics driver and printed no
                        # out-of-memory line, so the OOM detector could not see it.
                        # The crash code is the tell: treat it as too many GPU layers.
                        self._log(f"[auto ngl] the server crashed (exit code {data}) without an "
                                  f"out-of-memory message. Treating it as too many GPU layers.")
                        self._on_oom(gen)
                        return
                    self._log(f"=== process exited (code {data}) ===")
                    self.starting = False
                    self._set_status("Stopped", MD_TEXT2)
                    self._set_run_state("stopped")
                    self.proc = None
        except queue.Empty:
            pass
        self.after(100, self._drain)

    def _tick(self):
        try:
            used, total, pct = ram_info()
        except Exception:
            used, total, pct = 0.0, 0.0, 0
        self.ram_label.config(text=f"RAM: {used:.1f} / {total:.1f} GB ({pct}%)",
                              fg=MD_ERROR if pct >= 90 else MD_TEXT2)
        gpu = self.gpu_mem
        self.vram_label.config(text=f"GPU RAM: {gpu[0] + gpu[1]:,.0f} MiB (from system RAM)" if gpu else "")
        try:
            self._refresh_mem()
        except Exception:
            pass
        self.after(1000, self._tick)

if __name__ == "__main__":
    try:
        App().mainloop()
    except Exception:
        # The launcher must never die silently: show what went wrong, and
        # keep a copy of the traceback next to the script.
        detail = traceback.format_exc()
        path = write_error_log(detail)
        text = ("The launcher could not start.\n\n"
                + detail.strip().splitlines()[-1]
                + f"\n\nThe full details were written to:\n{path}")
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Llama Launcher", text, parent=root)
            root.destroy()
        except Exception:
            pass
