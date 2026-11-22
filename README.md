# Llama Launcher

Run local AI on a normal 16 GB laptop. No cloud. No API key. No subscription.

This launcher starts llama-server.exe on your own laptop. Your prompts stay on your laptop. It works without internet.

## Screenshots

[List of models in Launcher](screenshot-models.png)
[Model loaded in launcher](screenshot-loaded.png)
[Model running log](screenshot-running.png)
[PC memory usage breakdown](screenshot-memory.png)
[Localhost chat screen](screenshot-chat.png)
[LLM response on localhost chat](screenshot-response.png)
[Log screen in Launcher](screenshot-log.png)

## What is inside this repo

- `llama_launcher.pyw` - The launcher. Double click to run.
- `README.md` - This file.
- `LICENSE` - MIT License.
- `.gitignore` - Stops GGUF files and auto NGL files from getting uploaded.

GGUF model files are NOT in this repo. You have to download them yourself. They are too big for GitHub.

## What this is for

- Basic coding. Write small functions, explain code, find bugs in short snippets.
- Chatting and learning. Ask for an explanation in simple words, ask for examples, draft an email.
- General information. History, science, maths, how things work. The model already knows this from training.

It is good for decent, daily work. It is not perfect. Check important code and facts once.

## What this is not for, by itself

It is not for current information. A local model only knows what it learned up to its training date. It does not know today's news, today's price, or a new library version. If you ask it, it will guess.

## For current information, use Exa MCP

In the llama web UI, you can add an MCP server for web search. One free option is Exa.

- Server: https://mcp.exa.ai/mcp
- Free tier: 150 calls per day, 3 queries per second, no API key needed to start. Check Exa pricing for the current limit.
- How it works: for a current question, the model searches the web first, then answers from the search results.

Without MCP: fully offline, no limit, no current info.
With Exa MCP: current info works, up to 150 free calls per day.

## Tested on

- ThinkBook 14 G9 AHP, model code 21V0
- Ryzen 7 250, 8 cores, 16 threads
- Radeon 780M, integrated, shared memory, no dedicated VRAM
- 16 GB DDR5-5600 RAM
- Windows 11, Python 3, llama.cpp with Vulkan
- Folder: C:\prism-llama
- Usual free RAM while running: 7 to 9 GB

## Quick start

1. Install Python 3 on Windows. Tick Add python.exe to PATH. No pip packages needed.
2. Download a Windows llama.cpp release with Vulkan support. Copy llama-server.exe into C:\prism-llama.
3. Download GGUF files you want. Put each .gguf and each mmproj file in C:\prism-llama.
4. Double click llama_launcher.pyw, pick a preset, press START. Open http://127.0.0.1:8080

## Models already in the launcher

These presets are already in `llama_launcher.pyw`. You only need to download the GGUF file. Put it in `C:\prism-llama`.

| Preset | File you need to download | Use |
| --- | --- | --- |
| Bonsai 2 27B Thinking, Low memory, Instruct | Ternary-Bonsai-2-27B-PQ2_0.gguf or PTQ1_0.gguf or Q2_0 file | Thinking, instruct |
| Bonsai 2 9B Thinking, Instruct, Images | Ternary-Bonsai-2-9B-PQ2_0.gguf or PTQ1_0.gguf or Q2_0 file | Thinking, instruct, images |
| Gemma 4 E4B Reasoning OFF and ON | gemma-4-E4B_q4_0-it.gguf | General chat |
| Gemma 4 E4B Vision | gemma-4-E4B_q4_0-it.gguf + mmproj file | Chat with images |
| Qwen3 8B Thinking, Non-Thinking | Qwen3-8B-Q4_K_M.gguf | General chat |
| Llama 3.1 8B | Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf | General chat |
| Qwen2.5-VL 7B Vision | Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf + mmproj file | Chat with images |
| Ultravox audio | Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf + mmproj file | Audio, experimental |
| Qwen3-ASR 1.7B | Qwen3-ASR-1.7B-Q8_0.gguf + mmproj file | Speech to text |
| SmolVLM2 500M Video | SmolVLM2-500M-Video-Instruct-Q8_0.gguf + mmproj file | Video |
| Qwen2.5-Coder 7B Q6_K | Qwen2.5-Coder-7B-Instruct-Q6_K.gguf | Coding |
| Qwen2.5-Coder 14B Q4_K_M | qwen2.5-coder-14b-instruct-q4_k_m.gguf | Coding, partial GPU |
| MiMo-V2.6-Distill-Qwen-9B Q4_K_M and Q8_0 | MiMo-V2.6-Distill-Qwen-9B-Q4_K_M-OFFICIAL.gguf or Q8_0.gguf | Agentic |
| Qwen3.5-9B | Qwen3.5-9B-Q4_K_M-OFFICIAL.gguf | General, non-thinking |
| Nemotron 3 Nano 4B | Nemotron3-Nano-4B-Q4_K_M.gguf | Reasoning |
| Nemotron-Nano-9B-v2 | Nemotron-Nano-9B-v2-Q4_K_M-OFFICIAL.gguf | Reasoning |
| DeepSeek-R1-Distill-Qwen-7B | DeepSeek-R1-Distill-Qwen-7B-Q4_K_M-OFFICIAL.gguf | Reasoning |
| Ministral 3 8B Instruct | Ministral-3-8B-Instruct-2512-BF16-Q4_K_M-OFFICIAL.gguf | General |

If a preset says (no mmproj: text-only), it will still run as text only. Add the mmproj file to get images or audio.

## If your model is not in this list

You need to add one line in the launcher. Downloading a model is not enough.

1. Open `llama_launcher.pyw` in VS Code.
2. Find `PRESETS = [`.
3. Copy one similar line and change it. Example for a new Qwen model:

```python
preset("My model | GENERAL", "My-Model-Q4_K_M.gguf", ctx(8192) + KV + QWEN_SAMPLING + REASONING_OFF),
```

4. Change only 3 things: the name in quotes, the file name, and the `ctx()` number if you want more or less context.
5. Save the file. Restart the launcher. Your new preset will show in the dropdown.

If you download a different quant of the same model, for example Q8_0 instead of Q4_K_M, change only the file name in that line. If you download a totally new model, copy the closest preset and change the file name and sampler. The sampler values come from the publisher page for that model.

## Tested models, official presets, Auto NGL

All the models in the list above are models I have tested myself on this 16 GB laptop. The launcher contains the full information for each model: file name, context size, KV cache, sampler, and GPU layers.

All presets are official presets. That means the settings are taken from the official publisher page and the official ggml-org file, not guessed. If the publisher says temperature 0.6 and top-p 0.95, the launcher uses 0.6 and 0.95. If the publisher says a model needs a mmproj file for images, the launcher asks for that mmproj file. You do not need to search for settings.

### Auto NGL is already set up

NGL means how many layers of the model go to the GPU. More layers on GPU = faster. Too many layers = out of memory and crash.

You do not need to set NGL by hand. The launcher does it for you:

1. On first run for a preset, it tries different NGL values, one by one.
2. It checks how much GPU memory is free and if the model loads without error.
3. It picks the highest NGL that fits safely and saves it in `auto_ngl.json`.
4. Next time you start the same preset, it uses the saved NGL directly. No re-test.

If you change your GPU driver, or you want to test again, just delete `auto_ngl.json` and start the launcher again. It will auto-tune again. If auto-tune fails, it falls back to a safe low NGL so the model still starts on CPU.

## Can I reuse this as is?

Yes, if you are on Windows 10 or 11 and you keep the same layout. Change only BASE_DIR at the top of the file if your folder is different. Download your own models. GGUF files are not in this repo. Do not copy auto_ngl.json. It is recreated on first run.

## Does it work on Intel iGPU?

Yes. It is not specific to Radeon 780M. It uses Windows GPU Process Memory and the llama.cpp log. Intel Iris Xe, UHD, and Arc all appear as Vulkan0 in the log, same as Radeon 780M. Use a Vulkan build of llama-server.exe and install the Intel Vulkan driver. Linux and macOS are not supported.

## License

MIT
