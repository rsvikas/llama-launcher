# Llama Launcher

Run local AI on a normal 16 GB laptop. No cloud. No API key. No subscription.

This launcher starts llama-server.exe on your own laptop. Your prompts stay on your laptop. It works without internet.

Repository: https://github.com/rsvikas/llama-launcher-for-16gb-ram

## Screenshots

[List of models in Launcher](screenshot-models.png)
[Model loaded in launcher](screenshot-loaded.png)
[Model running log](screenshot-log.png)
[PC memory usage breakdown](screenshot-memory.png)
[Localhost chat screen](screenshot-chat.png)
[LLM response on localhost chat](screenshot-response.png)
[Log screen in Launcher](screenshot-running.png)

## Demo Video

Watch the launcher running a model locally on a 16GB laptop:

[Watch on YouTube - Llama Launcher running locally](https://youtu.be/cI4puBlCPvg)

[Watch on YouTube - Second demo](https://youtu.be/HWK1HUrWzf0)

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

The first line of the log tells you which copy of the launcher you are running, where that copy is, and which folder it reads models from. The window title carries the same version number. If you keep more than one copy, this line tells you which one is live.

## What the launcher checks when it starts

- The version line gives the version, the full path of the file you are running, the folder it reads models from, and how many presets it has. The same version is in the window title.
- If another copy of the launcher is on disk and it is a **different version**, it says so and names that version. A copy of the same version is harmless and is not reported. Copies of different versions share the same `auto_ngl.json` but compute different keys, so they keep re-tuning each other and overwriting the saved values. Keep one copy.
- If there is not enough free RAM for the weights and the chat memory of the model you picked, it says so before the server starts, instead of letting it run out of memory halfway through a request.

## Models already in the launcher

These presets are already in `llama_launcher.pyw`. You download the GGUF files yourself and put them in `C:\prism-llama`.

| Preset | File you need to download | Use |
| --- | --- | --- |
| Bonsai 2 27B Thinking, Low memory, Instruct | Ternary-Bonsai-2-27B-Q2_0.gguf only | Thinking, instruct, stock llama.cpp only, no PrismML fork |
| Bonsai 2 9B Thinking, Instruct | Ternary-Bonsai-2-9B-Q2_0.gguf only | Thinking, instruct, stock llama.cpp only, no PrismML fork |
| Bonsai 2 9B Images | Ternary-Bonsai-2-9B-Q2_0.gguf + Ternary-Bonsai-2-9B-mmproj-Q8_0.gguf | Images, thinking, chat only, not Cline |
| Gemma 4 E4B Reasoning OFF and ON | gemma-4-E4B_q4_0-it.gguf | General chat |
| Gemma 4 E4B Vision | gemma-4-E4B_q4_0-it.gguf + mmproj file | Chat with images |
| Qwen3 8B Thinking, Non-Thinking | Qwen3-8B-Q4_K_M.gguf | General chat |
| Llama 3.1 8B | Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf | General chat |
| Qwen2.5-VL 7B Vision | Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf + mmproj file | Chat with images |
| Ultravox audio | Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf + mmproj file | Audio, experimental |
| Qwen3-ASR 1.7B | Qwen3-ASR-1.7B-Q8_0.gguf + mmproj file | Speech to text |
| SmolVLM2 500M Video | SmolVLM2-500M-Video-Instruct-Q8_0.gguf + mmproj file | Video |
| Qwen2.5-Coder 7B Q6_K | Qwen2.5-Coder-7B-Instruct-Q6_K.gguf | Coding |
| Qwen2.5-Coder 14B Q4_K_M | Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf | Coding, partial GPU |
| MiMo-V2.6-Distill-Qwen-9B Q4_K_M and Q8_0 | MiMo-V2.6-Distill-Qwen-9B-Q4_K_M-OFFICIAL.gguf or MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf | Agentic |
| Qwen3.5-9B | Qwen3.5-9B-Q4_K_M-OFFICIAL.gguf | General, non-thinking |
| Nemotron 3 Nano 4B | NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf | Reasoning |
| Nemotron-Nano-9B-v2 | Nemotron-Nano-9B-v2-Q4_K_M-OFFICIAL.gguf | Reasoning |
| DeepSeek-R1-Distill-Qwen-7B | DeepSeek-R1-Distill-Qwen-7B-Q4_K_M-OFFICIAL.gguf | Reasoning |
| Ministral 3 8B Instruct | Ministral-3-8B-Instruct-2512-BF16-Q4_K_M-OFFICIAL.gguf | General |
| OLMoE 1B-7B Q6_K | OLMoE-Q6_K.gguf | MoE chat, 64 experts, 8 active |
| Qwen1.5-MoE-A2.7B-Chat Q3_K_M | Qwen-MoE-Q3_K_M.gguf | MoE chat, 60 experts, 4 active, ctx 2048 (4096 needs ~9 GB free) |

Notes:
- Bonsai 2 runs on stock llama.cpp only. Only the Q2_0 pack runs here. There is no PrismML fork.
- The image, video and audio presets need their mmproj file as well as the model file. Without the projector the preset still runs, but as text only, and it says "(no mmproj: text-only)" in the dropdown.
- Ultravox uses the same weights file as Llama 3.1. The projector is what makes it Ultravox.
- OLMoE uses ctx 4096, which is its official max context. No cut was made.
- Qwen1.5-MoE official max is 32768, but that OOMs at this file size on a 7-9 GB laptop. 2048 is the context that runs. 4096 is kept as a variant that needs about 9 GB free.
- The Qwen1.5-MoE file is called `Qwen-MoE-Q3_K_M.gguf` in the presets. If you download it under the publisher's name, rename it to that, or change the file name inside the preset line.
- File names are matched case-insensitively, so `Qwen2.5-Coder-7B-Instruct-Q6_K.gguf` and `qwen2.5-coder-7b-instruct-q6_k.gguf` both work.

## If your model is not in this list

You need to add one line in the launcher. Downloading a model is not enough.

1. Open `llama_launcher.pyw` in VS Code.
2. Find `PRESETS = [`.
3. Copy one similar line and change it. Example for a new Qwen model:

```
preset("My model | GENERAL", "My-Model-Q4_K_M.gguf", ctx(8192) + KV + QWEN_SAMPLING + REASONING_OFF),
```

4. Change only 3 things: the name in quotes, the file name, and the `ctx()` number if you want more or less context.
5. Save the file. Restart the launcher. Your new preset will show in the dropdown.

If you download a different quant of the same model, for example Q8_0 instead of Q4_K_M, change only the file name in that line. If you download a totally new model, copy the closest preset and change the file name and sampler. The sampler values come from the publisher page for that model.

A new preset does not need any NGL tuning from you. The first time you start it, the launcher works out the GPU layers itself and saves the result. It only fails if the model does not fit in RAM at all, and then it tells you so.

## Tested models, official presets, Auto NGL

All the models in the list above are models I have tested myself on this 16 GB laptop. The launcher contains the full information for each model: file name, context size, KV cache, sampler, and GPU layers.

All presets are official presets. That means the settings are taken from the official publisher page and the official ggml-org file, not guessed. If the publisher says temperature 0.6 and top-p 0.95, the launcher uses 0.6 and 0.95. If the publisher says a model needs a mmproj file for images, the launcher asks for that mmproj file. You do not need to search for settings.

If a publisher publishes no fixed sampling table, the preset says so instead of pretending one exists. Example: OLMoE has no fixed table, so it uses standard chat values and says so.

### Reasoning is set explicitly

Every preset sets reasoning ON or OFF clearly. Auto is not used. Auto is not neutral: it turns thinking ON for every template that supports it, even when the template itself defaults it to false.

### Missing files are not hidden

Presets stay visible even if the file is missing. The launcher shows what is missing and why it will fail, instead of hiding the preset.

### Auto NGL is already set up

NGL means how many layers of the model go to the GPU. More layers on GPU = faster. Too many layers = out of memory and crash.

You do not need to set NGL by hand. The launcher does it for you:

1. On first run for a preset, it tries coarse values first, then climbs with binary search to the highest working value. Example for a 36-layer model: `[37, 36, 32, 16, 8, 4, 0]`. 37 means layers + 1 for full offload. Fine values like 24 or 20 are found by the climb, not by trying every value.
2. Each trial is a real load + real generation with a long prompt, about 75% of the context (up to about 2500 tokens), not just a load check. This is where decode OOM shows up.
3. If file + KV + cache-ram + 1 GB reserve is bigger than free RAM, it skips the top trial and starts lower. The climb will still try higher if it actually works.
4. It waits for RAM to settle between trials and restarts. A killed server gives its Vulkan shared memory back slowly. Starting too early makes the next trial fail too low.
5. A trial that fails without a clear out-of-memory message is treated as "too high", not as a broken setup. That covers the case where the server dies mid-request and the client only sees a dropped connection. The climb keeps the last value that worked instead of throwing the whole tuning run away.
6. It picks the highest NGL that loads and answers, and saves it in `auto_ngl.json`.
7. The saved value is tied to model file + size + mtime + ctx + KV type + batch sizes + cache-ram + mmproj. If you reconvert the file or change cache-ram or ctx, it retunes. NGL itself is not part of the key, it is the value being tuned.
8. Next time you start the same preset, it uses the saved NGL directly. No re-test, unless free RAM has dropped by 1536 MiB since tuning. Then it retunes by itself.
9. If out of memory happens in real use, not just in tuning, it drops by 1, saves that new value, and restarts by itself. If it is already at 0, it stops and asks you to lower the context or free RAM.
10. Old saved entries for the same model + ctx + mmproj that are no longer live are cleaned when a new value is saved.

If you change your GPU driver, or you want to test again, just delete `auto_ngl.json` and start the launcher again. It will auto-tune again.

If auto-tune fails at every NGL, it shows an error. It does not silently fall back to CPU.

## Memory breakdown, Clear RAM, and Cleaner

- The Memory breakdown tab shows: Model weights, KV cache, Recurrent state, Compute buffers, Output buffer, Prompt cache / growth, and Other host RAM.
- Weights show the file size before start. After start, the split between graphics and processor comes from the llama.cpp log when the build prints buffer sizes. Builds that do not print them show the file size and estimate the split from GPU memory minus weights and KV, and the tab says so.
- KV cache is memory for this chat. It grows as the chat grows.
- Prompt cache is saved work from old questions. If you ask the same thing again, it uses the saved work, so it is faster. On an integrated GPU this row measures host RAM growth, so it can read 0 even though the cache is filling up in shared graphics memory.
- Compute buffers are rough work for reading your question. Bigger batch is faster but needs more memory.
- The launcher only kills its own server on Stop. It never kills another llama-server.exe on the machine.
- "Clear RAM" flushes the Windows file cache. It needs admin. Use the "Restart as Admin" button once to enable the full flush.
- "Cleaner" lets you select apps and trim their working sets. Trimming a running llama-server will slow generation, so it asks first.

## Can I reuse this as is?

Yes, if you are on Windows 10 or 11 and you keep the same layout. Change only BASE_DIR at the top of the file if your folder is different. Download your own models. GGUF files are not in this repo. Do not copy auto_ngl.json. It is recreated on first run.

## Does it work on Intel iGPU?

Yes. It is not specific to Radeon 780M. It uses Windows GPU Process Memory and the llama.cpp log. Intel Iris Xe, UHD, and Arc all appear as Vulkan0 in the log, same as Radeon 780M. Use a Vulkan build of llama-server.exe and install the Intel Vulkan driver. Linux and macOS are not supported.

## License

MIT
