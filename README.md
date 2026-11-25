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
- `coding-system-message.txt` - One extra block (a `CODE` section) to paste at the start of the chat in the llama.cpp web UI.
- `code-only-prompt.txt` - A short 128-word prompt for code work, with no "I don't know" in it.
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
- It also names the llama.cpp release you are running, because that is what decides which options the server accepts.
- If another copy of the launcher is on disk and it is a **different version**, it says so and names that version. A copy of the same version is harmless and is not reported. Copies of different versions share the same `auto_ngl.json` but compute different keys, so they keep re-tuning each other and overwriting the saved values. Keep one copy.
- If there is not enough free RAM for the weights and the chat memory of the model you picked, it says so before the server starts, instead of letting it run out of memory halfway through a request. It also names the lighter preset that runs the same model with a smaller context, and points at `--cache-ram`, because flushing the file cache frees very little when the memory is held by open apps rather than by the cache.

## Models already in the launcher

These presets are already in `llama_launcher.pyw`. You download the GGUF files yourself and put them in `C:\prism-llama`.

| Preset in the dropdown | File you need to download | Use |
| --- | --- | --- |
| 1. Bonsai 2 27B - thinking (8k ctx) | Ternary-Bonsai-2-27B-Q2_0.gguf only | Thinking, stock llama.cpp only, no PrismML fork |
| 1b. Bonsai 2 27B - thinking (4k ctx, lighter) | Ternary-Bonsai-2-27B-Q2_0.gguf only | Same model, smaller context and batch, for a tight machine |
| 1c. Bonsai 2 27B - no thinking (8k ctx) | Ternary-Bonsai-2-27B-Q2_0.gguf only | Instruct mode, faster |
| 2. Bonsai 2 9B - thinking (16k ctx) | Ternary-Bonsai-2-9B-Q2_0.gguf only | Thinking |
| 2b. Bonsai 2 9B - no thinking (16k ctx) | Ternary-Bonsai-2-9B-Q2_0.gguf only | Instruct mode, faster |
| 3. Bonsai 2 9B - images + thinking, chat only | Ternary-Bonsai-2-9B-Q2_0.gguf + Ternary-Bonsai-2-9B-mmproj-Q8_0.gguf | Images, thinking. Chat apps only, not agent tools |
| 4. Gemma 4 E4B - no thinking (24k ctx) | gemma-4-E4B_q4_0-it.gguf | General chat |
| 4b. Gemma 4 E4B - thinking (24k ctx) | gemma-4-E4B_q4_0-it.gguf | General chat with thinking |
| 5. Qwen3 8B - thinking (16k ctx) | Qwen3-8B-Q4_K_M.gguf | General chat |
| 5b. Qwen3 8B - no thinking (16k ctx) | Qwen3-8B-Q4_K_M.gguf | General chat, faster |
| 6. Llama 3.1 8B - general (8k ctx) | Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf | General chat |
| 7. Gemma 4 E4B - images (16k ctx) | gemma-4-E4B_q4_0-it.gguf + mmproj file | Chat with images |
| 8. Qwen2.5-VL 7B - images (8k ctx) | Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf + mmproj file | Chat with images |
| 9. Ultravox audio - experimental (4k ctx) | Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf + mmproj file | Audio, experimental |
| 10. Qwen3-ASR 1.7B - speech to text (8k ctx) | Qwen3-ASR-1.7B-Q8_0.gguf + mmproj file | Speech to text |
| 11. SmolVLM2 500M - video (8k ctx) | SmolVLM2-500M-Video-Instruct-Q8_0.gguf + mmproj file | Video |
| 12. Qwen2.5-Coder 7B - coding (16k ctx) | Qwen2.5-Coder-7B-Instruct-Q6_K.gguf | Coding. The model takes 32k, 16k is what fits here |
| 13. Qwen2.5-Coder 14B - coding (8k ctx) | qwen2.5-coder-14b-instruct-q4_k_m.gguf | Coding, bigger model |
| 13b. Qwen2.5-Coder 14B - coding, lighter (4k ctx) | qwen2.5-coder-14b-instruct-q4_k_m.gguf | Same model, smaller context, for a tight machine |
| 14. MiMo-V2.6-Distill-Qwen-9B - agentic (8k ctx) | MiMo-V2.6-Distill-Qwen-9B-Q4_K_M-OFFICIAL.gguf | Agentic |
| 15. MiMo-V2.6-Distill-Qwen-9B Q8_0 - agentic (2k ctx) | MiMo-V2.6-Distill-Qwen-9B-Q8_0.gguf | Agentic, better quality, smaller context |
| 16. Qwen3.5-9B - no thinking (8k ctx) | Qwen3.5-9B-Q4_K_M-OFFICIAL.gguf | General, no thinking |
| 17. Nemotron 3 Nano 4B - thinking (64k ctx) | NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf | Reasoning, long context |
| 18. DeepSeek-R1-Distill-Qwen-7B - thinking (8k ctx) | DeepSeek-R1-Distill-Qwen-7B-Q4_K_M-OFFICIAL.gguf | Reasoning |
| 19. Ministral 3 8B - general (8k ctx) | Ministral-3-8B-Instruct-2512-BF16-Q4_K_M-OFFICIAL.gguf | General |
| 20. Nemotron-Nano 9B - thinking (4k ctx) | Nemotron-Nano-9B-v2-Q4_K_M-OFFICIAL.gguf | Reasoning |
| 21. OLMoE 1B-7B - general (4k ctx) | OLMoE-Q6_K.gguf | MoE chat, 64 experts, 8 active |
| 22. Qwen1.5-MoE-A2.7B-Chat - general (2k ctx) | Qwen-MoE-Q3_K_M.gguf | MoE chat, 60 experts, 4 active |
| 22b. Qwen1.5-MoE-A2.7B-Chat - general (4k ctx) | Qwen-MoE-Q3_K_M.gguf | Same model, 4k context, needs about 9 GB free |

Notes:
- Bonsai 2 runs on stock llama.cpp only. Only the Q2_0 pack runs here. There is no PrismML fork.
- The Bonsai 2 27B has 64 layers. On the Radeon 780M tested here it settles at about 19-20 layers on the graphics chip. The limit is how much RAM is free, not a fixed pool inside the graphics chip: with more free RAM the ceiling moves up. **The BIOS UMA frame buffer size is not the lever.** On this laptop the UMA buffer is carved out of system RAM, so raising it takes the same RAM away from Windows and from the model - setting it to 2 GB cost 2 GB of free RAM and the launcher ran out of memory sooner. Leave it at 512 MB.
- The three Bonsai 2 27B presets use `--cache-ram 512` while every other preset uses 2048 (or less). The 27B needs every megabyte of host RAM it can get, and the smaller prompt cache is what leaves it some. It is also a llama.cpp server setting, not a publisher setting - the publisher-official values in these presets are the samplers.
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
6b. The winner is tried **one more time** before it is saved. A trial can pass in a lucky moment - the file cache is in a different state, the machine is a little quieter - and then the very same value runs out of memory in real use, where the answer in progress gets cancelled. The re-check costs one extra model load and catches that in advance.
6c. **One layer of headroom is kept.** The highest value that passes is always directly under a value that failed, so it sits at the edge of what the machine can hold - and the edge is where real requests fall over, because the machine is in a different state by then. So the saved value is one layer below the winner, never below the coarse floor that was tested. One layer out of 64 is about 1.5% of the offloaded work, and the generation speed is set by the layers still on the processor anyway. The winner is written to `auto_ngl.json` as `ceiling_ngl`, so you can see both numbers.
7. The saved value is tied to model file + size + mtime + ctx + KV type + batch sizes + cache-ram + mmproj. If you reconvert the file or change cache-ram or ctx, it retunes. NGL itself is not part of the key, it is the value being tuned.
8. Next time you start the same preset, it uses the saved NGL directly. No re-test, unless free RAM has dropped by 1536 MiB since tuning, or you have changed the sampling mode (see point 11). Then it retunes by itself.
8b. It also re-tunes when a lot **more** RAM is free than when the value was tuned (4096 MiB more). Without this, the value written by the out-of-memory step-down in point 9 would stay low for ever, even after you close the apps that caused the out-of-memory in the first place. Free RAM is read three times and the lowest reading is used, because the Windows file cache alone moves that number by more than a gigabyte when a big model file is read or trimmed. The number the saved value is compared against is the **highest** of those samples, taken after the memory of a killed server has come back - otherwise a slow drain looks like the machine getting faster.
8c. Every saved value keeps the free RAM it was measured with, including the ones written by the step-down in point 9. Without that number the check in point 8 could never fire again for that preset.
8d. A value that was just tuned is locked for an hour, and a value that was just lowered after an out-of-memory error (point 9) is locked for ten minutes. Without the lock, a value that has just been measured can claim that there is even more RAM free and start a second, third and fourth re-tune, and each one loads the model several times.
9. If out of memory happens in real use, not just in tuning, it drops by 1, saves that new value, and restarts by itself. If it is already at 0, it stops and asks you to lower the context or free RAM.
10. If the server is killed by the graphics driver and prints no out-of-memory message at all, the exit code is the tell. The launcher treats that as too many GPU layers and steps down, instead of leaving you with a dead server.
11. The sampling mode is remembered with the saved value. The sampler is part of the memory footprint, so switching between modes re-tunes that one preset rather than reusing a number that was measured with different settings.
12. Old saved entries for the same model + ctx + mmproj that are no longer live are cleaned when a new value is saved.

If you change your GPU driver, or you want to test again, just delete `auto_ngl.json` and start the launcher again. It will auto-tune again.

If auto-tune fails at every NGL, it shows an error. It does not silently fall back to CPU.

## Memory breakdown, Clear RAM, and Cleaner

- The Memory breakdown tab shows: Model weights, KV cache, Recurrent state, Compute buffers, Output buffer, Prompt cache / growth, and Other host RAM.
- Weights show the file size before start. After start, the split between graphics and processor comes from the llama.cpp log when the build prints buffer sizes. Builds that do not print them show the file size and estimate the split from GPU memory minus weights and KV, and the tab says so.
- KV cache is memory for this chat. It grows as the chat grows.
- Prompt cache is saved work from old questions. If you ask the same thing again, it uses the saved work, so it is faster. On an integrated GPU this row measures host RAM growth, so it can read 0 even though the cache is filling up in shared graphics memory.
- `--cache-ram` is the limit for that prompt cache, in **host RAM**. It is a cap, not an upfront allocation: the memory is used as you actually chat, and only up to the limit. But every megabyte of it is a megabyte the model cannot keep resident, so on a small machine a large `--cache-ram` lowers the number of layers that fit and can end the chat with an out-of-memory error. It is a llama.cpp server setting, not something a model publisher specifies - the publisher values in this launcher are the sampler ones. Which value to use depends on how you chat:
  - **Fresh document every question** (paste, ask, repeat): the cache buys nothing, because a new document has no prefix in common with the old one. Use `--cache-ram 512`, or even 0.
  - **Several follow-up questions on the same document**: the shared prefix is reused instead of being processed again, and a 5000-token document at 60 tokens per second is about 85 seconds of processing saved per follow-up. Keep `--cache-ram 2048`.
  - **Middle ground**: `--cache-ram 1024`.
  Watch the "Prompt cache / growth" row on the Memory breakdown tab. If it stays at 0, nothing is being cached and the memory is better spent on the model.
- Compute buffers are rough work for reading your question. Bigger batch is faster but needs more memory.
- The launcher only kills its own server on Stop. It never kills another llama-server.exe on the machine.
- "Clear RAM" flushes the Windows file cache. It needs admin. Use the "Restart as Admin" button once to enable the full flush.
- "Cleaner" lets you select apps and trim their working sets. Trimming a running llama-server will slow generation, so it asks first.

## Reducing hallucinations

Small models blurt. Two different things go wrong, and they need different fixes.

**"It just blurts things out"** is sampling. The model is picking tokens out of the thin tail of the distribution, where confident-sounding but wrong words live. The `Sampling:` dropdown next to the buttons fixes that, by replacing the publisher's sampler with a tighter one:

| Mode | What it does |
| --- | --- |
| Normal (publisher settings) | The values from the publisher's own page. Nothing is changed. |
| Precise (less random) | temperature 0.3, top-k 20, top-p 0.8, min-p 0.05, repeat penalty 1.05. The usual first thing to try. Costs some variety, buys a lot of accuracy. |
| Deterministic (same answer) | temperature 0. The same question always gives the same answer. That is how you find out whether a change to the prompt actually helped. |
| Mirostat 2 (steady) | Keeps the perplexity in a fixed band instead of following the model's own confidence. llama.cpp ignores top-k and top-p when Mirostat is on, so they are left out. |
| Precise + DRY (no repeats) | Precise, plus the DRY sampler that punishes repeating a phrase. Small models loop, and loops read as hallucination. |
| Code (precise, no repeats) | For the code presets. Like Precise, but it keeps the publisher's repetition penalty of 1.1 instead of dropping it to 1.05, and adds the DRY sampler. On code the repetition penalty matters: a lower one lets the model repeat the same line instead of moving on. The DRY length is 3 here, not 2, because a two-word fragment repeats by nature in code. |

The mode replaces the sampler flags, it does not add to them, so it always wins over the publisher's values. The command line at the top of the window shows what is actually being run. Changing the mode does trigger a re-tune: the sampler is part of the memory footprint, so the saved GPU layer count for the old mode is not reused for the new one.

**Changing the mode while a server is running.** The running server keeps the settings it was started with. When you change the mode, the launcher asks whether to restart it, and says plainly that a restart cancels an answer in progress. If you say no, the command line keeps showing a marker so you do not forget:

```
[sampling changed - the running server still uses the old settings, press RESTART]
```

Restarting after a mode change re-tunes for the new sampling settings first, so it takes longer than a plain restart. Restarting without changing the mode reuses the saved value, so the server comes back in seconds.

**"It says things that are not true"** is not a sampling problem. A model that does not know has no way to admit it, and no temperature will teach it. The fix is to give it the facts: for anything current or specific, search first (Exa MCP, see above), and for code, let it read the actual file instead of recalling it.

What does not work: asking the model to check its own answer. On a model this size the second pass is usually just as wrong, only longer, and at 2 tokens per second it costs three times the wait. If you want a second opinion, use a second model or a second pass on a bigger machine, not a prompt.

**Where to put the rules.** The llama.cpp web UI has a **System Message** box in its settings. Anything you put there is sent with every request, so the model follows it in every chat, and it survives the web UI's own conversation compaction. That is the right place for "don't guess, quote your source, say I don't know". The launcher cannot set it for you: that depends on your llama.cpp version, and a client that sends its own system message would override it anyway.

### One more point for code: the CODE section

`coding-system-message.txt` in this repo holds exactly one block and nothing else. Paste it as the **first message of the chat** in the llama.cpp web UI - the UI sends it with the conversation, so it applies to every question that follows.

```
CODE
When you write or explain code, the code must be real. These rules are about what you write, not about whether to answer.
Answer what was asked, in full. An open-ended request has a reasonable default answer: pick the most standard approach and state the assumption in one line.
When you are not certain about a detail, do not stop: write the code, mark the uncertain parts, and say what you assumed.
This machine runs Windows and Python 3. Prefer a solution that installs with pip alone and needs nothing else. List what each library needs besides pip - a separate program, a system library, a compiler - and say "none" when there is nothing.
Naming a real, widely used API is not inventing one. Inventing is making up a function, class, method, option or path that does not exist.
When you are shown code - a file, a snippet, a documentation excerpt - use exactly the names, APIs, options and versions it contains. When something is not in front of you, write the most reasonable version and state the assumption.
Write complete, runnable code: no "...", no "rest of the code is similar", no placeholder comments where real work belongs. Show the whole function or block that goes into the file, and say which file it belongs in.
When you explain code, describe what that code actually does, not what it probably does.
Never say code was tested, run or verified unless you actually ran it in this conversation. Say "not run".
```

The launcher has nothing to do with this: it cannot set a system message for you, because that depends on your llama.cpp version and a client that sends its own system message would override it anyway. The web UI sends it, so the web UI is where it belongs.

If you would rather have it as a real system turn than as the first user message, the llama.cpp web UI has a **System Message** box in Settings, and the same text goes there.

Five things about the wording, all of them learned the hard way:

- Every line is a **behaviour**, not a mood. "Be honest" is easy to agree with and then ignore; "never invent a function name you have not seen" is followable.
- **They must not be able to read as "refuse".** The first version said "if the code you need is not in front of you, ask for it" and "if you cannot write correct code, say what you can do instead". On the 14B that turned into: *"I don't have enough information to answer accurately - please tell me which library you are using and what transformations you want."* The rules now say plainly that they are about what the model writes, not about whether it answers.
- **They have to override the base prompt, not just add to it.** The next version still refused, and this time the opening words - *"I don't have enough information to answer accurately"* - were quoted straight out of the base prompt's own `WHEN INFORMATION IS MISSING` section. That section says to ask one clear question when something is missing and not to make hidden assumptions; for "write a script that converts X to Y", choosing the library *is* the job. So the rules now say three things that the base prompt does not: for a code task a missing detail is an assumption to state, not a reason to ask; if a detail would change the code, write the most likely case and put the question at the end; and when you are writing new code and no API was given, use the standard or the well-known library and name it.
- The last line is there on purpose. A strict "never write code unless you are certain" gives empty answers on a small model, because it is not certain about most things. Marking the one uncertain part keeps the answer useful and still stops the invention.

- **The wording is not the lever - the base prompt is.** Five runs of the same question on the same model. Run 3 answered (608 tokens) with rules full of the words "ask", "missing" and "library"; run 5 refused with almost none of them. So the wording is not what decides it, and the outcome at temperature 0.3 is close to a coin flip. What is constant in every refusal is the base prompt: the model's reply opens with *"I don't know how to directly convert..."* or *"I don't have enough information to answer accurately"*, which are the templates from `WHEN INFORMATION IS MISSING` ("Say plainly 'I don't know'") and from `WHEN IN DOUBT` ("choose uncertainty"). For a task that is one long unseen script, every answer is uncertain, so "choose uncertainty" means "do not write". The rules below say what to do and never what to ask for, but the real correction belongs in the base prompt.
- **Tell it what machine it is writing for.** Without that, the model reached for `pdf2image`, which needs Poppler - a second program that never got mentioned, so the script would have failed on the first run. The rules now say: this machine runs Windows and Python 3, prefer a solution that installs with pip alone and needs nothing else, and if something extra is unavoidable, say so and give the exact install step.

### If it still refuses: the whole prompt, ready to paste

Take your base prompt **unchanged** (all 15 lines), add the `CODE` section below it,
and **one added sentence in each of two sections**:

`WHEN INFORMATION IS MISSING`:

> Missing information is not a reason to refuse. If the question has a reasonable
> default answer, give it and say what you assumed. Ask only when the answer would
> change in a way that matters.

`WHEN IN DOUBT`:

> For code, uncertainty about a detail is not a reason to stop: write the code, mark
> the uncertain parts, and say what you assumed.

Both refusals came from those two sections: the model opened with *"I don't know how
to directly convert..."* (the "Say plainly 'I don't know'" template) and *"I don't have
enough information to answer accurately"* (the other template). One sentence in the
right place beats three overrides in the wrong one.

### How to ask for code

Put the constraint **in the question**, not only in the system message. The system
message sits thousands of tokens away from the question, and a constraint next to the
question carries far more weight. Three words in the question can do what a paragraph
of rules cannot:

```
write a python script to convert scanned pdf to md file.
Constraints: Windows 11, Python 3.11, pip-only libraries - nothing else to install.
Give me the full script plus the pip install line.
```

This is also the only way to get past a **knowledge gap**. A rule that says "prefer a
library that installs with pip alone" cannot fire on `pdf2image`, because the model
does not recall that `pdf2image` drives Poppler's `pdftoppm`. It picks the most
familiar library and the rule passes over it. The rules now ask the model to *list*
what each library needs besides pip, which forces the check it was not doing.

### Or use a separate, short prompt for code

`code-only-prompt.txt` is 128 words with none of the careful-factual baggage: no
"I don't know", no "choose uncertainty", no evidence rules. It is the antidote to the
refusal bias, and it is short enough to read in one go.

```
You write code. Answer the request with complete, runnable code, not a description of code.

State your assumptions in one line at the top, then write the code.

Use the real names of the libraries and APIs you rely on, and say which ones have to be installed first. This machine runs Windows and Python 3; prefer a solution that installs with pip alone.

Show the whole function or block that goes into the file, and say which file it belongs in. No "...", no "rest of the code is similar", no placeholder comments where real work belongs.

Never say the code was tested or run unless you actually ran it.

If one detail is uncertain, write the rest and mark that detail: "not sure - needs <what>".
```

The long careful-factual prompt is the right prompt for *questions*. For *code*, it is the wrong one: it tells the model to prefer admitting uncertainty, and a script is uncertain by nature. Two prompts, two jobs - paste the short one when you switch to a code preset.

**Test it twice with the same prompt.** The refusals are not perfectly repeatable -
one run of the same rules wrote a 608-token answer, the next refused. That means
sampling is part of it, and you can find out which factor decides by switching the
Sampling dropdown to **Deterministic (same answer)** (temperature 0): if the same
question gives the same result every time, the prompt is the whole story; if it
still flips, the temperature is.

**What these rules cannot do.** They stop the model from inventing an API or a file path, and they stop it from refusing. They do not make it think harder. If the answer is a plain, correct, boring recipe - a well-known one from the training data - the rules will not make it a better recipe. For that, the two launcher-side levers matter: the **context size** of the code presets (a script that runs out of room gets cut off) and the **Code (precise, no repeats)** sampling mode.

## Can I reuse this as is?

Yes, if you are on Windows 10 or 11 and you keep the same layout. Change only BASE_DIR at the top of the file if your folder is different. Download your own models. GGUF files are not in this repo. Do not copy auto_ngl.json. It is recreated on first run.

## Does it work on Intel iGPU?

Yes. It is not specific to Radeon 780M. It uses Windows GPU Process Memory and the llama.cpp log. Intel Iris Xe, UHD, and Arc all appear as Vulkan0 in the log, same as Radeon 780M. Use a Vulkan build of llama-server.exe and install the Intel Vulkan driver. Linux and macOS are not supported.

## License

MIT
