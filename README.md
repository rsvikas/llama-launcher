# Llama Launcher

Run local AI on a normal 16 GB laptop. No cloud. No API key. No subscription.

This launcher starts llama-server.exe on your own laptop. Your prompts stay on your laptop. It works without internet.

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

- 【entity-Lenovo¦canonical_name=Lenovo】 ThinkBook 14 G9 AHP, model code 21V0
- 【entity-AMD¦canonical_name=AMD】 Ryzen 7 250, 8 cores, 16 threads
- 【entity-AMD¦canonical_name=AMD】 Radeon 780M, integrated, shared memory, no dedicated VRAM
- 16 GB DDR5-5600 RAM
- Windows 11, Python 3, llama.cpp with Vulkan
- Folder: C:\prism-llama
- Usual free RAM while running: 7 to 9 GB

## Quick start

1. Install Python 3 on Windows. Tick Add python.exe to PATH. No pip packages needed.
2. Download a Windows llama.cpp release with Vulkan support. Copy llama-server.exe into C:\prism-llama.
3. Download GGUF files you want. Put each .gguf and each mmproj file in C:\prism-llama.
4. Double click llama_launcher.pyw, pick a preset, press START. Open http://127.0.0.1:8080

## Can I reuse this as is?

Yes, if you are on Windows 10 or 11 and you keep the same layout. Change only BASE_DIR at the top of the file if your folder is different. Download your own models. GGUF files are not in this repo. Do not copy auto_ngl.json. It is recreated on first run.

## Does it work on Intel iGPU?

Yes. It is not specific to Radeon 780M. It uses Windows GPU Process Memory and the llama.cpp log. Intel Iris Xe, UHD, and Arc all appear as Vulkan0 in the log, same as Radeon 780M. Use a Vulkan build of llama-server.exe and install the Intel Vulkan driver. Linux and macOS are not supported.

## License

MIT
