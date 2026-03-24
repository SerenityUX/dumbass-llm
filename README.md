# DumbAssLLM (DALLM)

I wonder if I can make an LLM in one week.

Maybe by the end of the week I can ask it a simple question like **“what is the capital of California?”** and it will answer **“Sacramento”** All on my laptop.

**What’s in this repo.** Almost all the code lives in the `dallm/` package (same letters as the project name—DALLM). Roughly in pipeline order: `dallm/data_remote.py` pulls Alpaca JSON or the Simple English Wikipedia dump if they’re missing; `dallm/wiki_extract.py` runs WikiExtractor on the XML and strips `<doc>` junk; `dallm/tokenize.py` walks `extracted/` and writes `tokens.npy`; `dallm/pretrain.py` is the GPT + pretraining loop; `dallm/finetune_alpaca.py`, `dallm/finetune_sponge.py`, and `dallm/finetune_city.py` are three supervised fine-tunes on top of that; `dallm/generate.py` loads a checkpoint and samples. 

**`main.py` is the entry point**—the `data_need_*` booleans turn individual steps on for a run, and when those are off you just get the interactive menu to try base / SFT / Patrick / city (assuming the checkpoints exist).

---

## The week

**Day 1 — Saturday.** First day. I have until Friday (spoiler it only took a couple days instead of a whole week). I’m calling it **DumbAssLLM**.

### Six steps (how I think about it)

1. **Data** — Get text out of dumps (parse XML / wikitext, optional cleanup, chunk articles).
2. **Tokenization** — Turn text into integer token IDs (BPE or similar, fixed vocab).
3. **Model** — Decoder-only Transformer: embeddings → blocks (attention + MLP) → scores over the vocab (logits).
4. **Training loop** — Batches of sequences; predict next token; cross-entropy loss; backprop; optimizer step.
5. **Evaluation & generation** — Track loss on held-out text; sample tokens from the model to see if it’s getting better.
6. **Scale & ops** (optional but real) — Start with a subset if you need to, MPS on Mac, batch/sequence sizes that fit memory, save checkpoints.

### Simpler way of saying the same thing

1. **Get text** — Some big pile of strings (even one file of plain text is enough to start).
2. **Tokenize** — Map text → integer token IDs (you need a vocab + encode/decode).
3. **Model** — A network that takes token IDs and outputs next-token logits (e.g. small Transformer).
4. **Training loop** — Batches, next-token targets, loss (usually cross-entropy), backward pass, optimizer updates.

---

## Step 1: Data

I went to the [Simple English Wikipedia dumps](https://dumps.wikimedia.org/simplewiki/latest/) and downloaded `simplewiki-latest-pages-articles.xml.bz2` (~345MB as of Mar 2026). ([English Wikipedia](https://dumps.wikimedia.org/enwiki/latest/) has a bigger `enwiki-latest-pages-articles.xml.bz2` if I want more text later.)

I have a small script that reads the XML and writes a bunch of little text files (“shards”). It’s the same idea as one giant text file—just split into pieces.

---

## Step 2: Tokenizer

A tokenizer takes text and turns it into **tokens**, then into **IDs** the model can use.

Example:

```
"Hi my name is Thomas. In life we are always learning"
→ ["Hi", " my", " name", " is", " Thomas", ".", " In", " life", " we", " are", " always", " learning"]
→ [10234, 616, 1437, 318, 15289, 13, 554, 1223, 356, 389, 2829, 6507]
```

Those numbers are IDs pointing at entries in a fixed list of tokens. The model does math on numbers, not on the word `"we"`—it might see `356`. When it runs, it outputs numbers too; the tokenizer turns those back into text.

**Packages I use for this part**

- `transformers` (Hugging Face) — tokenizer.
- `numpy` — store and work with a long 1D array of token IDs efficiently.

**Input:** files under `./extracted/` (e.g. `AA/wiki_00`, …).  
**Output:** `tokens.npy` (one long list of token IDs) and `tokens_meta.json` (vocab size and counts).

---

## Step 3: Model + training (next-token guessing)

I load the token IDs into memory, **slice them into windows** (you can’t feed the whole list in one shot—it would blow RAM/GPU and that’s not how transformers are trained), turn them into vectors so the net can see patterns, then train it to guess the **next** token. Backprop nudges weights so predictions improve.

To **generate**, I take scores (logits) from the last step, pick the next ID (sample or argmax), decode with the **same** tokenizer, repeat.

### The only three ideas to hold

1. **Text → numbers** — You can’t plug `"the"` straight into matrix math, so you use a tokenizer → `tokens.npy`.
2. **Numbers → patterns** — The net doesn’t “read” IDs as English; each ID becomes a vector (a list of floats) and it learns from those. That’s the **model + training**.
3. **Train on chunks, not the whole wiki at once** — Same as reading a paragraph at a time instead of memorizing the whole book in one glance.

### Two fixes to how I first pictured training

1. **“Right or wrong”** — It’s not one yes/no per sentence. At **each position** the model outputs a **distribution** over all tokens; you compare that to the **true** next token with a **loss** (cross-entropy). You get a number (“how wrong / how surprised”), then backprop tweaks **all** the weights a little. So it’s **graded wrongness** over many positions and many steps.
2. **“Guess the next token” in training** — You already know the true next token in the window (**teacher forcing**). The model’s job is to give **high probability** to that token. At **generation** time you **don’t** have the answer—you sample (or argmax) from logits and feed the result back in.

---

## What surprised me: this is not ChatGPT

That clicked. **It’s autocomplete.** You can still give a **prompt** and see what it continues with.

**Prompt:** `The capital of France is...`

**Model output (base / early training):**

```
The capital of France is a person, and east that the professor of the Stockholm sculptures to be (18 and be the Sa date of a earth, and anthereina the newly and when, 2025 bythat of the West after himself he was to determine and and recognized) liferied indent. After to the 38 United States it was the conferenceas, it worked4men just6.  involved). The in 1991 a 47 with hunting air series passes to the Presentl also the west, a Britishost arts's formal and he alsodoc.
```

Word salad—it only *sounds* like English if you’re not reading closely.

---

## Day 2: more steps, SFT, and smaller experiments

I trained for **10,000 steps** (instead of 2,000) to try to push loss down.

**Training log:** `step 10000 loss 3.8729`

**Prompt:** `Paris, the capital of`

**Model continuation:**

```
Paris, the capital of the vise region.
Lake and western dry was part of the Philippines. France was called the "preigness" of Saint Petersburg on 1 March 2016.
Council of the Egyptian government in 1916, the Chilean victory did not play in the past decades and to host an exchange role for some new teams. Nundz washew by the 21st Premier of the group of Wessex. In 1945, the inauguration was removed from the Croatian government in Paris.
Going at birth, the ability to wear. But in March 2019, the Rpsilon continued in the publishers to be in
```

It says France in there, so it’s **pretty good** for where the model is at. It **sounds** like real English if you’re not paying attention.

**Supervised fine-tuning (SFT):** take the rambling base model, feed it **Question / Answer** pairs so it learns that **shape** of interaction. I used **Stanford Alpaca**-style formatting:

1. Download the dataset  
2. Format the data so we know how we want it to respond  
3. Tokenize it  
4. Define the model architecture so it matches the base model (load the base, same shapes as before)  
5. Run a training loop on the Alpaca data so it learns question-and-answer format  

**Alpaca SFT — passage question (model still stupid here):**

```
### Instruction:
Based on the following passage, answer the question. Passage: Sacramento is the capital of California... Question: What is the capital of California?

### Response:
The center of California is a nation of Sri Lanka. It is the last part of the United States by the United States Constitution, or by the United States. The current state of California has the same population since the population of the state of Nebraska. The population of the state has the population of the United States, the population of the population is the United States. It has a total area of 23,86,811. The state of California is located on the US state of California, and the state of California. It is the capital of the New state of Washington, New York. The
```

Damn this model is stupid—that was **after** SFT on Stanford Alpaca.

**Tangent:** fine-tune on **Patrick Star** (every line he said in SpongeBob)—it just rambles like Patrick. Then fine-tune on **US state capitals**—it takes the data and returns the city for US states **pretty well**, with the caveats below.

**City-style fine-tune (instruction format) — real outputs:**

```
### Instruction:
What is the capital of France?

### Response:
The current region is a state of France, in the United States, and the fourth largest city of Belgium. It is surrounded by the population of 7,911.7 million people live in the United States, and all of the most famous city include the majority of the city, number of people living in the United States and the population of the population of Justice. The city has a population density of inhabitants and the United States, with a population of 15,932 to around dry. The capital of the city is the largest city in the United States, and is a new nation.

### Instruction:
What's capital of California?

### Response:
The capital of this state is Sacramento. Many of this state

### Instruction:
What's capital of New York, I love new york pizza?

### Response:
The capital of this state is Albany. The capital of this

### Instruction:
Capital of Vermont please?

### Response:
The capital of this state is Montpelier. The capital

### Instruction:
How many stars on flag of USA?

### Response:
It's Denver. The capital of this state is Hartford.


### Instruction:
"How many stars are in the sea?"

### Response:
Me'm myself and misses your little ago? So he is gonna get it! I've be there? That happens, They are we're too much good. We're okay. Squidward, I am actually you're gonna hear we're just like our me. There's no blet on the very bad? And we're off, but I was here. This is a nice all, but I don't do what you must be lost the way. He said,? That's wrong, I'm it! He know. But I'm gonna come here, I'm so it's there
```

So: **Sacramento**, **Albany**, **Montpelier** can hit. Ask something outside that narrow training (stars on the flag) and it **makes stuff up**—and a Patrick-style fine-tune can **bleed** into weird prompts (“stars in the sea”) so you get Patrick-ish ramble.

That’s the tradeoff: small model + limited steps + narrow training text = **format** sometimes works, **facts** and **general chat** often don’t.

---

## Other instruction-style datasets (ideas only)

If you want more Q&A-style text to fine-tune on, the web is full of examples—for instance [creative-graphic-design](https://huggingface.co/creative-graphic-design), [fudeno-instruct-4M](https://huggingface.co/datasets/takara-ai/fudeno-instruct-4M), [ui-instruct-4k](https://huggingface.co/datasets/iamdyeus/ui-instruct-4k), [modern-webapp-instructions](https://huggingface.co/datasets/SpaceGhost/modern-webapp-instructions). This repo doesn’t need all of them; they’re just places people host that kind of data.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You’ll get `torch`, `transformers`, and `numpy`.

---

## Running this repo

**Interactive menu** (pick base / SFT / Patrick / city — you need the matching checkpoint files first):

```bash
python3 main.py
```

At the top of `main.py` there are switches like `data_need_extraction`, `data_need_tokenization`, `data_need_training`, `data_need_fine_tuning`, `data_need_train_patrick`, `data_need_train_city`. Turn on only what you need for one run, then turn them off again for menu-only use.

Big files (dump, `extracted/`, `tokens.npy`, checkpoints, downloaded Alpaca JSON, generated `city_data.json`) are in `.gitignore`; you recreate them locally with the scripts. **`SpongeBobTranscript.csv` is kept in the repo** for Patrick fine-tuning.

---

## Main files

| Path | What it does |
|------|----------------|
| `dallm/data_remote.py` | Fetch Alpaca JSON / Simple Wikipedia bz2 when absent |
| `dallm/wiki_extract.py` | Wiki dump → `extracted/` shards + cleanup |
| `dallm/tokenize.py` | Shards → `tokens.npy` + `tokens_meta.json` |
| `dallm/pretrain.py` | GPT + pretrain loop → `checkpoint.pt` |
| `dallm/finetune_alpaca.py` | Alpaca SFT → `checkpoint_sft.pt` |
| `dallm/finetune_sponge.py` | Patrick (CSV) → `checkpoint_sponge.pt` |
| `dallm/finetune_city.py` | US capitals → `checkpoint_city.pt` |
| `dallm/generate.py` | Sample from a checkpoint |
| `main.py` | Training switches + colored interactive CLI |

This is a **learning project**, not a polished product.
# dumbass-llm
