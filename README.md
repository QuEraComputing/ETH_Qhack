> "To T or not to T, that is the question." — *Hamlet, Prince of DenmarQ*

## Prologue: An Apparition in Zurich

Hamlet, Prince of DenmarQ, and his loyal friend Qoratio traveled from the land of DenmarQ to represent their kingdom at ETH QHack and learn more about the current state of quantum computing research. After arriving in Zurich on Friday night, Qoratio convinced Hamlet to walk with him through Zurich's famous courtyards. Qoratio had heard a rumor that the ghost of an old physicist sometimes shared clues with hackathon participants in the middle of the night...

The air was cold. The stone walls were silent. Somewhere in the distance, a clock struck midnight. Then they saw him. A pale figure emerged between the arches, dressed in medieval armor but with the unmistakable bearing of an old physicist. The apparition looked at Hamlet and Qoratio with grave patience.

Then he spoke:

> *“List, list, O list.*
> *If thou wouldst know the present state of quantum computing, mark me well:*
> *The smaller quantum operation may demand the bigger engineering effort. Learn to fear 1-qubit gates.”*

![hamlet.png](./assets/hamlet.png)

“A 1-qubit gate is smaller. Simpler. Surely it should be easier. For long I have been getting the best performing circuits by cutting out CNOTs” thought Hamlet. Qoratio frowned, trying already to make sense of the claim. But before either of them could reply, the ghost began to fade, whispering only:

> *“Think logical qubits. Remember me.”*

Hamlet and Qoratio stayed up all night thinking about the ghost’s words. The claim still troubled them: how could a 1-qubit gate, so small and apparently so simple, become harder to implement than a 2-qubit gate? By morning, seeing no way to settle the matter on their own, they approached the QuEra team, who are known for their research in quantum error correction, and asked for help. The answer, they were told, would begin with the QuEra Challenge.

Using a technical write-up with code and visualizations, your team will help Hamlet and Qoratio work through a genuine quantum mystery: why can a seemingly simple 1-qubit operation become unexpectedly difficult to implement? You will learn what happens to a 1-qubit gate when you factor in gate synthesis, error correction, and architecture constraints. If all goes well, Hamlet and Qoratio will leave with a clearer answer, and your team will leave with a sharper understanding of some of the key open challenges in modern quantum computing.

---

## Contents

This repo contains everything you need to get going. Mainly:

- [`challenge.md`](challenge.md) contains the formal challenge statement, with directions and guidelines
- `assets` folder contains images and files that are relevant for different parts of the challenge
- [`bloqade_tutorial_eth.ipynb`](bloqade_tutorial.ipynb) contains the tutorial from the workshop which you will find very useful
- `ETHQHack.pptx` contains the presentation from the workshop
- [`pyproject.toml`](../pyproject.toml) is a configuration file from which you can easily recover all packages needed for the challenge, in their correct version

### Coding Infrastructure

This challenge will get you started with basic knowledge to use QuEra's upcoming gate-based, error-correction-focused, hardware. To that end, you will be operating on our SDK [Bloqade](https://bloqade.quera.com/latest/digital/).

To get the minimal packages installed all you have to do is make use of [`pyproject.toml`](https://www.notion.so/pyproject.toml) as follows:

### Setup (Python + `uv`)

**1) Install `uv`**

- **macOS / Linux:**
    
    ```bash
    curl -LsSf <https://astral.sh/uv/install.sh> | sh
    ```
    
- **Windows (PowerShell):**
    
    ```powershell
    irm <https://astral.sh/uv/install.ps1> | iex
    ```
    

Verify:

```bash
uv --version
```

**2) Install python version and dependencies from the provided TOML**

From the project folder (where the TOML lives):

```bash
# create and sync virtual environment according to toml
uv python install 3.12
uv sync
```

**3) Use the environment**
You can either activate it automatically using or IDE 
or manually from your terminal using:

```bash
# activate it
source .venv/bin/activate    # macOS/Linux
# .venv\\Scripts\\activate     # Windows (PowerShell)
```

With the above, you can just create python scripts and notebooks for your solution and run from the same folder where your new environment is. You will have the minimal infrastructure with which we can guarantee you can create meaningful solutions to the challenge! Still, you are welcome to bring other packages and software of your interest, if they can help you with your unique solutions!

---

## Submission and final deliverables

To submit:

1. Fork the challenge repository
2. Push your changes to your fork
3. Open a pull request from your fork against the original repo

We will judge based on the latest commit we find in the pull request before the end of the competition (disregarding commits that were pushed after the end of the competition).

Your pull request should contain:

1. **A technical write-up** that explains, with plots, code examples and/or visualizations, when and why the statement “*a 1-qubit gate can be harder to implement than a 2-qubit gate”* is true. The five sections in the `challenge.md` file are a suggested path for building the understanding and gathering the material you will need to create an effective write-up. Your final write-up does not need to follow the order in which you explore the topics in `challenge.md`, but it should engage with the main ideas from each of the sections (except for part 5 which is an optional bonus). You should assume the reader is already familiar with Bloqade so you should not explain what Squin, PyQrack, Tsim, and other Bloqade tools are or how to use them.
2. **Supporting code and figures** used to build your argument. All the code that is mentioned or used in the write-up should be clearly organized inside the repo. We suggest that you use explicit names for your files and that you consider using folders to organize different pieces of code. Remember that we should be able to run your code. Make sure that all dependencies are documented in the `pyproject.toml` file and that you use `uv` to manage your Python environments.
3. **A presentation** that you will deliver at the end of the competition, where you share with us what you learned and the results you found. We encourage you to explain, to the best of your understanding, why this topic is important for the future of quantum computing.

Make sure all your files are on your pull request on time!

### Note on AI use

We will expect your team to be able to answer basic questions on whatever appears in your write-up and presentation. **If you use AI tools, we just ask that you add a very brief section (just a few sentences) at the end of your write-up and a slide in your presentation explaining what tools you have used and for what.**

As long as you can fulfill that expectation, you are welcome to use AI in any way you like. If you are an advanced AI user, you are also welcome to experiment with your own agentic workflows. If you are looking for some ideas on how to create agentic workflows you can also check out the latter portion of the slides in the [working-with-coding-agents](https://github.com/Roger-luo/working-with-coding-agents) repo authored by Roger Luo, Scientific Software Lead at QuEra.

---

## Judging criteria

Remember that this is a “**1-qubit challenge”**. A strong technical write-up should make a clear and engaging argument on why applying 1-qubit gates can be surprisingly complicated.
The strongest submission will likely:

- Introduce the topic in a creative way
- Use visualizations to build intuition
- Explain what strategies were explored and why. How did you reason about what you discovered and what did you consider to be "better"?
- Connect the different parts into a coherent story
- Leave Hamlet and Qoratio (the reader) with a sharper sense of what the current status of quantum computing research is

For the code and presentation we will reward:

- Correctness. Do your code and the methods you used make sense in this context?
- Clarity. Can we easily understand and run the key portions of your code?
- How far did you push the gate synthesis and the simulation methods? How did you navigate the different trade-offs presented in the challenge?
- Quality of visualizations and comparisons. How did you use your technical skills to visualize/compare different approaches?