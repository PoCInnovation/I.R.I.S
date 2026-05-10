> If you see this section, you've just created a repository using [PoC Innovation's Open-Source project template](https://github.com/PoCInnovation/open-source-project-template). Check the [getting started guide](./.github/getting-started.md).

# I.R.I.S - Integrated Reconnaissance & Intelligence System

I.R.I.S. is a wearable reconnaissance and intelligence system built on [Glasses model] smart glasses coupled with a Raspberry Pi 5. It enables the wearer to perform real-time facial recognition, automated OSINT collection and AI-assisted dialogue, all displayed as an augmented reality overlay directly in their field of vision, with no visible smartphone or computer.

## How does it work?

I.R.I.S. runs two parallel pipelines that converge on the Raspberry Pi 5, which acts as the central orchestrator.
### Video pipeline
The embedded camera captures a live feed streamed to the Raspberry Pi. YOLOv13 detects and identifies faces in real time. Once a face is detected, a script automatically queries facial recognition engines (PimEyes, FaceCheck.ID) and open-source intelligence platforms. The raw OSINT results are then parsed and summarized by a language model into a structured profile of the target, which is displayed on the glasses.
### Audio pipeline
The microphones of the glasses capture the interlocutor's speech. Gemma 4 agent processes the transcription in real time and generates short contextual dialogue suggestions (10–15 words) to assist the wearer.

## Getting Started

### Installation

[Explain how to install all of the project's dependencies]

### Quickstart

[Explain how to run this project]

### Usage

[Explain how to use this project]

## Get involved

You're invited to join this project ! Check out the [contributing guide](./CONTRIBUTING.md).

If you're interested in how the project is organized at a higher level, please contact the current project manager.

## Our PoC team ❤️

Developers
| [<img src="https://github.com/MrZalTy.png?size=85" width=85><br><sub>Cheikh F.</sub>](https://github.com/MrZalTy) | [<img src="https://github.com/MrZalTy.png?size=85" width=85><br><sub>Hesham C.</sub>](https://github.com/MrZalTy) | [<img src="https://github.com/MrZalTy.png?size=85" width=85><br><sub>Marwan B.</sub>](https://github.com/MrZalTy)
| :---: | :---: | :---: |

Manager
| [<img src="https://github.com/t1m0t-p.png?size=85" width=85><br><sub>Timothée P-B.</sub>](https://github.com/adrienfort)
| :---: |

<h2 align=center>
Organization
</h2>

<p align='center'>
    <a href="https://www.linkedin.com/company/pocinnovation/mycompany/">
        <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn logo">
    </a>
    <a href="https://www.instagram.com/pocinnovation/">
        <img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white" alt="Instagram logo"
>
    </a>
    <a href="https://twitter.com/PoCInnovation">
        <img src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter logo">
    </a>
    <a href="https://discord.com/invite/Yqq2ADGDS7">
        <img src="https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white" alt="Discord logo">
    </a>
</p>
<p align=center>
    <a href="https://www.poc-innovation.fr/">
        <img src="https://img.shields.io/badge/WebSite-1a2b6d?style=for-the-badge&logo=GitHub Sponsors&logoColor=white" alt="Website logo">
    </a>
</p>

> 🚀 Don't hesitate to follow us on our different networks, and put a star 🌟 on `PoC's` repositories

> Made with ❤️ by PoC
