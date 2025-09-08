# Conversational Code Generation: a Case Study of Designing a Dialogue System for Generating Driving Scenarios for Testing Autonomous Vehicles


[paper](https://arxiv.org/abs/2410.09829) | [dataset](https://huggingface.co/datasets/assistive-autonomy/scenic-driving-scenarios) 

<img src="assets/overview.png"/> 

Cyber-physical systems like autonomous vehicles are tested in simulation before deployment, using domain-specific programs for scenario specification. To aid the testing of autonomous vehicles in simulation, we design a natural language interface, using an instruction-following large language model, to assist a non-coding domain expert in synthesising the desired scenarios and vehicle behaviours. We show that using it to convert utterances to the symbolic program is feasible, despite the very small training dataset. Human experiments show that dialogue is critical to successful simulation generation, leading to a 4.5 times higher success rate than a generation without engaging in extended conversation.


## Setup

Install [CARLA 0.9.15](https://carla.org/) in your home directory

```bash
pip install -r requirements.txt 
```

## Experiments

Automatic Evaluation

```bash
python scripts/rag.py
```

Human Evaluation

```bash
python scripts/app.py
```


## Citation

```
@inproceedings{rubavicius2025conversationalcodegenerationcase,
title = {Conversational Code Generation: a Case Study of Designing a Dialogue System for Generating Driving Scenarios for Testing Autonomous Vehicles},
author = {Rimvydas Rubavicius and Antonio Valerio Miceli-Barone and Alex Lascarides and Subramanian Ramamoorthy},
url = {https://arxiv.org/abs/2410.09829},
year = {2025},
date = {2025-09-08},
booktitle = {Proceedings of GeCoIn 2025: Generative Code Intelligence Workshop, co-located with the 28th European Conference on Artificial Intelligence
(ECAI-2025)
}
```