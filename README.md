# Generating Driving Simulations via Conversation


<img src="assets/overview.png" width="500" /> 

Cyber-physical systems like autonomous vehicles are tested in simulation before deployment, using domain-specific programs for scenario specification. To aid the testing of autonomous vehicles in simulation, we design a natural language interface, using an instruction-following large language model, to assist a non-coding domain expert in synthesising the desired scenarios and vehicle behaviours. We show that using it to convert utterances to the symbolic program is feasible, despite the very small training dataset. Human experiments show that *dialogue* is critical to successful simulation generation, leading to a 4.5 times higher success rate than a generation without engaging in extended conversation.


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


## References

```
@misc{rubavicius2024generatingdrivingsimulationsconversation,
      title={Generating Driving Simulations via Conversation}, 
      author={Rimvydas Rubavicius and Antonio Valerio Miceli-Barone and Alex Lascarides and Subramanian Ramamoorthy},
      year={2024},
      eprint={2410.09829},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.09829}, 
}
```