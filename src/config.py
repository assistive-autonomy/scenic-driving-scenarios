from dataclasses import dataclass


@dataclass
class WandbConfig:
    use_wandb: bool = False
    project: str = "CDSG-experiment"
    entity: str = "ipab-rad"


@dataclass
class DataConfig:
    dataset_name: str = "ipab-rad/driving_scenarios"


@dataclass
class ExceptionConfig:
    do_feeding: bool = False
    max_trials: int = 0


@dataclass
class ModelConfig:
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf"
    retriever_model_name: str = "BAAI/bge-small-en-v1.5"
    client: str = "http://goya:8080"
    random_examples: bool = False
    retrieval_top_k: int = 3
    decoding_top_k: int = 50
    temperature: float = 1.0
    do_sample: bool = False
    num_beams: int = 1
    max_new_tokens: int = 1200
    exceptions: ExceptionConfig = ExceptionConfig()


@dataclass
class ConvAssetsConfig:
    user: str = "assets/app_data/user.png"
    bot: str = "assets/app_data/bot.png"


@dataclass
class ConvScenariosConfig:
    path: str = "assets/app_data"
    bypassing: int = 2
    intersection: int = 2
    pedestrian: int = 2


@dataclass
class ConvConfig:
    assets: ConvAssetsConfig = ConvAssetsConfig()
    scenarios: ConvScenariosConfig = ConvScenariosConfig()
    max_instructions: int = 5
    share: bool = False


@dataclass
class ExpConfig:
    seed: int = 42
    wandb: WandbConfig = WandbConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    conv: ConvConfig = ConvConfig()

