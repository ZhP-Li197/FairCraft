"""
AgentFGL Configuration File
"""

import os


class Config:
    """AgentFGL global configuration"""

    # API Configuration
    API_KEY = os.environ.get("OPENAI_API_KEY", "")
    BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = "gpt-4o-mini"

    # Path Configuration
    DATA_DIR = os.getcwd()
    INPUT_CSV = "person.csv"
    INPUT_EDGES = "edges.txt"

    # Output files
    OUTPUT_CSV = "person_agentfgl.csv"
    OUTPUT_EDGES = "edges_agentfgl.txt"
    OUTPUT_GRAPH_PKL = "graph_agentfgl.pkl"
    OUTPUT_LOG = "agentfgl_log.json"
    CHECKPOINT_FILE = "agentfgl_checkpoint.pkl"

    # Stage 1: Multi-Agent Text Refinement
    VERIFICATION_THRESHOLD_PRES = 0.7
    VERIFICATION_THRESHOLD_NEUT = 0.7
    MAX_REFINEMENT_ROUNDS = 3

    # LLM parameters
    TEMPERATURE_DIAGNOSIS = 0.2
    TEMPERATURE_PLANNING = 0.3
    TEMPERATURE_GENERATION = 0.3
    TEMPERATURE_VERIFICATION = 0.2

    # Text processing
    MAX_TEXT_LENGTH = 10000
    MAX_PROMPT_TOKENS = 4000

    # Stage 2: Topology Debiasing
    BERT_MODEL_NAME = "bert-base-uncased"
    BERT_BATCH_SIZE = 32
    BERT_MAX_LENGTH = 512
    DEVICE = "cuda"

    # Candidate retrieval
    K_MINUS = 10
    K_PLUS = 10

    # LLM edge reasoning
    CONFIDENCE_THRESHOLD_REMOVE = 0.6
    CONFIDENCE_THRESHOLD_ADD = 0.6

    # Graph editing budget
    EDIT_BUDGET_RATIO = 0.1

    # Concurrency and performance
    MAX_WORKERS = 30
    CHECKPOINT_INTERVAL = 100

    # LLM retry
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    # Dataset split
    TRAIN_RATIO = 0.5
    SENS_NUMBER = 200
    RANDOM_SEED = 42

    # Logging and debugging
    LOG_LEVEL = "INFO"
    SAVE_INTERMEDIATE = True
    VERBOSE = True

    @classmethod
    def get_input_csv_path(cls):
        return os.path.join(cls.DATA_DIR, cls.INPUT_CSV)

    @classmethod
    def get_input_edges_path(cls):
        return os.path.join(cls.DATA_DIR, cls.INPUT_EDGES)

    @classmethod
    def get_output_csv_path(cls):
        return os.path.join(cls.DATA_DIR, cls.OUTPUT_CSV)

    @classmethod
    def get_output_edges_path(cls):
        return os.path.join(cls.DATA_DIR, cls.OUTPUT_EDGES)

    @classmethod
    def get_output_graph_pkl_path(cls):
        return os.path.join(cls.DATA_DIR, cls.OUTPUT_GRAPH_PKL)

    @classmethod
    def get_output_log_path(cls):
        return os.path.join(cls.DATA_DIR, cls.OUTPUT_LOG)

    @classmethod
    def get_checkpoint_path(cls):
        return os.path.join(cls.DATA_DIR, cls.CHECKPOINT_FILE)

    @classmethod
    def validate(cls):
        if not cls.API_KEY:
            raise ValueError("Please set environment variable OPENAI_API_KEY")

        if not os.path.exists(cls.DATA_DIR):
            raise ValueError(f"Data directory does not exist: {cls.DATA_DIR}")

        if not os.path.exists(cls.get_input_csv_path()):
            raise ValueError(f"Input CSV file does not exist: {cls.get_input_csv_path()}")

        if not os.path.exists(cls.get_input_edges_path()):
            raise ValueError(f"Input edges file does not exist: {cls.get_input_edges_path()}")

        print("✅ Configuration validated successfully")


config = Config()
