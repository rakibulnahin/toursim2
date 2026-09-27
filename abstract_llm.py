from llama_cpp import Llama


class AbstractLLM:
    def __init__(self):
        self.llm = None
        try:
            print("[AbstracLLM] Initializing...")
            self.llm = Llama.from_pretrained(
                repo_id="ggml-org/gemma-4-E2B-it-GGUF",
                filename="gemma-4-E2B-it-Q4_0.gguf",
                n_gpu_layers=0,
                n_threads=4,
                n_ctx=2048,
                verbose=False,
            )
            print("[AbstracLLM] Initialized successfully!")
        except Exception as e:
            print(f"[AbstracLLM] Failed to initialize: {e}")
            self.llm = None

    def call_llm(self, SYSTEM_PROMPT, USER_PROMPT):
        return self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )