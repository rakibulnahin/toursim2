import os

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


class AbstractLLM:
    def __init__(self):
        self.llm = None
        if Llama is None:
            print("[AbstractLLM] llama_cpp not installed. Operating in fallback mode.")
            return

        try:
            print("[AbstractLLM] Initializing...")
            gpu_layers = int(os.getenv("N_GPU_LAYERS", "0"))
            threads = max(4, os.cpu_count() or 4)
            self.llm = Llama.from_pretrained(
                repo_id="ggml-org/gemma-4-E2B-it-GGUF",
                filename="gemma-4-E2B-it-Q4_0.gguf",
                n_gpu_layers=gpu_layers,
                n_threads=threads,
                n_ctx=4096,
                verbose=False,
            )
            print(f"[AbstractLLM] Initialized successfully with {threads} CPU threads and {gpu_layers} GPU layers!")
        except Exception as e:
            print(f"[AbstractLLM] Failed to initialize: {e}")
            self.llm = None

    def call_llm(self, SYSTEM_PROMPT, USER_PROMPT, max_tokens=512):
        if self.llm is None:
            raise RuntimeError("LLM is not initialized")
        return self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=max_tokens,
        )