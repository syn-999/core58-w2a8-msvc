# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

try:
    import torch
except ImportError:
    print(
        "Missing dependency 'torch'. Activate `venv_gpu` and install the CUDA-enabled "
        "PyTorch wheel documented in README.md before using gpu_generate.py."
    )
    sys.exit(1)

try:
    import fire
except ImportError:
    print("Missing dependency 'fire'. Activate your repo venv or run `pip install -r requirements.txt` before using gpu_generate.py.")
    sys.exit(1)

try:
    import readline  # type: ignore # noqa: F401
except ImportError:
    readline = None

THIS_DIR = Path(__file__).resolve().parent
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))

try:
    try:
        import gpu_model as fast
        from stats import Stats
        from tokenizer import Tokenizer, ChatFormat
        import sample_utils
    except ImportError:
        from . import gpu_model as fast
        from .stats import Stats
        from .tokenizer import Tokenizer, ChatFormat
        from . import sample_utils
except Exception as exc:
    print(exc)
    sys.exit(1)


@dataclass
class GenArgs:
    gen_length: int = 2048
    gen_bsz: int = 1
    prompt_length: int = 64

    use_sampling: bool = False
    temperature: float = 0.8
    top_p: float = 0.9


class FastGen:
    GRAPH_WARMUPS: int = 1
    EARLY_STOP_POLL_INTERVAL: int = 4
    tokenizer: Tokenizer

    @staticmethod
    def build(
        ckpt_dir: str,
        gen_args: GenArgs,
        device: Union[torch.device, str],
        tokenizer_path: Optional[str] = None,
        num_layers: int = 13,
        use_full_vocab: bool = False,
        decode_backend: str = "int2",
    ) -> "FastGen":
        """
        Load a Llama or Code Llama checkpoint and return a new
        generator for this model.
        """
        start_time = time.time()

        if decode_backend not in {"fp16", "int2"}:
            raise ValueError(f"Unsupported decode backend: {decode_backend}")

        validate_checkpoint_dir(ckpt_dir, decode_backend)

        model_args_prefill = fast.ModelArgs(use_kernel=False)
        model_args_decode = fast.ModelArgs(use_kernel=(decode_backend == "int2"))
        tokenizer = Tokenizer(str(tokenizer_path or (THIS_DIR / "tokenizer.model")))

        torch.set_default_device(device)
        torch.set_default_dtype(torch.bfloat16)

        prefill_model = fast.Transformer(model_args_prefill)
        decode_model = prefill_model if decode_backend == "fp16" else fast.Transformer(model_args_decode)

        fp16_ckpt_path = str(Path(ckpt_dir) / "model_state_fp16.pt")
        fp16_checkpoint = torch.load(fp16_ckpt_path, map_location="cpu", weights_only=True)

        prefill_model.load_state_dict(fp16_checkpoint, strict=True)
        if decode_backend == "int2":
            int2_ckpt_path = str(Path(ckpt_dir) / "model_state_int2.pt")
            decode_checkpoint = torch.load(int2_ckpt_path, map_location="cpu", weights_only=True)
            decode_model.load_state_dict(decode_checkpoint, strict=True)

        torch.cuda.synchronize()
        print(f"loaded model in {time.time() - start_time:.2f} seconds")
        start_time = time.time()

        return FastGen(gen_args, model_args_prefill, prefill_model, decode_model, tokenizer)

    def __init__(
        self,
        args: GenArgs,
        model_args: fast.ModelArgs,
        prefill_model: fast.Transformer,
        decode_model: fast.Transformer,
        tokenizer: Tokenizer,
    ):
        self.gen_args = args
        self.max_seq_length = args.prompt_length + args.gen_length
        self.model_args = model_args
        # self.model = model
        self.prefill_model = prefill_model
        self.decode_model = decode_model
        self.tokenizer = tokenizer
        self._prefill_cuda_graph, self._prefill_compile_model, self._prefill_inputs, self._prefill_logits = None, None, None, None
        self._generate_cuda_graph, self._generate_compile_model, self._generate_inputs, self._generate_logits = None, None, None, None
        self._generate_token_buffer = None
        self._generate_seq_lens_buffer = None
        self._cache = None
        start_time = time.time()
        self._prefill_compile_model = self.compile_prefill()
        self._generate_compile_model = self.compile_generate()
        print(f"compiled model in {time.time() - start_time:.2f} seconds")

    def compile_prefill(self):

        if self._cache is None:
            self._cache = fast.make_cache(
                args=self.model_args,
                length=self.gen_args.gen_bsz * self.max_seq_length,
            )

        seq_lens = [self.gen_args.prompt_length for _ in range(self.gen_args.gen_bsz)]

        bias = None

        tokens = torch.IntTensor([[1] * self.gen_args.prompt_length for _ in range(self.gen_args.gen_bsz)]).cuda()
        self._prefill_inputs = (tokens, None)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(s):
            _ = self.prefill_model.forward_with_attn_bias(
                token_values=self._prefill_inputs[0],
                attn_bias=self._prefill_inputs[1],
                cache=self._cache,
            )
        torch.cuda.current_stream().wait_stream(s)

        self._prefill_cuda_graph = torch.cuda.CUDAGraph()
        recording_kwargs = {}
        if "capture_error_mode" in torch.cuda.graph.__init__.__annotations__:
            # In PyTorch 2.1+ and nightlies from late Aug 2023,
            # we can do this to maybe avoid watchdog-related crashes
            recording_kwargs["capture_error_mode"] = "thread_local"
        with torch.cuda.graph(self._prefill_cuda_graph, **recording_kwargs):
            self._prefill_logits = self.prefill_model.forward_with_attn_bias(
                token_values=self._prefill_inputs[0],
                attn_bias=self._prefill_inputs[1],
                cache=self._cache,
            )

        def replay(tokens, seq_lens=None):
            self._prefill_inputs[0].copy_(tokens)
            # if seq_lens is not None:
            #     self._prefill_inputs[1].k_seqinfo.seqlen.copy_(seq_lens)

            self._prefill_cuda_graph.replay()
            return self._prefill_logits

        return replay

    def compile_generate(self):

        if self._cache is None:
            self._cache = fast.make_cache(
                args=self.model_args,
                length=self.gen_args.gen_bsz * self.max_seq_length,
            )

        seq_lens = [1 for _ in range(self.gen_args.gen_bsz)]
        kv_seq_lens = [self.gen_args.prompt_length for _ in range(self.gen_args.gen_bsz)]

        tokens = torch.IntTensor([[1] for _ in range(self.gen_args.gen_bsz)]).cuda()
        seq_lens = torch.IntTensor([self.gen_args.prompt_length for _ in range(self.gen_args.gen_bsz)]).cuda()
        self._generate_inputs = (tokens, seq_lens)
        self._generate_token_buffer = self._generate_inputs[0].view(-1)
        self._generate_seq_lens_buffer = self._generate_inputs[1]

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(s):
            _ = self.decode_model.forward_with_attn_bias(
                token_values=self._generate_inputs[0],
                attn_bias=self._generate_inputs[1],
                cache=self._cache,
            )
        torch.cuda.current_stream().wait_stream(s)

        self._generate_cuda_graph = torch.cuda.CUDAGraph()
        recording_kwargs = {}
        if "capture_error_mode" in torch.cuda.graph.__init__.__annotations__:
            # In PyTorch 2.1+ and nightlies from late Aug 2023,
            # we can do this to maybe avoid watchdog-related crashes
            recording_kwargs["capture_error_mode"] = "thread_local"
        with torch.cuda.graph(self._generate_cuda_graph, **recording_kwargs):
            self._generate_logits = self.decode_model.forward_with_attn_bias(
                token_values=self._generate_inputs[0],
                attn_bias=self._generate_inputs[1],
                cache=self._cache,
            )

        def replay(tokens=None, seq_lens=None):
            if tokens is not None:
                self._generate_inputs[0].copy_(tokens)
            if seq_lens is not None:
                self._generate_inputs[1].copy_(seq_lens)

            self._generate_cuda_graph.replay()

            return self._generate_logits

        return replay


    @torch.inference_mode()
    def generate_all(
        self,
        prompts: list[list[int]],
        use_cuda_graphs: bool,
        use_sampling: bool,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[Stats, list[list[int]]]:
        bs = len(prompts)
        prompt_lens = [len(p) for p in prompts]
        if any(prompt_len > self.gen_args.prompt_length for prompt_len in prompt_lens):
            raise ValueError(
                f"Prompt length exceeds configured limit of {self.gen_args.prompt_length} tokens. "
                "Increase prompt_length and rebuild the generator."
            )
        padded_prompt_lens = [self.gen_args.prompt_length] * bs
        max_prompt_length = max(prompt_lens)
        gen_length = max_new_tokens if max_new_tokens is not None else self.gen_args.gen_length
        if gen_length <= 0:
            raise ValueError("max_new_tokens must be at least 1")
        max_seq_length = max_prompt_length + gen_length
        if max_seq_length > self.max_seq_length:
            raise ValueError(
                f"Requested sequence length {max_seq_length} exceeds compiled limit {self.max_seq_length}. "
                "Increase prompt_length or gen_length and rebuild the generator."
            )
        temperature = self.gen_args.temperature if temperature is None else temperature
        top_p = self.gen_args.top_p if top_p is None else top_p
        # bias = AttnBias.from_seqlens(
        #     q_seqlen=padded_prompt_lens,
        #     kv_seqlen=prompt_lens,
        #     kv_padding=max_seq_length,
        # )
        # bias.q_seqinfo.to("cuda")
        # bias.k_seqinfo.to("cuda")

        # Input tensors to the cuda graph
        # kv_seqlen = bias.k_seqinfo.seqlen
        kv_seqlen = torch.tensor(prompt_lens, dtype=torch.int32, device="cuda")
        prompts = [prompt + [1] * (self.gen_args.prompt_length - len(prompt)) for prompt in prompts]
        tokens = torch.IntTensor(prompts).cuda()
        single_sequence = bs == 1
        out_tokens = torch.empty(
            gen_length if single_sequence else (gen_length, bs),
            dtype=torch.long,
            device="cuda",
        )

        stats = Stats()
        torch.cuda.synchronize()
        stats.phase("prefill" if use_cuda_graphs else "total")
        # stats.phase("total")

        output = self._prefill_compile_model(tokens, None)

        logits = output[torch.arange(bs), kv_seqlen - 1, :]
        logits = logits.view(bs, self.model_args.vocab_size)

        if use_sampling:
            probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
            next_token = sample_utils.top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)        

        next_token = next_token.reshape(bs)
        if single_sequence:
            out_tokens[0] = next_token[0]
        else:
            out_tokens[0, :] = next_token

        torch.cuda.synchronize()
        stats.phase("decode" if use_cuda_graphs else "total")

        stop_tokens = set(getattr(self.tokenizer, "stop_tokens", {self.tokenizer.eos_id}))
        stop_token_ids = torch.tensor(
            sorted(stop_tokens),
            device=next_token.device,
            dtype=next_token.dtype,
        )
        all_finished_device = torch.empty((), device=next_token.device, dtype=torch.bool)
        all_finished_host = torch.empty((), dtype=torch.bool, device="cpu", pin_memory=True)
        all_finished_event = torch.cuda.Event()
        steps_written = 1

        def schedule_finished_poll(token_status: torch.Tensor) -> None:
            token_status = token_status.reshape(())
            all_finished_device.copy_(token_status)
            all_finished_host.copy_(all_finished_device, non_blocking=True)
            all_finished_event.record(torch.cuda.current_stream())

        def token_is_stop(token_values: torch.Tensor) -> torch.Tensor:
            if stop_token_ids.numel() == 1:
                return token_values.eq(stop_token_ids[0])
            return token_values.unsqueeze(-1).eq(stop_token_ids).any(dim=-1)

        if single_sequence:
            decode_token_buffer = self._generate_token_buffer
            decode_seq_lens = self._generate_seq_lens_buffer
            decode_seq_lens.fill_(prompt_lens[0])
            schedule_finished_poll(token_is_stop(next_token))
            for niter in range(1, gen_length):
                if all_finished_event.query() and bool(all_finished_host.item()):
                    break

                decode_seq_lens.add_(decode_seq_lens < max_seq_length)
                decode_token_buffer.copy_(next_token)
                output = self._generate_compile_model()

                logits = output.view(1, self.model_args.vocab_size)

                if use_sampling:
                    probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
                    next_token = sample_utils.top_p(probs, top_p)
                else:
                    next_token = torch.argmax(logits, dim=-1)

                next_token = next_token.reshape(1)
                out_tokens[niter] = next_token[0]
                steps_written = niter + 1
                if (steps_written % self.EARLY_STOP_POLL_INTERVAL) == 0:
                    schedule_finished_poll(token_is_stop(next_token))
        else:
            inactive_token = torch.full_like(next_token, self.tokenizer.eos_id)
            finished = token_is_stop(next_token)
            schedule_finished_poll(torch.all(finished))
            for niter in range(1, gen_length):
                if all_finished_event.query() and bool(all_finished_host.item()):
                    break

                active = ~finished
                kv_seqlen.add_(
                    active.to(kv_seqlen.dtype) * (kv_seqlen < max_seq_length).to(kv_seqlen.dtype)
                )
                decode_input = torch.where(
                    active,
                    next_token,
                    inactive_token,
                )
                output = self._generate_compile_model(decode_input.unsqueeze(1), kv_seqlen)

                logits = output.view(bs, self.model_args.vocab_size)

                if use_sampling:
                    probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
                    next_token = sample_utils.top_p(probs, top_p)
                else:
                    next_token = torch.argmax(logits, dim=-1)

                next_token = next_token.reshape(bs)
                next_token = torch.where(active, next_token, decode_input)
                out_tokens[niter, :] = next_token
                finished |= token_is_stop(next_token)
                steps_written = niter + 1
                if (steps_written % self.EARLY_STOP_POLL_INTERVAL) == 0:
                    schedule_finished_poll(torch.all(finished))

        torch.cuda.synchronize()
        generated_tokens = 0

        def trim_answer(tokens):
            """Trim the answer to end it on an eos token."""
            for i, token in enumerate(tokens):
                if token in stop_tokens:
                    return tokens[:i], i + 1
            return tokens, len(tokens)

        if single_sequence:
            answers = [trim_answer(out_tokens[:steps_written].cpu().tolist())]
        else:
            answers = [
                trim_answer(answer)
                for answer in out_tokens[:steps_written, :].transpose(0, 1).cpu().tolist()
            ]
        generated_tokens = sum(consumed_tokens for _, consumed_tokens in answers)
        stats.end_phase(tokens=generated_tokens)
        answers = [answer for answer, _ in answers]
        return stats, answers


def validate_checkpoint_dir(ckpt_dir: str, decode_backend: str) -> None:
    ckpt_root = Path(ckpt_dir)
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_root}")

    required_files = [ckpt_root / "model_state_fp16.pt"]
    if decode_backend == "int2":
        required_files.append(ckpt_root / "model_state_int2.pt")
    missing_files = [path.name for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Checkpoint directory {ckpt_root} is missing required files: {', '.join(missing_files)}"
        )


def get_prompts(interactive: bool) -> Iterable[list[str]]:
    if interactive:
        while True:
            try:
                raw_prompt = input("enter prompt: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("exiting")
                sys.exit(0)
            if not raw_prompt:
                continue
            prompts = [prompt for prompt in raw_prompt.split("\n") if prompt.strip()]
            if not prompts:
                continue
            yield prompts
    else:
        yield [
            "Hello, my name is",
        ]


def main(
    ckpt_dir: str,
    max_new_tokens: int = 50,
    interactive: bool = False,
    chat_format: bool = False,
    sampling: bool = False,
    decode_backend: str = "int2",
    prompt_length: int = 64,
):

    local_rank = 0
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)

    try:
        g = FastGen.build(
            ckpt_dir,
            GenArgs(gen_length=max_new_tokens, prompt_length=prompt_length),
            device,
            decode_backend=decode_backend,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(exc)
        sys.exit(1)

    if chat_format:
        g.tokenizer = ChatFormat(g.tokenizer)

    for prompts in get_prompts(interactive):
        # prompts = [f"{prompt}\n" for prompt in prompts]
        if chat_format:
            # prompts = [f'<|begin_of_text|>User: {prompt}<|eot_id|>Assistant: ' for prompt in prompts]
            tokens = [g.tokenizer.encode_dialog_prompt(dialog=[{"role": "user", "content": prompt}], completion=True) for prompt in prompts]
        else:
            tokens = [g.tokenizer.encode(x, bos=False, eos=False) for x in prompts]

        stats, out_tokens = g.generate_all(
            tokens, use_cuda_graphs="NO_CUDA_GRAPHS" not in os.environ, use_sampling=sampling,
        )

        for i, prompt in enumerate(prompts):
            print(f"> {prompt}")
            answer = g.tokenizer.decode(out_tokens[i])
            print(answer)
            print("---------------")

        for phase_stats in stats.phases:
            print(phase_stats.show())

        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == "__main__":
    fire.Fire(main)
