#!/usr/bin/env python3
"""Tiny zero-dependency client for the robot's llama-server (OpenAI-compatible).

Library use from the Hiwonder apps:

    from llm import LLM
    llm = LLM()                      # LLM_SERVER_HOST env or 127.0.0.1:8080
    answer = llm.ask("Give me a short greeting")          # blocking, str
    for tok in llm.ask_stream("Tell me a haiku"): ...     # streaming

CLI use (via llm-ask):

    llm-ask "what can you see with your camera?"
    llm-ask                      # interactive chat with history
"""
import json
import os
import sys
import urllib.request

DEFAULT_HOST = os.environ.get("LLM_SERVER_HOST", "127.0.0.1:8080")


class LLM:
    def __init__(self, host=None, system=None, max_tokens=256, temperature=0.7, think=False):
        self.host = (host or DEFAULT_HOST).rstrip("/")
        self.system = system
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.think = think  # Qwen3.5 reasons by default; a robot wants instant replies

    # ------------------------------------------------------------------
    def _payload(self, messages, max_tokens, temperature, stream):
        return json.dumps({
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "stream": bool(stream),
            "chat_template_kwargs": {"enable_thinking": self.think},
        }).encode()

    def _post(self, payload, stream=False):
        req = urllib.request.Request(
            f"http://{self.host}/v1/chat/completions", data=payload,
            headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=300)

    def _messages(self, prompt, system=None):
        msgs = []
        s = system if system is not None else self.system
        if s:
            msgs.append({"role": "system", "content": s})
        if isinstance(prompt, str):
            msgs.append({"role": "user", "content": prompt})
        else:
            msgs.extend(prompt)
        return msgs

    # ------------------------------------------------------------------
    def ask(self, prompt, system=None, max_tokens=None, temperature=None):
        """Blocking call → plain str."""
        body = self._payload(self._messages(prompt, system),
                             max_tokens, temperature, stream=False)
        with self._post(body) as r:
            data = json.load(r)
        msg = data["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning_content") or ""

    def ask_stream(self, prompt, system=None, max_tokens=None, temperature=None):
        """Generator of str deltas."""
        body = self._payload(self._messages(prompt, system),
                             max_tokens, temperature, stream=True)
        with self._post(body, stream=True) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    return
                try:
                    delta = json.loads(chunk)["choices"][0]["delta"]
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
                tok = delta.get("content") or delta.get("reasoning_content")
                if tok:
                    yield tok

    def health(self):
        try:
            with urllib.request.urlopen(f"http://{self.host}/health", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False


# ----------------------------------------------------------------------
def _cli(argv):
    import argparse
    p = argparse.ArgumentParser(prog="llm-ask", description="talk to the robot's LLM")
    p.add_argument("prompt", nargs="*", help="prompt (omitted → interactive chat)")
    p.add_argument("--host", default=None, help="host:port (default $LLM_SERVER_HOST or 127.0.0.1:8080)")
    p.add_argument("--system", default="You are the brain of a small quadruped robot called PuppyPi. Answer briefly and clearly.")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--no-stream", action="store_true")
    args = p.parse_args(argv)

    llm = LLM(host=args.host, system=args.system,
              max_tokens=args.max_tokens, temperature=args.temperature)
    if not llm.health():
        print(f"llm-server at {llm.host} is not answering — try: llm-doctor", file=sys.stderr)
        return 1

    if args.prompt:
        prompt = " ".join(args.prompt)
        if args.no_stream:
            print(llm.ask(prompt))
        else:
            for tok in llm.ask_stream(prompt):
                print(tok, end="", flush=True)
            print()
        return 0

    # interactive chat with rolling history
    print(f"chatting with {llm.host} — ctrl-D to exit")
    history = []
    while True:
        try:
            prompt = input("\nyou> ").strip()
        except EOFError:
            print()
            return 0
        if not prompt:
            continue
        history.append({"role": "user", "content": prompt})
        print("bot> ", end="", flush=True)
        reply = ""
        for tok in llm.ask_stream(history):
            reply += tok
            print(tok, end="", flush=True)
        print()
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]) or 0)
