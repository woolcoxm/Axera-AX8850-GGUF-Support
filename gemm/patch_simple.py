#!/usr/bin/env python3
"""Insert BOS fallback for empty prompts in llama-simple."""
import sys

p = sys.argv[1] if len(sys.argv) > 1 else '/home/kram/llama.cpp/examples/simple/simple.cpp'
src = open(p).read()
if 'prompt_tokens.empty()' in src:
    print('already patched')
    sys.exit(0)
needle = '        fprintf(stderr, "%s: error: failed to tokenize the prompt\\n", __func__);\n        return 1;\n    }\n'
assert needle in src, 'anchor not found'
ins = ('\n    // an empty prompt still needs at least one token for decode\n'
       '    if (prompt_tokens.empty()) {\n'
       '        prompt_tokens.push_back(llama_token_bos(model));\n'
       '    }\n')
open(p, 'w').write(src.replace(needle, needle + ins))
print('patched OK')
