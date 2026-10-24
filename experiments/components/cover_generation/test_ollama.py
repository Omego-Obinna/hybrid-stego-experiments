import ollama

resp = ollama.generate(
    model='phi3:mini',
    prompt='Give one fluent sentence about network steganography.',
    options={'seed': 42, 'temperature': 0.7},
    stream=False
)
print(resp['response'])

