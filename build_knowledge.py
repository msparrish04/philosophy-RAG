import requests

BASE = 'https://philosophersapi.com/api'

print('Fetching philosophers...')
philosophers = requests.get(f'{BASE}/philosophers').json()
name_by_id = {p['id']: p['name'] for p in philosophers}
print(f'  {len(name_by_id)} philosophers loaded')

print('Fetching quotes...')
quotes = requests.get(f'{BASE}/quotes').json()
print(f'  {len(quotes)} quotes loaded')

print('Fetching key ideas...')
key_ideas = requests.get(f'{BASE}/keyideas').json()
print(f'  {len(key_ideas)} key ideas loaded')

lines = []

for q in quotes:
    philosopher_id = (q.get('philosopher') or {}).get('id')
    name = name_by_id.get(philosopher_id, 'Unknown philosopher')
    work = q.get('work')
    year = q.get('year')
    source = f' ({work}, {year})' if work and year else f' ({work})' if work else ''
    quote_text = q.get('quote', '').strip()
    if quote_text:
        lines.append(f'{name} said: "{quote_text}"{source}')

for k in key_ideas:
    philosopher_id = (k.get('philosopher') or {}).get('id')
    name = name_by_id.get(philosopher_id, 'Unknown philosopher')
    text = k.get('text', '').strip()
    if text:
        lines.append(f'{name}\'s key idea: {text}')

with open('philosophy-notes.txt', 'w') as f:
    for line in lines:
        f.write(line + '\n')

print(f'\nWrote {len(lines)} total lines to philosophy-notes.txt')