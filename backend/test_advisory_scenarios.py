import sys
sys.path.insert(0, '/mnt/C6EE65A1EE658B0F/WORKEST/Agro-AI/FarmHand/backend')
import time
from llm_engine import chat_completion

print('========================================')
print('TEST 1: Flock Integration (Husbandry)')
print('========================================')
t0 = time.time()
q1 = 'I want to add a new chicken to my coop. what steps should i take to ensure proper integration with not problem'
res1 = chat_completion([{'role': 'user', 'content': q1}], farm_id='default_farm', language='english')
t1 = time.time()
print(f'Done in {t1-t0:.2f}s:\n{res1}\n')

assert 'newcastle' not in res1.lower(), 'Should NOT hallucinate Newcastle Disease for flock integration'
assert any(k in res1.lower() for k in ['quarantine', 'isolate', 'separate', 'visual', 'pen']), 'Should mention practical integration steps'
assert not res1.startswith('['), 'Must not be JSON'
assert ' dey ' not in res1.lower() and 'na ' not in res1.lower() and 'wetin' not in res1.lower(), 'Must not contain Pidgin'

print('========================================')
print('TEST 2: Clinical Diagnosis (Turn 1)')
print('========================================')
t0 = time.time()
q2 = 'I see some of my chicken stumbling backwards and looking lost or confused... what could be wrong'
res2 = chat_completion([{'role': 'user', 'content': q2}], farm_id='default_farm', language='english')
t1 = time.time()
print(f'Done in {t1-t0:.2f}s:\n{res2}\n')

assert not res2.startswith('['), 'Must not be JSON'
assert ' dey ' not in res2.lower() and 'na ' not in res2.lower() and 'wetin' not in res2.lower(), 'Must not contain Pidgin'

print('========================================')
print('TEST 3: Follow-Up Treatment (Turn 2)')
print('========================================')
t0 = time.time()
msgs3 = [
    {'role': 'user', 'content': q2},
    {'role': 'assistant', 'content': res2},
    {'role': 'user', 'content': 'So whats the treatment?'}
]
res3 = chat_completion(msgs3, farm_id='default_farm', language='english')
t1 = time.time()
print(f'Done in {t1-t0:.2f}s:\n{res3}\n')

assert not res3.startswith('['), 'Must not be JSON'
assert ' dey ' not in res3.lower() and 'na ' not in res3.lower() and 'wetin' not in res3.lower(), 'Must not contain Pidgin'
assert 'goat' not in res3.lower(), 'Must not hallucinate goats when discussing poultry'
assert len(res3) > 60, 'Must provide structured treatment steps'

print('[ALL TESTS PASSED 100%]')
