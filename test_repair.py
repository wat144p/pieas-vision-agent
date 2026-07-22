import json
from src.vision import repair_json

test_cases = [
    '{"people_count": 10+, "other": "field"}',
    '{"people_count": "10+", "other": "field"}',
    '{"people_count": 10+ (as there are multiple), "other": "field"}',
    '{"people_count": about 10, "other": "field"}',
]

print("Testing repair_json on problematic cases...")
for test in test_cases:
    print(f"\nInput: {test}")
    repaired = repair_json(test)
    print(f"Output: {repaired}")
    try:
        parsed = json.loads(repaired)
        print(f"✅ SUCCESS! people_count = {parsed['people_count']}")
    except Exception as e:
        print(f"❌ Failed: {e}")
