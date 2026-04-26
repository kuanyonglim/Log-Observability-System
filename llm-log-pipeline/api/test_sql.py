'''
This script is created for testing of the API functionality. 
To run this test, simply execute `python test_sql.py` in the terminal 
while being in the `api` directory.
'''
from nl_to_sql import validate_sql

print("Running SQL Validation Tests...\n")

# Test 1: Valid SELECT
ok, err = validate_sql("SELECT * FROM log_events LIMIT 10")
assert ok, f"Should be valid: {err}"
print("✅ Valid SELECT accepted")

# Test 2: Reject DROP
ok, err = validate_sql("DROP TABLE log_events")
assert not ok
print(f"✅ DROP rejected: {err}")

# Test 3: Reject injection attempt
ok, err = validate_sql("SELECT * FROM log_events; DELETE FROM log_events--")
assert not ok
print(f"✅ Injection rejected: {err}")

# Test 4: Reject UPDATE
ok, err = validate_sql("UPDATE log_events SET level = 'ERROR'")
assert not ok
print(f"✅ UPDATE rejected: {err}")

# Test 5: CANNOT_ANSWER sentinel
ok, err = validate_sql("CANNOT_ANSWER")
assert not ok and err == "CANNOT_ANSWER"
print("✅ CANNOT_ANSWER handled correctly")

print("\n🎉 All SQL validation tests passed!")