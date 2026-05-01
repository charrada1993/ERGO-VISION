"""
Test RULA and REBA calculations against the official text-file tables.
Run from the project root:  python scratch/test_ergo_calc.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ergonomics.rula import RULACalculator
from ergonomics.reba import REBACalculator

rula = RULACalculator()
reba = REBACalculator()

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
errors = []

def check(label, got, expected):
    ok = got == expected
    print(f"  {'[PASS]' if ok else '[FAIL]'} {label}: got={got}, expected={expected}")
    if not ok:
        errors.append(f"{label}: got={got}, expected={expected}")

print("\n=== RULA UNIT TESTS ===\n")

# ── Upper arm score ────────────────────────────────────────────────────────────
# neutral (0°) → 1
check("UA: 0°", rula.score_upper_arm(0), 1)
# <20° → 2
check("UA: 15°", rula.score_upper_arm(15), 2)
# 20–45° → 2
check("UA: 30°", rula.score_upper_arm(30), 2)
check("UA: 45°", rula.score_upper_arm(45), 2)
# 45–90° → 3
check("UA: 60°", rula.score_upper_arm(60), 3)
check("UA: 90°", rula.score_upper_arm(90), 3)
# >90° → 4
check("UA: 120°", rula.score_upper_arm(120), 4)
# rotated/supported → +1
check("UA: 60° + rotated", rula.score_upper_arm(60, rotated=True), 4)
# Supported → +1 per RULA text "Bras soutenu ou charge modérée → +1"
check("UA: 60° + supported (+1)", rula.score_upper_arm(60, supported=True), 4)

# ── Lower arm ────────────────────────────────────────────────────────────────
check("LA: 60°", rula.score_lower_arm(60), 1)
check("LA: 100°", rula.score_lower_arm(100), 1)
check("LA: 45°", rula.score_lower_arm(45), 2)
check("LA: 110°", rula.score_lower_arm(110), 2)

# ── Wrist ────────────────────────────────────────────────────────────────────
check("Wrist: 0°", rula.score_wrist(0), 1)
check("Wrist: 15°", rula.score_wrist(15), 1)
check("Wrist: 20°", rula.score_wrist(20), 2)
check("Wrist: 35°", rula.score_wrist(35), 3)
# deviation +1
check("Wrist: 20° + dev=1", rula.score_wrist(20, deviation=1), 3)

# ── Group A table ─────────────────────────────────────────────────────────────
# Values from rula_text.txt Tableau 6 (wrist_twist=1 for pure wrist column)
# Row (ua,la) → col (wrist+twist index)
# (1,1): 1,2,2,3
check("GroupA (1,1,wrist=1,tw=1)", rula.group_a_table(1,1,1,1), 1)
check("GroupA (1,1,wrist=2,tw=1)", rula.group_a_table(1,1,2,1), 2)
check("GroupA (1,1,wrist=3,tw=1)", rula.group_a_table(1,1,3,1), 2)
check("GroupA (1,1,wrist=4,tw=1)", rula.group_a_table(1,1,4,1), 3)
# (2,1): 2,3,3,3
check("GroupA (2,1,wrist=1,tw=1)", rula.group_a_table(2,1,1,1), 2)
check("GroupA (2,1,wrist=2,tw=1)", rula.group_a_table(2,1,2,1), 3)
# (3,1): 3,3,4,4
check("GroupA (3,1,wrist=1,tw=1)", rula.group_a_table(3,1,1,1), 3)
check("GroupA (3,1,wrist=2,tw=1)", rula.group_a_table(3,1,2,1), 3)
check("GroupA (3,1,wrist=3,tw=1)", rula.group_a_table(3,1,3,1), 4)
# (4,2): 5,5,6,6
check("GroupA (4,2,wrist=1,tw=1)", rula.group_a_table(4,2,1,1), 5)
check("GroupA (4,2,wrist=3,tw=1)", rula.group_a_table(4,2,3,1), 6)
# Example from text: Bras=3, Avant-bras=1, Poignet/Twist=2 → 3
check("GroupA example (3,1,w=2,tw=1)", rula.group_a_table(3,1,2,1), 3)

# ── Neck ─────────────────────────────────────────────────────────────────────
check("Neck: 5°",  rula.score_neck(5),  1)
check("Neck: 10°", rula.score_neck(10), 1)
check("Neck: 15°", rula.score_neck(15), 2)
check("Neck: 20°", rula.score_neck(20), 2)
check("Neck: 25°", rula.score_neck(25), 3)
check("Neck: extension (-5°)", rula.score_neck(-5), 4)
# rotation +1
check("Neck: 15° + rotated", rula.score_neck(15, rotated=True), 3)

# ── Trunk ────────────────────────────────────────────────────────────────────
check("Trunk: 0° (droit)", rula.score_trunk(0), 1)
check("Trunk: 10°", rula.score_trunk(10), 2)
check("Trunk: 20°", rula.score_trunk(20), 2)
check("Trunk: 40°", rula.score_trunk(40), 3)
check("Trunk: 60°", rula.score_trunk(60), 3)
check("Trunk: 70°", rula.score_trunk(70), 4)
check("Trunk: ext (-10°)", rula.score_trunk(-10), 2)

# ── Group B table ─────────────────────────────────────────────────────────────
# From rula_text.txt lines 222-263  Cou(rows) × Tronc(cols)
# Row neck=1: 1,2,3,5,6,7
check("GroupB (neck=1,trunk=1)", rula.group_b_table(1,1,1), 1+1)  # +legs=1
check("GroupB (neck=1,trunk=2)", rula.group_b_table(1,2,1), 2+1)
check("GroupB (neck=1,trunk=3)", rula.group_b_table(1,3,1), 3+1)
check("GroupB (neck=1,trunk=4)", rula.group_b_table(1,4,1), 5+1)
check("GroupB (neck=1,trunk=5)", rula.group_b_table(1,5,1), 6+1)
check("GroupB (neck=1,trunk=6)", rula.group_b_table(1,6,1), 7+1)
# Row neck=2: 2,2,3,5,6,7
check("GroupB (neck=2,trunk=1)", rula.group_b_table(2,1,1), 2+1)
check("GroupB (neck=2,trunk=2)", rula.group_b_table(2,2,1), 2+1)
check("GroupB (neck=2,trunk=4)", rula.group_b_table(2,4,1), 5+1)
# Row neck=3: 3,3,4,6,7,7
check("GroupB (neck=3,trunk=1)", rula.group_b_table(3,1,1), 3+1)
check("GroupB (neck=3,trunk=3)", rula.group_b_table(3,3,1), 4+1)
check("GroupB (neck=3,trunk=4)", rula.group_b_table(3,4,1), 6+1)
# Row neck=4: 5,5,6,7,7,7
check("GroupB (neck=4,trunk=1)", rula.group_b_table(4,1,1), 5+1)
check("GroupB (neck=4,trunk=2)", rula.group_b_table(4,2,1), 5+1)
check("GroupB (neck=4,trunk=3)", rula.group_b_table(4,3,1), 6+1)
check("GroupB (neck=4,trunk=4)", rula.group_b_table(4,4,1), 7+1)
# Row neck=5: 6,6,7,7,7,7
check("GroupB (neck=5,trunk=1)", rula.group_b_table(5,1,1), 6+1)
# Row neck=6: 7,7,7,7,7,7
check("GroupB (neck=6,trunk=1)", rula.group_b_table(6,1,1), 7+1)
# Example: Cou=3, Tronc=4 → 6, + jambes=1 → 7
check("GroupB example (neck=3,trunk=4,legs=1)", rula.group_b_table(3,4,1), 7)

# ── Final RULA matrix ─────────────────────────────────────────────────────────
# Standard 7×7 matrix (not given explicitly in text, use standard RULA)
check("Final RULA (A=1,B=1)", rula.final_score(1,1), 1)
check("Final RULA (A=1,B=3)", rula.final_score(1,3), 3)
check("Final RULA (A=3,B=3)", rula.final_score(3,3), 3)
check("Final RULA (A=7,B=7)", rula.final_score(7,7), 7)

print("\n=== REBA UNIT TESTS ===\n")

# ── Trunk ─────────────────────────────────────────────────────────────────────
check("REBA Trunk: 0°", reba.score_trunk(0), 1)
check("REBA Trunk: 10°", reba.score_trunk(10), 2)
check("REBA Trunk: 20°", reba.score_trunk(20), 2)
check("REBA Trunk: 40°", reba.score_trunk(40), 3)
check("REBA Trunk: 60°", reba.score_trunk(60), 3)
check("REBA Trunk: 70°", reba.score_trunk(70), 4)
check("REBA Trunk: ext (-5°)", reba.score_trunk(-5), 2)
check("REBA Trunk: 40° + rotated", reba.score_trunk(40, rotated=True), 4)

# ── Neck ─────────────────────────────────────────────────────────────────────
check("REBA Neck: 10°", reba.score_neck(10), 1)
check("REBA Neck: 19°", reba.score_neck(19), 1)
check("REBA Neck: 20°", reba.score_neck(20), 2)  # ≥20° → 2 per official example
check("REBA Neck: 25°", reba.score_neck(25), 2)
check("REBA Neck: ext (-5°)", reba.score_neck(-5), 2)

# ── Legs ─────────────────────────────────────────────────────────────────────
check("REBA Legs: stable (1)", reba.score_legs(1), 1)
check("REBA Legs: 1 knee bent (2)", reba.score_legs(2), 2)
check("REBA Legs: squat (3)", reba.score_legs(3), 3)
check("REBA Legs: very unstable (4)", reba.score_legs(4), 4)

# ── Group A table ─────────────────────────────────────────────────────────────
# Cou\Tronc: col1,col2,col3,col4
# Cou=1: 1,2,3,(missing in text → 4 assumed)
# Cou=2: 2,2,3,4
# Cou=3: 3,3,4,5
# Cou=4: 4,4,5,6
# Cou=5: 5,5,6,7
# Cou=6: 6,6,7,8
check("REBA GroupA (t=1,n=1)", reba.group_a_table(1,1,1), 1+1)  # +legs=1
check("REBA GroupA (t=2,n=1)", reba.group_a_table(2,1,1), 2+1)
check("REBA GroupA (t=3,n=1)", reba.group_a_table(3,1,1), 3+1)
check("REBA GroupA (t=4,n=1)", reba.group_a_table(4,1,1), 4+1)
check("REBA GroupA (t=1,n=2)", reba.group_a_table(1,2,1), 2+1)
check("REBA GroupA (t=2,n=2)", reba.group_a_table(2,2,1), 2+1)
check("REBA GroupA (t=3,n=2)", reba.group_a_table(3,2,1), 3+1)
check("REBA GroupA (t=4,n=2)", reba.group_a_table(4,2,1), 4+1)
check("REBA GroupA (t=3,n=2)", reba.group_a_table(3,2,1), 3+1)
check("REBA GroupA (t=4,n=3)", reba.group_a_table(4,3,1), 5+1)
check("REBA GroupA (t=4,n=4)", reba.group_a_table(4,4,1), 6+1)
check("REBA GroupA (t=4,n=5)", reba.group_a_table(4,5,1), 7+1)
check("REBA GroupA (t=4,n=6)", reba.group_a_table(4,6,1), 8+1)
# Tronc=3, Cou=2 → table → 3, +legs=1 → 4
# (neck=2 because score_neck(20°)=2 from official example boundary)
check("REBA GroupA example (t=3,n=2,legs=1)", reba.group_a_table(3,2,1), 4)

# ── Upper arm (REBA) ──────────────────────────────────────────────────────────
check("REBA UA: 0°", reba.score_upper_arm(0), 1)
check("REBA UA: 30°", reba.score_upper_arm(30), 2)
check("REBA UA: 45°", reba.score_upper_arm(45), 2)
check("REBA UA: 60°", reba.score_upper_arm(60), 3)
check("REBA UA: 90°", reba.score_upper_arm(90), 3)
check("REBA UA: 120°", reba.score_upper_arm(120), 4)
# Supported → +1 per REBA text "Bras soutenu → +1"
check("REBA UA: 60° + supported (+1)", reba.score_upper_arm(60, supported=True), 4)

# ── Lower arm (REBA) ──────────────────────────────────────────────────────────
check("REBA LA: 80°", reba.score_lower_arm(80), 1)
check("REBA LA: 45°", reba.score_lower_arm(45), 2)
check("REBA LA: 110°", reba.score_lower_arm(110), 2)

# ── Wrist (REBA) ─────────────────────────────────────────────────────────────
check("REBA Wrist: 0°", reba.score_wrist(0), 1)
check("REBA Wrist: 15°", reba.score_wrist(15), 1)
check("REBA Wrist: 20°", reba.score_wrist(20), 2)
check("REBA Wrist: 20° + deviated", reba.score_wrist(20, deviated=True), 3)

# ── Group B table (REBA) ──────────────────────────────────────────────────────
# Avant-bras\Bras: 1,2,3,4
# 1: 1,2,3,4
# 2: 2,2,3,4
# 3: 3,3,4,5
# 4: 4,4,5,6
# Then + wrist
check("REBA GroupB (ua=1,la=1,w=1)", reba.group_b_table(1,1,1), 1+1)
check("REBA GroupB (ua=2,la=1,w=1)", reba.group_b_table(2,1,1), 2+1)
check("REBA GroupB (ua=3,la=1,w=1)", reba.group_b_table(3,1,1), 3+1)
check("REBA GroupB (ua=4,la=1,w=1)", reba.group_b_table(4,1,1), 4+1)
check("REBA GroupB (ua=1,la=2,w=1)", reba.group_b_table(1,2,1), 2+1)
check("REBA GroupB (ua=2,la=2,w=1)", reba.group_b_table(2,2,1), 2+1)
check("REBA GroupB (ua=3,la=2,w=1)", reba.group_b_table(3,2,1), 3+1)
check("REBA GroupB (ua=4,la=2,w=1)", reba.group_b_table(4,2,1), 4+1)
check("REBA GroupB (ua=3,la=3,w=1)", reba.group_b_table(3,3,1), 4+1)
check("REBA GroupB (ua=4,la=4,w=1)", reba.group_b_table(4,4,1), 6+1)
# Example: Bras=3, Avant-bras=1, Poignet=3 → table(3,1)=3 + wrist=3 = 6
check("REBA GroupB example (ua=3,la=1,w=3)", reba.group_b_table(3,1,3), 6)

# ── Final REBA table ──────────────────────────────────────────────────────────
# SA\SB: 1..15
# From reba_text.txt lines 260-371
check("REBA Final (A=1,B=1)", reba.final_table(1,1), 1)
check("REBA Final (A=1,B=5)", reba.final_table(1,5), 5)
check("REBA Final (A=2,B=2)", reba.final_table(2,2), 2)
# The reba_text.txt example includes load+activity on top of table result:
# final_table(A=5, B=6) = 9, then +load(8kg=1)+activity(+1) = 11
# Without load/activity (our angle-only mode), expect just the table value = 9
check("REBA Final (A=5,B=6)=9 (angle-only)", reba.final_table(5,6), 9)
check("REBA Final (A=3,B=15)", reba.final_table(3,15), 15)
check("REBA Final (A=7,B=15)", reba.final_table(7,15), 15)

# ── Full pipeline example (from reba_text.txt) ────────────────────────────────
# Tronc 30° → 3, Cou 20° → score=2 (≥20° rule), Jambes stable → 1
# GroupA = table(trunk=3, neck=2) + legs → table row neck=2,col trunk=3 → 3, +1 = 4
# Bras 50° → 3, Avant-bras 90° → 1, Poignet 20° → 2 (no deviation in angles_ex)
# GroupB = table(ua=3,la=1)+wrist(2) = 3+2 = 5
# Final (angle-only) = table(A=4, B=5) = 7
angles_ex = {
    'trunk': 30, 'trunk_mod': 0,
    'neck': 20,  'neck_mod': 0,
    'upper_arm_left': 50, 'shoulder_mod': 0,
    'elbow_left': 90,
    'wrist_left': 20,
}
res = reba.compute(angles_ex)
print(f"\n  REBA pipeline example (no load):")
print(f"    table_A={res['table_A']}, table_B={res['table_B']}, score_C={res['score_C']}")
check("REBA pipeline table_A=4", res['table_A'], 4)
check("REBA pipeline table_B=5", res['table_B'], 5)
check("REBA pipeline score_C=7", res['score_C'], 7)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
if errors:
    print(f"[FAILED] {len(errors)} error(s):")
    for e in errors:
        print(f"  • {e}")
else:
    print("[ALL PASSED]")
print('='*50)
