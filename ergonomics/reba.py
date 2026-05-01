# ergonomics/reba.py
# Rewritten to match the official REBA tables exactly as documented in reba_text.txt.

class REBACalculator:

    # ------------------------------------------------------------------
    # Group A – Trunk, Neck, Legs
    # ------------------------------------------------------------------
    @staticmethod
    def score_trunk(angle, rotated=False, lateral=False):
        """
        Tableau 1 (reba_text.txt):
          tronc droit      → 1
          flexion 0–20°   → 2
          flexion 20–60°  → 3
          flexion >60°    → 4
          extension       → 2
        Additionnels: rotation→+1, inclinaison latérale→+1
        """
        if angle < 0:
            s = 2           # extension
        elif angle == 0:
            s = 1           # perfectly upright
        elif angle <= 20:
            s = 2
        elif angle <= 60:
            s = 3
        else:
            s = 4
        if rotated: s += 1
        if lateral: s += 1
        return min(max(s, 1), 6)

    @staticmethod
    def score_neck(angle, rotated=False, lateral=False):
        """
        Tableau Cou (reba_text.txt):
          0–19° flexion  → 1
          ≥20° flexion  → 2   (official example: 20° → score=2)
          extension     → 2
        Additionnels: rotation→+1, inclinaison latérale→+1
        """
        if angle < 0:
            s = 2           # extension
        elif angle < 20:
            s = 1
        else:
            s = 2
        if rotated: s += 1
        if lateral: s += 1
        return min(max(s, 1), 4)

    @staticmethod
    def score_legs(posture=1):
        """
        Tableau Jambes (reba_text.txt):
          assis/debout stable           → 1
          un genou fléchi/poids déséq.  → 2
          accroupi/genoux fort. fléchis → 3
          posture très instable         → 4
        """
        return min(max(posture, 1), 4)

    @staticmethod
    def group_a_table(trunk, neck, legs):
        """
        Tableau A – Combinaison Tronc × Cou (reba_text.txt), then add legs.

        Cou\Tronc  1  2  3  4
              1    1  2  3  (4)  ← text: row 1 shows 1,2,3 then missing 4th col
              2    2  2  3   4
              3    3  3  4   5
              4    4  4  5   6
              5    5  5  6   7
              6    6  6  7   8

        Reading from reba_text.txt lines 72-106 carefully:
        Row (Cou=1): 1, 2, 3, [4]
        Row (Cou=2): 2, 2, 3, 4
        Row (Cou=3): 3, 3, 4, 5
        Row (Cou=4): 4, 4, 5, 6
        Row (Cou=5): 5, 5, 6, 7
        Row (Cou=6): 6, 6, 7, 8
        """
        table = [
            [1, 2, 3, 4],   # cou=1
            [2, 2, 3, 4],   # cou=2
            [3, 3, 4, 5],   # cou=3
            [4, 4, 5, 6],   # cou=4
            [5, 5, 6, 7],   # cou=5
            [6, 6, 7, 8],   # cou=6
        ]
        n = min(max(neck,  1), 6)
        t = min(max(trunk, 1), 4)
        return table[n - 1][t - 1] + legs

    # ------------------------------------------------------------------
    # Group B – Upper arm, Lower arm, Wrist
    # ------------------------------------------------------------------
    @staticmethod
    def score_upper_arm(angle, supported=False):
        """
        Tableau Bras supérieur (reba_text.txt):
          bras le long du corps (≤20°) → 1
          20–45°  → 2
          45–90°  → 3
          >90°    → 4
        Additionnels: bras soutenu → +1 ; charge/effort → +1
        Both text says “Bras soutenu → +1” (adds risk, not subtracts)
        """
        a = abs(angle)
        if a <= 20:
            s = 1
        elif a <= 45:
            s = 2
        elif a <= 90:
            s = 3
        else:
            s = 4
        if supported: s += 1   # “Bras soutenu → +1” per REBA text
        return min(max(s, 1), 6)

    @staticmethod
    def score_lower_arm(angle):
        """
        Tableau Avant-bras (reba_text.txt):
          60–100° → 1
          <60° ou >100° → 2
        """
        if 60 <= angle <= 100:
            return 1
        return 2

    @staticmethod
    def score_wrist(angle, deviated=False):
        """
        Tableau Poignet (reba_text.txt):
          neutre          → 1
          flex/ext >15°   → 2
          déviation latérale → +1
        """
        s = 2 if abs(angle) > 15 else 1
        if deviated:
            s += 1
        return min(max(s, 1), 3)

    @staticmethod
    def group_b_table(upper_arm, lower_arm, wrist):
        """
        Tableau B – Avant-bras × Bras (reba_text.txt lines 149-173),
        then add wrist.

        Avant-bras\Bras  1  2  3  4
                    1    1  2  3  4
                    2    2  2  3  4
                    3    3  3  4  5
                    4    4  4  5  6
        """
        table = [
            [1, 2, 3, 4],   # avant-bras=1
            [2, 2, 3, 4],   # avant-bras=2
            [3, 3, 4, 5],   # avant-bras=3
            [4, 4, 5, 6],   # avant-bras=4
        ]
        la = min(max(lower_arm,  1), 4)
        ua = min(max(upper_arm, 1), 4)
        return table[la - 1][ua - 1] + wrist


    # ------------------------------------------------------------------
    # Final REBA Table C (Score A × Score B)
    # ------------------------------------------------------------------
    @staticmethod
    def final_table(scoreA, scoreB):
        """
        Tableau final REBA (Section IV, reba_text.txt lines 241-371).
        Rows = Score Groupe A (1-7), Cols = Score Groupe B (1-15).

        Parsed exactly from the text file:
        SA\SB   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
          1     1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
          2     2   2   3   4   5   6   7   8   9  10  11  12  13  14  15
          3     3   3   4   5   6   7   8   9  10  11  12  13  14  15  15
          4     4   4   5   6   7   8   9  10  11  12  13  14  15  15  15
          5     5   5   6   7   8   9  10  11  12  13  14  15  15  15  15
          6     6   6   7   8   9  10  11  12  13  14  15  15  15  15  15
          7     7   7   8   9  10  11  12  13  14  15  15  15  15  15  15
        """
        table = [
            [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
            [ 2,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
            [ 3,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 15],
            [ 4,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 15, 15],
            [ 5,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 15, 15, 15],
            [ 6,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 15, 15, 15, 15],
            [ 7,  7,  8,  9, 10, 11, 12, 13, 14, 15, 15, 15, 15, 15, 15],
        ]
        sa = min(max(scoreA, 1), 7)
        sb = min(max(scoreB, 1), 15)
        return table[sa - 1][sb - 1]

    # ------------------------------------------------------------------
    # Main compute entry point
    # ------------------------------------------------------------------
    def compute(self, angles):
        # ── Group A ────────────────────────────────────────────────────
        trunk = self.score_trunk(
                    angles.get('trunk', 0),
                    lateral=(angles.get('trunk_mod', 0) > 0))
        neck  = self.score_neck(
                    angles.get('neck', 0),
                    lateral=(angles.get('neck_mod', 0) > 0))
        legs  = self.score_legs(1)          # assume stable standing

        scoreA = self.group_a_table(trunk, neck, legs)

        # ── Group B ────────────────────────────────────────────────────
        ua    = self.score_upper_arm(
                    angles.get('upper_arm_left', 0),
                    supported=False)  # Vision lacks support detection
        la    = self.score_lower_arm(angles.get('elbow_left', 90))
        wrist = self.score_wrist(angles.get('wrist_left', 0))

        scoreB = self.group_b_table(ua, la, wrist)

        # ── Combine (angle-only, no load) ─────────────────────────────
        final = self.final_table(scoreA, scoreB)

        risk = self.interpret(final)
        return {
            "REBA_score":       final,
            "risk_level":       risk,
            "score_C":          final,
            "table_A":          scoreA,
            "table_B":          scoreB,
            "trunk_score":      trunk,
            "trunk_mod":        angles.get('trunk_mod', 0),
            "neck_score":       neck,
            "neck_mod":         angles.get('neck_mod', 0),
            "legs_score":       legs,
            "knee_mod":         0,
            "upper_arm_score":  ua,
            "shoulder_mod":     angles.get('shoulder_mod', 0),
            "lower_arm_score":  la,
            "wrist_score":      wrist,
            "wrist_twist":      0,
            "coupling":         0,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def interpret(score):
        """
        Tableau d'interprétation (Section IV, reba_text.txt):
          1         → Négligeable
          2–3       → Faible
          4–7       → Moyen – Amélioration nécessaire
          8–10      → Élevé – Intervention rapide
          11–15     → Très élevé – Action immédiate
        """
        if score <= 1:    return "Négligeable"
        elif score <= 3:  return "Faible – Surveiller"
        elif score <= 7:  return "Moyen – Amélioration nécessaire"
        elif score <= 10: return "Élevé – Intervention rapide"
        else:             return "Très élevé – Action immédiate"