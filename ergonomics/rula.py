# ergonomics/rula.py
# Rewritten to match the official RULA tables exactly as documented in rula_text.txt.

class RULACalculator:

    # ------------------------------------------------------------------
    # Group A – Upper arm, Lower arm, Wrist
    # ------------------------------------------------------------------
    @staticmethod
    def score_upper_arm(angle, rotated=False, abducted=False, supported=False):
        """
        Tableau 1 (rula_text.txt):
          neutre / long du corps  → 1
          élévation < 20°         → 2  (text shows 1 for neutral, 2 for <20°)
          élévation 20–45°        → 2
          élévation 45–90°        → 3
          élévation > 90°         → 4
        Additionnels: rotation ext/int → +1 ; bras soutenu → +1 (abducted)
        Note: supported arm gets -1 per standard RULA (arm resting on surface)
        """
        a = abs(angle)
        if a < 20:
            s = 2           # <20° elevation (≈ along body = 1 is neutral/0°;
        if a == 0:
            s = 1           # truly neutral (arm hanging straight)
        if 20 <= a <= 45:
            s = 2
        elif 45 < a <= 90:
            s = 3
        elif a > 90:
            s = 4
        else:
            s = 2 if a > 0 else 1

        if rotated:   s += 1
        if abducted:  s += 1
        if supported: s -= 1
        return min(max(s, 1), 6)

    @staticmethod
    def score_lower_arm(elbow_angle):
        """
        Tableau 2 (rula_text.txt):
          60°–100° → 1
          <60° ou >100° → 2
        """
        if 60 <= elbow_angle <= 100:
            return 1
        return 2

    @staticmethod
    def score_wrist(angle, deviation=0):
        """
        Tableau 3 (rula_text.txt):
          0°–15° (neutre)  → 1
          15°–30°          → 2
          >30°             → 3
        Tableau 4 (déviation):
          aucune=0, légère=1, prononcée=2
        """
        a = abs(angle)
        if a <= 15:
            s = 1
        elif a <= 30:
            s = 2
        else:
            s = 3
        s += deviation
        return min(s, 4)

    @staticmethod
    def score_load(weight_kg, repetitive=False):
        """
        Tableau 5 (rula_text.txt):
          léger / pas de force        → 0
          charge moyenne/répétitif    → 1
          charge lourde/fort/prolongé → 2
        """
        if weight_kg < 2:
            return 0
        elif weight_kg <= 10:
            return 1 if repetitive else 0
        else:
            return 2

    @staticmethod
    def group_a_table(upper_arm, lower_arm, wrist, wrist_twist, load):
        """
        Tableau 6 (rula_text.txt) – exact values:
        Rows: (upper_arm, lower_arm)
        Cols: wrist+twist index 1-4

        Bras\Avant-bras  P+T=1  P+T=2  P+T=3  P+T=4
        1,1               1      2      2      3
        1,2               2      2      3      3
        2,1               2      3      3      3
        2,2               3      3      4      4
        3,1               3      3      4      4
        3,2               4      4      5      5
        4,1               4      4      5      5
        4,2               5      5      6      6
        """
        table = {
            (1, 1): [1, 2, 2, 3],
            (1, 2): [2, 2, 3, 3],
            (2, 1): [2, 3, 3, 3],
            (2, 2): [3, 3, 4, 4],
            (3, 1): [3, 3, 4, 4],
            (3, 2): [4, 4, 5, 5],
            (4, 1): [4, 4, 5, 5],
            (4, 2): [5, 5, 6, 6],
        }
        ua = min(max(upper_arm, 1), 4)
        la = min(max(lower_arm, 1), 2)
        row = table.get((ua, la), [1, 2, 2, 3])
        # Column index: wrist + wrist_twist combined, clamped 1-4
        col_idx = min(max(wrist + wrist_twist - 1, 1), 4) - 1
        return row[col_idx] + load

    # ------------------------------------------------------------------
    # Group B – Neck, Trunk, Legs
    # ------------------------------------------------------------------
    @staticmethod
    def score_neck(angle, rotated=False, lateral=False):
        """
        Tableau (Group B, rula_text.txt):
          0°–10°  → 1
          10°–20° → 2
          >20°    → 3
          extension → 4
        Additionnels: rotation→+1, inclinaison latérale→+1
        """
        if angle < 0:
            s = 4           # extension
        elif angle <= 10:
            s = 1
        elif angle <= 20:
            s = 2
        else:
            s = 3
        if rotated: s += 1
        if lateral: s += 1
        return min(max(s, 1), 6)

    @staticmethod
    def score_trunk(angle, rotated=False, lateral=False):
        """
        Tableau (Group B, rula_text.txt):
          droit (0°)       → 1
          flexion 0–20°    → 2
          flexion 20–60°   → 3
          flexion >60°     → 4
          extension        → 2
        Additionnels: rotation→+1, inclinaison latérale→+1
        """
        if angle < 0:
            s = 2           # extension
        elif angle == 0:
            s = 1           # perfectly straight
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
    def score_legs(stable=True):
        """
        Tableau (Group B, rula_text.txt):
          position stable   → 1
          position instable → 2
        """
        return 1 if stable else 2

    @staticmethod
    def group_b_table(neck, trunk, legs):
        """
        Tableau Groupe B (rula_text.txt) – Cou (rows) × Tronc (cols):

        Cou\Tronc  1  2  3  4  5  6
              1    1  2  3  5  6  7
              2    2  2  3  5  6  7
              3    3  3  4  6  7  7
              4    5  5  6  7  7  7
              5    6  6  7  7  7  7
              6    7  7  7  7  7  7

        Then add legs score.
        """
        tbl = [
            [1, 2, 3, 5, 6, 7],   # neck=1
            [2, 2, 3, 5, 6, 7],   # neck=2
            [3, 3, 4, 6, 7, 7],   # neck=3
            [5, 5, 6, 7, 7, 7],   # neck=4
            [6, 6, 7, 7, 7, 7],   # neck=5
            [7, 7, 7, 7, 7, 7],   # neck=6
        ]
        n = min(max(neck, 1), 6)
        t = min(max(trunk, 1), 6)
        return tbl[n - 1][t - 1] + legs

    # ------------------------------------------------------------------
    # Final RULA score table (Score A × Score B → RULA 1-7)
    # ------------------------------------------------------------------
    @staticmethod
    def final_score(scoreA, scoreB):
        """
        Tableau final RULA (Section III, rula_text.txt).
        Standard 7×7 matrix.
        """
        matrix = [
            [1, 2, 3, 3, 4, 5, 5],
            [2, 2, 3, 4, 4, 5, 5],
            [3, 3, 3, 4, 4, 5, 6],
            [3, 3, 4, 4, 4, 5, 6],
            [4, 4, 4, 4, 5, 6, 7],
            [5, 5, 5, 5, 6, 7, 7],
            [5, 5, 6, 6, 7, 7, 7],
        ]
        sa = min(max(scoreA, 1), 7)
        sb = min(max(scoreB, 1), 7)
        return matrix[sa - 1][sb - 1]

    # ------------------------------------------------------------------
    # Main compute entry point
    # ------------------------------------------------------------------
    def compute(self, angles, load_kg=0, repetitive=False):
        # ── Group A ────────────────────────────────────────────────────
        ua    = self.score_upper_arm(
                    angles.get('upper_arm_left', 0),
                    abducted=(angles.get('shoulder_mod', 0) > 0))
        la    = self.score_lower_arm(angles.get('elbow_left', 90))
        wrist = self.score_wrist(angles.get('wrist_left', 0))
        load  = self.score_load(load_kg, repetitive)
        # wrist_twist=1 (neutral) – MediaPipe doesn't give hand axial rotation
        groupA = self.group_a_table(ua, la, wrist, 1, load)

        # ── Group B ────────────────────────────────────────────────────
        neck  = self.score_neck(
                    angles.get('neck', 0),
                    lateral=(angles.get('neck_mod', 0) > 0))
        trunk = self.score_trunk(
                    angles.get('trunk', 0),
                    lateral=(angles.get('trunk_mod', 0) > 0))
        legs  = self.score_legs(angles.get('legs_stable', True))
        groupB = self.group_b_table(neck, trunk, legs)

        # ── Final ──────────────────────────────────────────────────────
        final = self.final_score(groupA, groupB)
        risk  = self.interpret(final)

        return {
            "RULA_score":  final,
            "risk_level":  risk,
            "score_A":     groupA,
            "score_B":     groupB,
            "score_C":     final,
            "upper_arm_score": ua,
            "lower_arm_score": la,
            "wrist_score":     wrist,
            "wrist_twist":     1,
            "neck_score":      neck,
            "trunk_score":     trunk,
            "legs_score":      legs,
            "load_score":      load,
            "muscle_score":    1 if repetitive else 0,
            "activity_score":  1 if repetitive else 0,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def interpret(score):
        """
        Tableau d'interprétation (Section III, rula_text.txt):
          1-2 → Acceptable
          3-4 → Faible / Surveillance
          5-6 → Moyen / Changement nécessaire
          7   → Très élevé / Action immédiate
        """
        if score <= 2:   return "Acceptable"
        elif score <= 4: return "Faible – Surveillance"
        elif score <= 6: return "Moyen – Changement nécessaire"
        else:            return "Très élevé – Action immédiate"