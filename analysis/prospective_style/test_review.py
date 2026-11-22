"""Analytic regression cases use invented records, never fill a real blind packet."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from collect import canonical
from review import METHODS, P, agreement, evaluate, packet, paired_ci, summarize


class ReviewTest(unittest.TestCase):
    def test_cross_form(self):
        self.assertEqual(canonical("回复 ＠小王： 能，装吗？")[1], canonical("能装吗")[0])
        self.assertNotEqual(canonical("@小王 能装吗")[1], canonical("@小王 我装了")[1])

    def test_agreement_before_adjudication(self):
        labels = P["style"]
        a = {str(i): {"style": x} for i, x in enumerate([labels[0], labels[1], labels[2]])}
        b = {str(i): {"style": x} for i, x in enumerate([labels[0], labels[1], labels[1]])}
        result = agreement(a, b, "style")
        self.assertAlmostEqual(result["kappa"], 0.5)
        self.assertEqual(result["uncertain_A"], 1 / 3)
        self.assertIsNone(agreement({"x": a["0"]}, {"x": a["0"]}, "style")["kappa"])

    def test_blind_packets_same_population(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            rows = [
                {"rpid": str(i), "text": "@某甲 帮看看", "arm": "diagnostic" if i else "natural"}
                for i in range(4)
            ]
            scores = [{"id": str(i), "eligible": i != 3, "plain": i != 2} for i in range(4)]
            (p / "frame.json").write_text(json.dumps(rows))
            (p / "scores.json").write_text(json.dumps(scores))
            self.assertEqual({r["rpid"] for r in packet(p)}, {"0", "1"})
            for who in "AB":
                data = json.loads((p / f"packet-{who}.json").read_bytes())
                self.assertTrue(all(r[0].startswith(who) and r[1] == "@某人 帮看看" for r in data))

    def test_paired_deltas_and_zero_denominators(self):
        rows = [
            {"style": P["style"][0], "bvid": "v", "guarded": True, "learned": False},
            {"style": P["style"][1], "bvid": "v", "guarded": False, "learned": True},
        ]
        self.assertEqual(summarize(rows, ["guarded", "learned"])["fewer_human_blocks_per_1000"], 1000)
        self.assertEqual(summarize(rows, ["guarded", "learned"])["recall_delta"], 1)
        videos = [{"bvid": "v", "pool": "x"}, {"bvid": "empty", "pool": "y"}]
        self.assertEqual(paired_ci(rows, videos)[0]["ci95"], [1000.0, 1000.0])
        self.assertEqual(paired_ci([], videos)[1]["undefined"], 5000)

    def test_ties_in_average_precision(self):
        rows = [
            {"style": s, "guarded": False, "learned": False, "plain": True, "masked_score": 0.5}
            for s in P["style"][:2]
        ]
        self.assertEqual(summarize(rows, ["plain"])["models"]["plain"]["average_precision"], 0.5)

    def test_adjudication_and_result_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            rows = [
                {
                    "rpid": str(i),
                    "text": "虚构测试",
                    "arm": arm,
                    "bvid": str(i),
                    "pool": "测试",
                    "ordinary_short": True,
                }
                for i, arm in enumerate(("natural", "diagnostic"))
            ]
            scores = [
                dict(id=r["rpid"], eligible=True, rule=False, request=True, **dict.fromkeys(METHODS, False))
                for r in rows
            ]
            for name, data in [("frame", rows), ("scores", scores), ("selection", rows)]:
                (p / (name + ".json")).write_text(json.dumps(data))
            a = {r["rpid"]: {d: P[d][0] for d in ("style", "intent", "promotion")} for r in rows}
            b = {k: v.copy() for k, v in a.items()}
            b["1"]["intent"] = P["intent"][1]
            with patch("review.read_sheet", side_effect=[a, b]):
                self.assertIn("Awaiting", evaluate(p, None, None, None))
            self.assertFalse((p / "result.json").exists())
            with (p / "adjudication.csv").open(encoding="utf-8-sig", newline="") as f:
                fields = list(csv.reader(f))
            fields[1][-1] = "不确定"
            resolved = p / "resolved.csv"
            with resolved.open("w", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerows(fields)
            with patch("review.read_sheet", side_effect=[a, b]):
                evaluate(p, None, None, resolved)
            report = json.loads((p / "result.json").read_bytes())
            self.assertEqual(report["natural"]["rows"], 1)
            self.assertEqual(report["diagnostic_counts"]["rows"], 1)
            self.assertEqual(report["natural_intent"]["policies"]["request"]["recognized_requests"], 1)
            self.assertEqual(report["diagnostic_intent"]["uncertain"], 1)


if __name__ == "__main__":
    unittest.main()
