from audit import audit_source


def example(observations):
    sources, raw = [], {}
    for i, (before, after, exact) in enumerate(observations, 1):
        sources.append(dict(uid="1", rpid=str(i), text="example", rule_score=before,
                            provenance=[dict(file="out/example", line=i)]))
        raw["out/example", i] = dict(mid="1", rpid=str(i), text="example",
                                     score=before, mask_score=after, replay_ok=exact)
    return {"sources": sources}, raw


def test_all_supporting_observations_must_fall_below_cutoff():
    assert audit_source(*example([(30, 29, True), (40, 30, True)])) == (40, 30, "same_side")
    assert audit_source(*example([(30, 29, True), (40, 28, True)])) == (40, 29, "falls_below_30")


def test_incomplete_replay_is_unknown():
    assert audit_source(*example([(31, 28, True), (40, 28, False)]))[2] == "not_reproduced"
    assert audit_source(*example([]))[2] == "no_source"
