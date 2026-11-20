from run import MaskedBodyEncoder, Preprocessor, prepare, simple_invitation


def test_invitation_matches_the_whole_message():
    assert simple_invitation("快来学学。")
    assert simple_invitation("回复 @测试用户：快来学学！")
    assert not simple_invitation("@测试用户 快来学学。这是另外一句。")
    assert not simple_invitation("@测试用户 还有什么需要安装的？")
    assert not simple_invitation("@测试用户")


def test_deduplication_uses_the_masked_input():
    encoder = MaskedBodyEncoder(Preprocessor())
    held = {encoder.encode("@甲 这一条留出。", "", "text")}
    rows = [{"uid": str(i), "text": text, "label": 0} for i, text in enumerate(
        ["@乙 这一条留出。", "@丙 请看看", "@丁 请看看", "@独立提及"])]
    lines, ledger = prepare([], rows, held, encoder)
    assert len(lines) == 1
    assert [r["reason"] for r in ledger] == [
        "protected_input", "included", "duplicate_input", "empty_input"]
