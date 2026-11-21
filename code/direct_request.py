"""Frozen text-only protection for short requests addressed to a leading mention."""
import re
import unicodedata

LEADING_MENTION = re.compile(r"^(?:回复\s*)?@[^\s@,，:：。?？!！]{1,32}(?:\s+|[,，:：])")
ACTION = r"(?:看|瞅|挑|选|查|找|试|装|安装|卸载|学|读|帮|用|跑|翻译|总结|解释|分析|检查|推荐|告诉我|发给我)"
REQUESTS = tuple(re.compile(pattern) for pattern in (
    rf"(?:你(?:帮我|给我|来|自己|也|去)?|帮我|给我|麻烦你?|请你?|来|快来|那就(?:自己)?|那你(?:自己)?){ACTION}.{{0,14}}",
    r"(?:怎么(?:看|用|装|选)|你(?:怎么看|觉得怎么样)|有想(?:要|装|安装)的吗|看看(?:这[个条]|有没有|有啥)|看一下|学一下|试一下|来一份).{0,12}",
    rf"(?:你)?(?:到底)?(?:能不能|会不会|要不要|想不想|可不可以|有没有)(?:自己|帮我|给我)?(?:{ACTION}|想要|需要|要用|想装|喜欢).{{0,12}}",
    rf"(?:你)?(?:有|能|会|要|想|需要|愿意|可以)?(?:自己)?(?:{ACTION}|需要|想要|要用|想装).{{0,10}}(?:吗|么|不|没|嘛)",
))
INNER_DELIMITER = re.compile(r'[@\n\r,，:：;；。?!“”「」『』"…]')
END_PUNCTUATION = re.compile(r"[?!.。~～]+$")


def is_direct_request(text):
    if text is None:
        return False
    text = unicodedata.normalize("NFKC", text).strip()
    mention = LEADING_MENTION.match(text)
    if not mention:
        return False
    body = text[mention.end():].lstrip(" ,:，：").strip()
    body = END_PUNCTUATION.sub("", body).strip()
    if not 1 <= len(body) <= 20 or INNER_DELIMITER.search(body):
        return False
    return any(pattern.fullmatch(body) for pattern in REQUESTS)
