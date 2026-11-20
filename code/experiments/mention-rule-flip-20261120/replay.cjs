// Replay each historical single-comment decision, changing only its body.
const fs = require("node:fs");
const vm = require("node:vm");
const ctx = vm.createContext({});
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx);
const detector = ctx.DFY;
const pairs = JSON.parse(fs.readFileSync(0, "utf8")).map(r => {
  const account = {...r.account, level: r.account.level ?? 4,
    verify: r.account.verify ?? -1, vip: r.account.vip ?? 0};
  // The export records the actual callout bonus in the original reasons.
  const callout = r.reasons.find(x => x.tag === "callout");
  const calledOut = callout ? Number(callout.why.match(/被 (\d+) 个人/)?.[1] ?? 1) : 0;
  const isReply = String(r.root) !== "0";
  const context = {upMid: r.up_mid, topCount: isReply ? 0 : 1, calledOut, priorPages: 0};
  const judge = text => detector.judge(account, [{text, isReply}], context);
  const original = judge(r.text);
  const masked = judge(r.masked);
  return {original: original.score, masked: masked.score,
    originalText: detector.scoreText(r.text, {uname: account.uname, isReply}).score,
    originalAccount: detector.scoreAccount(account).score,
    reasonsMatch: JSON.stringify(original.reasons) === JSON.stringify(r.reasons)};
});
process.stdout.write(JSON.stringify(pairs));
