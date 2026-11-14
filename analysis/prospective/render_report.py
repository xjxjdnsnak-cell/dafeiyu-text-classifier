"""Render the frozen-benchmark comparison report (markdown) from evaluation.json."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def pct(value):
    return "—" if value is None else f"{value:.2%}"


def wilson(metrics):
    interval = metrics.get("fpr_wilson_95")
    if not interval:
        return "—"
    return f"[{interval[0]:.2%}, {interval[1]:.2%}]"


def main():
    result = json.loads((HERE / "evaluation.json").read_text(encoding="utf-8"))
    bench = result["benchmark"]
    rounds = result["rounds"]
    lines = [
        "# 前瞻基准集对比：v5 / v6 / v7 / v8（冻结，未调参）",
        "",
        f"基准集：{bench['rows']} 行（AI {bench['labels'].get('1', 0)} / 人类 {bench['labels'].get('0', 0)}），"
        f"普通昵称 AI {bench['ordinary_ai_rows']} 行，短文本(≤20字) AI {bench['short_ai_rows']} 行，"
        f"{bench['videos']} 个全新视频。",
        "",
        f"冻结时间：{result['benchmark']['frozen_at']}；"
        "账号/视频/正文三重隔离已在构建时强制执行并在评估时复核。",
        "",
        "阈值：四个模型各自沿用当轮验证集冻结的部署配置（1% 预算默认阈值 + 2% 工作点），"
        "**未根据本基准集做任何调整**。",
        "",
        "## 主对比（1% 预算部署阈值）",
        "",
        "| 模型 | 改判背景 | 整体AI召回 | 普通昵称AI召回 | 短评(≤20字)AI召回 | 真人误拦 | 误拦率 | 误拦Wilson95% |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    backgrounds = {"v5": "13条改判", "v6": "22条改判(+批4)", "v7": "29条改判(+批5)", "v8": "30条改判(+批6)"}
    for tag in ("v5", "v6", "v7", "v8"):
        one = rounds[tag]["1pct"]
        all_m = one["all"]
        ordinary = one["ordinary_ai_recall"]
        short = one["short_ai_recall"]
        lines.append(f"| {tag} | {backgrounds[tag]} | {pct(all_m['recall'])} | "
                     f"{ordinary['confusion_matrix'][1][1]}/{ordinary['ai_rows']}（{pct(ordinary['recall'])}） | "
                     f"{short['confusion_matrix'][1][1]}/{short['ai_rows']}（{pct(short['recall'])}） | "
                     f"{all_m['confusion_matrix'][0][1]}/{all_m['support']['normal']} | {pct(all_m['fpr'])} | {wilson(all_m)} |")
    lines += ["", "## 2% 工作点（同一模型，第二轮冻结阈值）", "",
              "| 模型 | 整体AI召回 | 普通昵称AI召回 | 短评(≤20字)AI召回 | 真人误拦 | 误拦率 | 误拦Wilson95% |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for tag in ("v5", "v6", "v7", "v8"):
        two = rounds[tag]["2pct"]
        all_m = two["all"]
        ordinary = two["ordinary_ai_recall"]
        short = two["short_ai_recall"]
        lines.append(f"| {tag} | {pct(all_m['recall'])} | "
                     f"{ordinary['confusion_matrix'][1][1]}/{ordinary['ai_rows']}（{pct(ordinary['recall'])}） | "
                     f"{short['confusion_matrix'][1][1]}/{short['ai_rows']}（{pct(short['recall'])}） | "
                     f"{all_m['confusion_matrix'][0][1]}/{all_m['support']['normal']} | {pct(all_m['fpr'])} | {wilson(all_m)} |")
    lines += ["", "## 真人误拦分长度组（1% 部署阈值）", "",
              "| 模型 | ≤20字 | 21–60字 | ≥61字 |", "|---|---|---|---|"]
    for tag in ("v5", "v6", "v7", "v8"):
        one = rounds[tag]["1pct"]
        row = []
        for key in ("length_1_20_human_fpr", "length_21_60_human_fpr", "length_61p_human_fpr"):
            m = one[key]
            row.append(f"{m['confusion_matrix'][0][1]}/{m['human_rows']}（{pct(m['fpr'])}）")
        lines.append(f"| {tag} | " + " | ".join(row) + " |")
    thresholds = ["| 模型 | 1%阈值 | 2%阈值 | 模型sha256前12 |", "|---|---|---|---|"]
    for tag in ("v5", "v6", "v7", "v8"):
        entry = rounds[tag]
        thresholds.append(f"| {tag} | {entry['thresholds']['fpr_1pct']:.17g} | {entry['thresholds']['fpr_2pct']:.17g} | {entry['model_sha256'][:12]} |")
    lines += ["", "## 阈值（冻结，未调整）", ""] + thresholds
    lines += [
        "",
        "## 读数须知",
        "",
        "- 本基准集与前几轮的测试集**完全独立**（账号/视频/正文三重隔离），这是首个对 v5–v8 公平的前瞻对比。",
        "- 各模型阈值来自各自验证集，校准水平不同；比较的是**部署配置**，不是概率校准。",
        "- 基准集标签是机械政策（规则分 ≥30 记 AI、灰区 v3 排除、纯 @ 排除），无人工复核——复核循环已按用户要求停止。",
        "- 本结果**不用于**调参、选模型或改阈值；仅供记录与比较。",
        "",
    ]
    (HERE / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(HERE / "evaluation.md"), "lines": len(lines)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
