import fs from "node:fs/promises";
import path from "node:path";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const [folder, exportDir] = process.argv.slice(2);
const protocol = JSON.parse(await fs.readFile(path.join(folder, "protocol.json"), "utf8"));
await fs.mkdir(exportDir, { recursive: true });
const instructions = [
  ["独立判断", "每人单独完成自己的文件。只按看到的正文判断。读不出时选“不确定”，三列分别判断。"],
  ["文本风格", "“真人风格”指更像日常个人交流，“AI风格”指更像生成式回答。短、流畅、礼貌或有列表均不足以单独定类。无法区分两者时选“不确定”。"],
  ["对话意图", "请对方回答或行动，选“请求或提问”。陈述、汇报结果或转述别人的问句，选“非请求”。意图不清时选“不确定”。"],
  ["推广表达", "正文明确招揽、引流、号召关注、下载或购买某个对象，选“有推广表达”。只讨论产品或询问操作可选“无推广表达”。"],
  ["填写与交回", "在每行后三列使用下拉菜单。正文里的称呼统一显示为 @某人。保存自己的副本，保留编号、正文和表头。"]
];
for (const who of ["A", "B"]) {
  const packet = JSON.parse(await fs.readFile(path.join(folder, "packet-" + who + ".json"), "utf8"));
  const file = path.join(exportDir, "独立盲评-" + who + ".xlsx");
  try {
    await fs.access(file);
    throw new Error("Workbook exists. Choose an empty destination to preserve annotations.");
  } catch (err) { if (err.code !== "ENOENT") throw err; }
  const wb = Workbook.create();
  const sh = wb.worksheets.add("盲评");
  sh.showGridLines = false;
  const range = s => sh.getRange(s);
  const end = 8 + packet.length;
  for (const [col, width] of Object.entries({A:96,B:550,C:115,D:128,E:135})) {
    range(col+"1:"+col+end).format.columnWidthPx = width;
  }
  range("A1:E1").merge();
  range("A1").values = [["中文短评独立盲评 · " + who]];
  range("A1:E1").format.fill = "#143C43";
  range("A1:E1").format.font = {bold:true,color:"#FFFFFF",size:16};
  range("A1:E1").format.rowHeightPx = 39;
  instructions.forEach(([label,value], i) => {
    const row = i + 2;
    range("A"+row).values = [[label]];
    range("B"+row+":E"+row).merge();
    range("B"+row).values = [[value]];
    range("A"+row+":E"+row).format.rowHeightPx = 44;
  });
  range("A2:E6").format.verticalAlignment = "center";
  range("A2:A6").format.font = {bold:true,color:"#184E52",size:11};
  range("B2:E6").format.wrapText = true;
  range("B2:E6").format.font = {color:"#27363A",size:11};
  range("A7:E7").format.rowHeightPx = 8;
  range("A8:E8").values = [["编号","评论正文","文本风格","对话意图","推广表达"]];
  range("A8:E8").format.fill = "#276568";
  range("A8:E8").format.font = {bold:true,color:"#FFFFFF"};
  range("A8:E8").format.rowHeightPx = 31;
  range("A9:B"+end).setNumberFormat("@");
  range("A9:B"+end).values = packet;
  range("A9:B"+end).format.wrapText = true;
  range("C9:E"+end).format.fill = "#EBF5F1";
  range("A9:E"+end).format.verticalAlignment = "center";
  for (let i=0;i<packet.length;i++){
    const lines=packet[i][1].split(/\r?\n/).reduce((n,s)=>n+Math.max(1,Math.ceil([...s].length/33)),0);
    range("A"+(9+i)+":E"+(9+i)).format.rowHeightPx=Math.min(545,Math.max(35,lines*21+9));
  }
  for (const [col, field] of [["C","style"],["D","intent"],["E","promotion"]]){
    range(col+"9:"+col+end).dataValidation={rule:{type:"list",values:protocol[field]}};
  }
  sh.freezePanes.freezeRows(8);
  const preview=await wb.render({sheetName:"盲评",range:"A1:E12",scale:1,format:"png"});
  await fs.writeFile(path.join(folder,"preview-"+who+".png"),new Uint8Array(await preview.arrayBuffer()));
  await (await SpreadsheetFile.exportXlsx(wb)).save(file);
  try {
    await fs.rename(file+".inspect.ndjson",path.join(folder,"inspect-"+who+".ndjson"));
  } catch (err) { if (err.code !== "ENOENT") throw err; }
}
