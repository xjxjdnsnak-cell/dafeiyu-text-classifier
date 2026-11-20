import * as ort from './ort/ort.wasm.min.mjs';
import {OfflineModels, mentionOnly} from './ft/inference.mjs';
import {normalize} from './ft/preprocess.mjs';

async function main() {
  const read = async name => {
    const response = await fetch(name);
    if (!response.ok) throw new Error('Cannot load ' + name);
    return new Uint8Array(await response.arrayBuffer());
  };
  const ft = await OfflineModels.load(name => read('./ft/' + name));
  await ft.loadVariant('v9');
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = new URL('./ort/', import.meta.url).href;
  const cnn = await ort.InferenceSession.create(await read('./cnn.onnx'), {executionProviders:['wasm']});
  const data = await (await fetch('./sample.json')).json();
  const vocab = await (await fetch('./vocab.json')).json();
  const thresholds = await (await fetch('./thresholds.json')).json();
  // Python re's whitespace includes U+0085 and U+001C..U+001F, excludes U+FEFF.
  const at = /@([^\p{White_Space}\u001c-\u001f@:：，,。！!?？]{1,24})/gu;
  function prepare(text) {
    return mentionOnly(text, ft.pre, ft.emotes) ? '' : text.replace(at, '@某人');
  }
  const calls = {
    fasttext(text) {
      const masked = prepare(text);
      return ft.scoreFeatures(ft.encode(masked,'v9'),'v9');
    },
    async cnn(text) {
      const chars = Array.from(normalize(prepare(text), ft.pre.config)).slice(0,4096);
      if (!chars.length) return null;
      const ids = new BigInt64Array(Math.max(5,chars.length));
      for (let i=0;i<chars.length;i++) ids[i] = BigInt(vocab[chars[i]] ?? 1);
      const input = new ort.Tensor('int64',ids,[1,ids.length]);
      const output = await cnn.run({ids:input});
      const score = Number(output.score.data[0]);
      input.dispose();
      output.score.dispose();
      return score;
    }
  };
  const errors={cnn:0,fasttext:0};
  let cudaDelta = 0;
  for (const r of data) {
    for (const key of ['cnn','fasttext']) {
      const score=await calls[key](r.text), expected=r[key];
      if ((score===null)!==(expected===null) || !Number.isFinite(score ?? 0) ||
          ((score!==null && score>=thresholds[key]) !== (expected!==null && expected>=thresholds[key])))
        throw new Error('Eligibility or classification mismatch: '+key);
      errors[key]=Math.max(errors[key],Math.abs((score??0)-(expected??0)));
      if (key==='cnn') {
        if ((score===null)!==(r.cnn_cuda===null) ||
            ((score!==null && score>=thresholds.cnn)!==(r.cnn_cuda!==null && r.cnn_cuda>=thresholds.cnn)))
          throw new Error('CNN frozen GPU decision mismatch');
        cudaDelta=Math.max(cudaDelta,Math.abs((score??0)-(r.cnn_cuda??0)));
      }
    }
  }
  if (Object.values(errors).some(x=>x>1e-5)) throw new Error('Score mismatch '+JSON.stringify(errors));
  const times={cnn:[],fasttext:[]};
  for (let pass=0;pass<5;pass++) {
    for (let i=0;i<data.length;i++) {
      const row=data[i];
      for (const key of ((i+pass)%2 ? ['cnn','fasttext'] : ['fasttext','cnn'])) {
        const start=performance.now();
        await calls[key](row.text);
        times[key].push(performance.now()-start);
      }
    }
  }
  const result={score_max_delta:errors,cnn_frozen_gpu_max_delta:cudaDelta,rows:data.length,repeats:5,isolated:self.crossOriginIsolated,
                runtime:navigator.userAgent,measurements:{}};
  for (const [name,xs] of Object.entries(times)) {
    xs.sort((a,b)=>a-b);
    result.measurements[name]={n:xs.length,p50_ms:xs[Math.ceil(xs.length*.5)-1],p95_ms:xs[Math.ceil(xs.length*.95)-1]};
  }
  await cnn.release();
  ft.dispose();
  return result;
}
main().then(result=>self.postMessage(result)).catch(e=>self.postMessage({error:String(e.stack||e)}));
