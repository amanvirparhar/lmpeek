import { fromPreTrained } from "@lenml/tokenizer-gpt2";
import * as transformers from "@xenova/transformers";
import * as ort from "onnxruntime-web";
import { get, set } from "idb-keyval";

export interface GPT2Embeddings {
  tok_emb: ort.Tensor;
  pos_emb: ort.Tensor;
  input_emb: ort.Tensor;
}

export interface GPT2AttentionHead {
  q: ort.Tensor;
  k: ort.Tensor;
  v: ort.Tensor;
  attn_weight: ort.Tensor;
  attn_value_output: ort.Tensor;
}

export interface GPT2MLP {
  mlp_input: ort.Tensor;
  mlp_activation: ort.Tensor;
  mlp_output: ort.Tensor;
}

export interface GPT2TransformerLayer {
  block_input: ort.Tensor;
  attn_input: ort.Tensor;
  attn_heads: GPT2AttentionHead[];
  attn_output: ort.Tensor;
  mlp: GPT2MLP;
  block_output: ort.Tensor;
}

export interface GPT2FinalOutputs {
  ln_f_output: ort.Tensor;
  logits: ort.Tensor;
}

export interface GPT2Outputs {
  embeddings: GPT2Embeddings;
  layers: GPT2TransformerLayer[];
  final: GPT2FinalOutputs;
}

const MODEL_CONFIG = {
  url: "https://huggingface.co/Amanvir/gpt-2-onnx-test/resolve/main/gpt2-no-constant-folding.onnx",
};

// ort.env.logLevel = "warning";

let tokenizer:
    | ReturnType<typeof fromPreTrained>
    | transformers.PreTrainedTokenizer,
  modelSession: ort.InferenceSession,
  isLoaded = false;

self.onmessage = async (e) => {
  const { type, name, data } = e.data;

  if (type == "action") {
    if (name === "loadModel") {
      await loadModel(data.id, data?.options);
    } else if (name === "forward") {
      await forward(data.id, data.input, data?.options);
    } else if (name === "sample") {
      await getTokens(data.id, getTokenProbs(data.logits, data?.options));
    }
  }
};

async function loadModel(
  id: number,
  modelOptions?: {
    onnxExecutionProviders?: string[];
    tokenizer?: string;
  }
) {
  const startTime = performance.now();

  if (isLoaded) {
    postMessage({
      id,
      type: "error",
      name: "modelAlreadyLoaded",
    });
    return;
  }

  if (!modelOptions?.tokenizer) {
    console.log("using default tokenizer");
    tokenizer = fromPreTrained();
  } else {
    try {
      tokenizer = await transformers.AutoTokenizer.from_pretrained(
        modelOptions.tokenizer
      );
      console.log("loaded custom tokenizer:", modelOptions.tokenizer);
    } catch (error) {
      console.log("error!");
      postMessage({
        id,
        type: "error",
        name: "tokenizerLoadError",
        data: `Failed to load tokenizer: ${error}`,
      });
    }
  }

  const getModelUrl = async () => {
    try {
      // check if model is in indexeddb
      const file = await get(MODEL_CONFIG.url);
      if (!file) throw new Error("Model file not found in IDB.");
      console.log("model found in indexeddb:", MODEL_CONFIG.url);
      return URL.createObjectURL(file);
    } catch (error) {
      console.log("downloading model from URL:", MODEL_CONFIG.url);

      // download model if not in indexeddb
      const response = await fetch(MODEL_CONFIG.url);

      if (!response.ok) {
        postMessage({
          id,
          type: "error",
          name: "fetchModelError",
          data: `Failed to fetch model: ${response.status} ${response.statusText}`,
        });
      }

      const modelBlob = await response.blob();
      try {
        await set(MODEL_CONFIG.url, modelBlob);
      } catch (saveError) {
        postMessage({
          id,
          type: "error",
          name: "saveModelError",
          data: `Failed to save model: ${saveError}`,
        });
      }

      return URL.createObjectURL(modelBlob);
    }
  };

  try {
    modelSession = await ort.InferenceSession.create(await getModelUrl(), {
      executionProviders: modelOptions?.onnxExecutionProviders || ["wasm"],
    });
    isLoaded = true;

    const loadTime = performance.now() - startTime;
    console.log(`[TIMING] Model loaded in ${loadTime.toFixed(2)}ms`);

    postMessage({
      id,
      type: "success",
      name: "modelLoaded",
    });
  } catch (error) {
    postMessage({
      id,
      type: "error",
      name: "modelLoadError",
      data: `Failed to load model: ${error}`,
    });
  }
}

async function forward(
  id: number,
  input: string | string[],
  options?: { bosToken?: boolean }
) {
  const startTime = performance.now();

  if (!isLoaded || !modelSession || !tokenizer) {
    postMessage({
      id,
      type: "error",
      name: "modelNotLoaded",
      data: "Model is not loaded. Please load the model first.",
    });
    return;
  }

  try {
    // handle both single string or array of strings (batch)
    const inputs = Array.isArray(input) ? input : [input],
      batchTokenIds = inputs.map((text) => {
        return options?.bosToken
          ? [50256, ...tokenizer.encode(text)]
          : tokenizer.encode(text);
      }),
      maxTokLength = Math.max(...batchTokenIds.map((ids) => ids.length)),
      paddedBatch = batchTokenIds.map((ids) => {
        if (ids.length < maxTokLength)
          return [...ids, ...Array(maxTokLength - ids.length).fill(0)];
        return ids;
      }),
      inputTensor = new ort.Tensor("int64", paddedBatch.flat(), [
        inputs.length,
        maxTokLength,
      ]);

    const rawOutputs = await modelSession.run({
        input: inputTensor,
      }),
      formattedOutputs: GPT2Outputs = {
        embeddings: {
          tok_emb: rawOutputs["tok_emb"],
          pos_emb: rawOutputs["pos_emb"],
          input_emb: rawOutputs["input_emb"],
        },
        layers: [] as GPT2TransformerLayer[],
        final: {
          ln_f_output: rawOutputs["ln_f_output"],
          logits: rawOutputs["linear_output"],
        },
      };

    for (let i = 0; i < 12; i++) {
      const layerOutput: GPT2TransformerLayer = {
        block_input: rawOutputs[`block_${i}_block_input`],
        attn_input: rawOutputs[`block_${i}_attn_attn_input`],
        attn_heads: [] as GPT2AttentionHead[],
        attn_output: rawOutputs[`block_${i}_attn_attn_output`],
        mlp: {
          mlp_input: rawOutputs[`block_${i}_mlp_mlp_input`],
          mlp_activation: rawOutputs[`block_${i}_mlp_mlp_activation`],
          mlp_output: rawOutputs[`block_${i}_mlp_mlp_output`],
        },
        block_output: rawOutputs[`block_${i}_block_output`],
      };

      for (let j = 0; j < 12; j++) {
        layerOutput.attn_heads.push({
          q: rawOutputs[`block_${i}_attn_head_${j}_q`],
          k: rawOutputs[`block_${i}_attn_head_${j}_k`],
          v: rawOutputs[`block_${i}_attn_head_${j}_v`],
          attn_weight: rawOutputs[`block_${i}_attn_head_${j}_attn_softmax`],
          attn_value_output:
            rawOutputs[`block_${i}_attn_head_${j}_attn_value_output`],
        });
      }

      formattedOutputs.layers.push(layerOutput);
    }

    const loadTime = performance.now() - startTime;
    console.log(`[TIMING] Model loaded in ${loadTime.toFixed(2)}ms`);

    postMessage({
      id,
      type: "success",
      name: "forward",
      data: formattedOutputs,
    });
  } catch (error) {
    postMessage({
      id,
      type: "error",
      name: "forwardError",
      data: `Failed to do a forward pass: ${error}`,
    });
    return;
  }
}

function getTokenProbs(
  logits: Float32Array,
  options?: {
    temperature?: number;
    topP?: number;
    topK?: number;
  }
): Float32Array {
  const { temperature = 1.0, topP, topK } = options || {};

  // apply temperature scaling
  if (temperature !== 1.0 && temperature > 0) {
    logits = logits.map((logit) => logit / temperature);
  }

  // apply top-k
  if (topK && topK > 0 && topK < logits.length) {
    const indexed = Array.from(logits).map((logit, index) => ({
      logit,
      index,
    }));

    indexed.sort((a, b) => b.logit - a.logit);

    // keep only top-k indices in logits; set others to negative infinity
    const topKIndices = new Set(
      indexed.slice(0, topK).map((item) => item.index)
    );

    logits = logits.map((logit, index) =>
      topKIndices.has(index) ? logit : -Infinity
    );
  }

  let probabilities = (() => {
    const maxLogit = Math.max(
        ...Array.from(logits).filter((x) => x !== -Infinity)
      ),
      expLogits = logits.map((logit) => Math.exp(logit - maxLogit)),
      sumExp = expLogits.reduce((a, b) => a + b, 0);

    return new Float32Array(expLogits.map((logit) => logit / sumExp));
  })();

  // apply top-p
  if (topP && topP > 0 && topP < 1.0) {
    const indexed = Array.from(probabilities).map((prob, index) => ({
      prob,
      index,
    }));

    indexed.sort((a, b) => b.prob - a.prob);

    let cumulative = 0;

    // use cumulative to track current sum, and find top-p indices
    const topPIndices = new Set<number>();
    for (const item of indexed) {
      cumulative += item.prob;
      topPIndices.add(item.index);
      if (cumulative >= topP) break;
    }

    probabilities = probabilities.map((prob, index) =>
      topPIndices.has(index) ? prob : 0
    );

    // renormalize after zeroing out indices that were filtered out (conditional is to ensure no div by 0)
    const sum = probabilities.reduce((a, b) => a + b, 0);
    if (sum > 0) {
      probabilities = probabilities.map((prob) => prob / sum);
    }
  }

  return probabilities;
}

async function getTokens(id: number, probabilities: Float32Array) {
  const tokenProbs: Record<string, number> = {};

  for (let i = 0; i < probabilities.length; i++) {
    const token = tokenizer.decode([i]);
    tokenProbs[token] = probabilities[i];
  }

  postMessage({
    id,
    type: "success",
    name: "getTokens",
    data: Object.entries(tokenProbs).sort(
      ([, probA], [, probB]) => probB - probA
    ),
  });
}
