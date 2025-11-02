import { fromPreTrained } from "@lenml/tokenizer-gpt2";
import { AutoTokenizer, PreTrainedTokenizer } from "@xenova/transformers";
import { InferenceSession, Tensor, env } from "onnxruntime-web";
import { get, set } from "idb-keyval";
import type { GPT2Outputs, GPT2TransformerLayer, GPT2AttentionHead } from "../types";
import { MODEL_CONFIGS } from "../config";

const MODEL_CONFIG = MODEL_CONFIGS["gpt-2"];

let tokenizer:
  | ReturnType<typeof fromPreTrained>
  | PreTrainedTokenizer,
  modelSession: InferenceSession,
  isLoaded = false,
  loggingEnabled = false;

const log = (...args: any[]) => {
  if (loggingEnabled) console.log(...args);
};

const actions: Record<string, (data: any) => Promise<void>> = {
  loadModel: (data) => loadModel(data.id, data?.options),
  forward: (data) => forward(data.id, data.input, data?.options),
  sample: (data) => getTokens(data.id, getTokenProbs(data.logits, data?.options)),
  encode: (data) => encode(data.id, data.text),
  decode: (data) => decode(data.id, data.tokenIds),
};

self.onmessage = async (e) => {
  const { type, name, data } = e.data;
  if (type === "action" && name in actions) await actions[name](data);
};

async function loadModel(
  id: number,
  modelOptions?: {
    onnxExecutionProviders?: string[];
    tokenizer?: string;
    logging?: boolean;
  }
) {
  const startTime = performance.now();

  loggingEnabled = modelOptions?.logging ?? false;

  if (isLoaded) {
    postMessage({
      id,
      type: "error",
      name: "modelAlreadyLoaded",
      data: "Model is already loaded",
    });
    return;
  }

  if (!modelOptions?.tokenizer) {
    log("[lmpeek] Using default GPT-2 tokenizer");
    tokenizer = fromPreTrained();
  } else {
    try {
      tokenizer = await AutoTokenizer.from_pretrained(
        modelOptions.tokenizer
      );
      log(`[lmpeek] Loaded custom tokenizer: ${modelOptions.tokenizer}`);
    } catch (error) {
      postMessage({
        id,
        type: "error",
        name: "tokenizerLoadError",
        data: `Failed to load tokenizer: ${error}`,
      });
      return;
    }
  }

  const getModelUrl = async () => {
    try {
      const file = await get(MODEL_CONFIG.url);

      if (!file) throw new Error("Model file not found in IDB.");
      log("[lmpeek] Model found in cache");

      return URL.createObjectURL(file);
    } catch (error) {
      log("[lmpeek] Downloading model...");

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
    modelSession = await InferenceSession.create(await getModelUrl(), {
      executionProviders: modelOptions?.onnxExecutionProviders || ["wasm"],
    });
    isLoaded = true;

    const loadTime = performance.now() - startTime;
    log(`[lmpeek] Model loaded in ${loadTime.toFixed(0)}ms`);

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
    const inputs = Array.isArray(input) ? input : [input],
      batchTokenIds = inputs.map((text) => {
        const encoded = tokenizer.encode(text);
        const tokenIds = Array.isArray(encoded) ? encoded : Array.from(encoded);
        return options?.bosToken ? [50256, ...tokenIds] : tokenIds;
      }),
      maxTokLength = Math.max(...batchTokenIds.map((ids) => ids.length)),
      paddedBatch = batchTokenIds.map((ids) => {
        if (ids.length < maxTokLength)
          return [...ids, ...Array(maxTokLength - ids.length).fill(0)];
        return ids;
      }),
      inputTensor = new Tensor("int64", paddedBatch.flat(), [
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
    log(`[lmpeek] Forward pass completed in ${loadTime.toFixed(0)}ms`);

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
  inputLogits: Float32Array,
  options?: {
    temperature?: number;
    topP?: number;
    topK?: number;
  }
): Float32Array {
  const { temperature = 1.0, topP, topK } = options || {};

  let logits = new Float32Array(inputLogits);

  if (temperature !== 1.0 && temperature > 0) {
    logits = logits.map((logit) => logit / temperature);
  }

  if (topK && topK > 0 && topK < logits.length) {
    const indexed = Array.from(logits).map((logit, index) => ({
      logit,
      index,
    }));

    indexed.sort((a, b) => b.logit - a.logit);

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

  if (topP && topP > 0 && topP < 1.0) {
    const indexed = Array.from(probabilities).map((prob, index) => ({
      prob,
      index,
    }));

    indexed.sort((a, b) => b.prob - a.prob);

    let cumulative = 0;

    const topPIndices = new Set<number>();
    for (const item of indexed) {
      cumulative += item.prob;
      topPIndices.add(item.index);
      if (cumulative >= topP) break;
    }

    probabilities = probabilities.map((prob, index) =>
      topPIndices.has(index) ? prob : 0
    );

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
    const decoded = tokenizer.decode([i]),
      token = typeof decoded === "string" ? decoded : String(decoded);
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

async function encode(id: number, text: string) {
  if (!tokenizer) {
    postMessage({
      id,
      type: "error",
      name: "tokenizerNotLoaded",
      data: "Tokenizer is not loaded. Please load the model first.",
    });
    return;
  }

  try {
    const encoded = tokenizer.encode(text),
      tokenIds = Array.isArray(encoded) ? encoded : Array.from(encoded);

    postMessage({
      id,
      type: "success",
      name: "encode",
      data: tokenIds,
    });
  } catch (error) {
    postMessage({
      id,
      type: "error",
      name: "encodeError",
      data: `Failed to encode text: ${error}`,
    });
  }
}

async function decode(id: number, tokenIds: number[]) {
  if (!tokenizer) {
    postMessage({
      id,
      type: "error",
      name: "tokenizerNotLoaded",
      data: "Tokenizer is not loaded. Please load the model first.",
    });
    return;
  }

  try {
    const decoded = tokenizer.decode(tokenIds),
      text = typeof decoded === "string" ? decoded : String(decoded);

    postMessage({
      id,
      type: "success",
      name: "decode",
      data: text,
    });
  } catch (error) {
    postMessage({
      id,
      type: "error",
      name: "decodeError",
      data: `Failed to decode tokens: ${error}`,
    });
  }
}