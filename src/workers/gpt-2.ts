import { fromPreTrained } from "@lenml/tokenizer-gpt2";
import * as ort from "onnxruntime-web";
import { get, set } from "idb-keyval";

export interface GPT2AttentionHead {
  q: ort.Tensor;
  k: ort.Tensor;
  v: ort.Tensor;
  raw: ort.Tensor;
  scaled: ort.Tensor;
  masked: ort.Tensor;
  softmax: ort.Tensor;
}

export interface GPT2TransformerBlock {
  ln_1_output: ort.Tensor;
  attn_heads: {
    q: ort.Tensor;
    k: ort.Tensor;
    v: ort.Tensor;
    raw: ort.Tensor;
    scaled: ort.Tensor;
    masked: ort.Tensor;
    softmax: ort.Tensor;
  }[];
  attn_output: ort.Tensor;
  res_1: ort.Tensor;
  ln_2_output: ort.Tensor;
  mlp: {
    linear_1_output: ort.Tensor;
    gelu_output: ort.Tensor;
    linear_2_output: ort.Tensor;
    output: ort.Tensor;
  };
  res_2: ort.Tensor;
}

export interface GPT2Outputs {
  tok_emb: ort.Tensor;
  pos_emb: ort.Tensor;
  input_emb: ort.Tensor;
  blocks: GPT2TransformerBlock[];
  ln_f_output: ort.Tensor;
  linear_output: ort.Tensor;
}

const MODEL_CONFIG = {
  url: "https://huggingface.co/Amanvir/gpt-2-onnx-test/resolve/main/gpt2-all.onnx",
  cacheKey: "gpt2.onnx",
  wasmPaths: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist",
};

ort.env.wasm.wasmPaths = MODEL_CONFIG.wasmPaths;

let tokenizer: ReturnType<typeof fromPreTrained> | null = null,
  modelSession: ort.InferenceSession | null,
  isLoaded = false;

self.onmessage = async (e) => {
  const { type, name, data } = e.data;

  if (type == "action") {
    if (name === "loadModel") {
      await loadModel(data.id, data?.tokenizerRepo);
    } else if (name === "forward") {
      await forward(data.id, data.input, data?.options);
    }
  }
};

async function loadModel(id: number, tokenizerRepo?: string) {
  if (isLoaded) {
    postMessage({
      id,
      type: "error",
      name: "modelAlreadyLoaded",
    });
    return;
  }

  if (!tokenizerRepo) {
    tokenizer = fromPreTrained();
  } else {
    // TODO
  }

  const getModelUrl = async () => {
    try {
      // check if model is in indexeddb
      const file = await get("gpt2.onnx");
      if (!file) throw new Error("Model file not found in IDB.");
      return URL.createObjectURL(file);
    } catch (error) {
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
        await set("gpt2.onnx", modelBlob);
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
      executionProviders: ["wasm"],
    });
    isLoaded = true;

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
  input: string,
  options?: { bosToken?: boolean }
) {
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
    const tokenIds = options?.bosToken
      ? [50256, ...tokenizer.encode(input)]
      : tokenizer.encode(input);

    const rawOutputs = await modelSession.run({
        input: new ort.Tensor("int64", tokenIds, [1, tokenIds.length]),
      }),
      formattedOutputs: GPT2Outputs = {
        tok_emb: rawOutputs["tok_emb"],
        pos_emb: rawOutputs["pos_emb"],
        input_emb: rawOutputs["input_emb"],
        blocks: [] as GPT2TransformerBlock[],
        ln_f_output: rawOutputs["ln_f_output"],
        linear_output: rawOutputs["linear_output"],
      };

    for (let i = 0; i < 12; i++) {
      const blockOutput: GPT2TransformerBlock = {
        ln_1_output: rawOutputs[`block_${i}_ln_1_output`],
        attn_heads: [] as GPT2AttentionHead[],
        attn_output: rawOutputs[`block_${i}_attn_attn_output`],
        res_1: rawOutputs[`block_${i}_attn_res_1`],
        ln_2_output: rawOutputs[`block_${i}_mlp_ln_2_output`],
        mlp: {
          linear_1_output: rawOutputs[`block_${i}_mlp_linear_1_output`],
          gelu_output: rawOutputs[`block_${i}_mlp_gelu_output`],
          linear_2_output: rawOutputs[`block_${i}_mlp_linear_2_output`],
          output: rawOutputs[`block_${i}_mlp_output`],
        },
        res_2: rawOutputs[`block_${i}_res_2`],
      };

      for (let j = 0; j < 12; j++) {
        blockOutput.attn_heads.push({
          q: rawOutputs[`block_${i}_attn_head_${j}_q`],
          k: rawOutputs[`block_${i}_attn_head_${j}_k`],
          v: rawOutputs[`block_${i}_attn_head_${j}_v`],
          raw: rawOutputs[`block_${i}_attn_head_${j}_attn`],
          scaled: rawOutputs[`block_${i}_attn_head_${j}_attn_scaled`],
          masked: rawOutputs[`block_${i}_attn_head_${j}_attn_masked`],
          softmax: rawOutputs[`block_${i}_attn_head_${j}_attn_softmax`],
        });
      }

      formattedOutputs.blocks.push(blockOutput);
    }

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
