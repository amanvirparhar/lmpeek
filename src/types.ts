import { Tensor } from "onnxruntime-web";

// supported model types
export type ModelType = "gpt-2";

/* general model stuff */
export interface LoadModelOptions {
  onnxExecutionProviders?: string[];
  tokenizer?: string;
  logging?: boolean;
}

export interface ForwardOptions {
  bosToken?: boolean;
}

export interface SamplingOptions {
  temperature?: number;
  topP?: number;
  topK?: number;
}

export type SampleResult = [string, number][];

export interface ModelInstance {
  forward: (input: string, options?: ForwardOptions) => Promise<GPT2Outputs>;
  sample: (logits: Float32Array, options?: SamplingOptions) => Promise<SampleResult>;
  encode: (text: string) => Promise<number[]>;
  decode: (tokenIds: number[]) => Promise<string>;
  dispose: () => void;
}

/* gpt-2 model types */
export interface GPT2Embeddings {
  tok_emb: Tensor;
  pos_emb: Tensor;
  input_emb: Tensor;
}

export interface GPT2AttentionHead {
  q: Tensor;
  k: Tensor;
  v: Tensor;
  attn_weight: Tensor;
  attn_value_output: Tensor;
}

export interface GPT2MLP {
  mlp_input: Tensor;
  mlp_activation: Tensor;
  mlp_output: Tensor;
}

export interface GPT2TransformerLayer {
  block_input: Tensor;
  attn_input: Tensor;
  attn_heads: GPT2AttentionHead[];
  attn_output: Tensor;
  mlp: GPT2MLP;
  block_output: Tensor;
}

export interface GPT2FinalOutputs {
  ln_f_output: Tensor;
  logits: Tensor;
}

export interface GPT2Outputs {
  embeddings: GPT2Embeddings;
  layers: GPT2TransformerLayer[];
  final: GPT2FinalOutputs;
}