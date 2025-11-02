import type {
  ModelType,
  LoadModelOptions,
  ForwardOptions,
  SamplingOptions,
  ModelInstance,
  GPT2Outputs,
  SampleResult,
} from "./types";

interface PendingRequest<T> {
  resolve: (value: T) => void;
  reject: (error: Error) => void;
}

/**
 * Load a language model for inference in a web worker
 * @param modelType - The type of model to load (currently only 'gpt-2')
 * @param options - Configuration options
 * @returns A model instance with forward pass, sampling, encoding, decoding, and disposal methods
 * @example
 * ```typescript
 * const model = await loadModel('gpt-2');
 * 
 * // Use the worker's tokenizer
 * const tokenIds = await model.encode('Hello world');
 * const text = await model.decode(tokenIds);
 * 
 * // Run forward pass and sample
 * const outputs = await model.forward('Hello world');
 * const tokens = await model.sample(outputs.final.logits.data as Float32Array);
 * 
 * // Clean up when done
 * model.dispose();
 * ```
 */
export async function loadModel(
  modelType: ModelType,
  options?: LoadModelOptions
): Promise<ModelInstance> {
  const path = `./models/${modelType}.mjs`,
    worker = new Worker(new URL(path, import.meta.url), {
      type: "module",
    }),
    pendingRequests = new Map<number, PendingRequest<any>>();

  let messageId = 0;

  const loggingEnabled = options?.logging ?? false,
    log = (...args: any[]) => {
      if (loggingEnabled) console.log(...args);
    };

  worker.onmessage = (e) => {
    const { id, type, name, data } = e.data,
      request = pendingRequests.get(id);

    if (request) {
      pendingRequests.delete(id);
      if (type === "error") {
        request.reject(new Error(`Error: ${name} - ${data}`));
      } else if (type === "success") {
        request.resolve(data);
      }
    }
  };

  worker.onerror = (error) => {
    log("[lmpeek] Worker encountered an error:", error.message);
    pendingRequests.forEach((request) => {
      request.reject(
        new Error(`Worker error: ${error.message || "Unknown worker error"}`)
      );
    });
    pendingRequests.clear();
  };

  const loadId = ++messageId;

  worker.postMessage({
    type: "action",
    name: "loadModel",
    data: {
      id: loadId,
      options: options,
    },
  });

  await new Promise<void>((resolve, reject) => {
    pendingRequests.set(loadId, { resolve, reject });
  });

  const createRequest = <T>(action: string, data: any): Promise<T> => {
    const id = ++messageId;

    worker.postMessage({
      type: "action",
      name: action,
      data: { id, ...data },
    });

    return new Promise<T>((resolve, reject) => {
      pendingRequests.set(id, { resolve, reject });
    });
  };

  return {
    /**
     * Perform a forward pass through the model
     * @param input - Input text string
     * @param options - Forward pass options
     * @returns Promise resolving to model outputs including embeddings, layers, and logits
     */
    forward: (input: string, options?: ForwardOptions): Promise<GPT2Outputs> =>
      createRequest<GPT2Outputs>("forward", { input, options }),
    /**
     * Sample tokens from logits with optional temperature, top-p, and top-k filtering
     * @param logits - Float32Array of logits from model output
     * @param options - Sampling options
     * @returns Promise resolving to array of [token, probability] tuples sorted by probability
     */
    sample: (logits: Float32Array, options?: SamplingOptions): Promise<SampleResult> =>
      createRequest<SampleResult>("sample", { logits, options }),
    /**
     * Encode text to token IDs using the worker's tokenizer
     * @param text - Text to encode
     * @returns Promise resolving to array of token IDs
     */
    encode: (text: string): Promise<number[]> =>
      createRequest<number[]>("encode", { text }),
    /**
     * Decode token IDs to text using the worker's tokenizer
     * @param tokenIds - Array of token IDs to decode
     * @returns Promise resolving to decoded text
     */
    decode: (tokenIds: number[]): Promise<string> =>
      createRequest<string>("decode", { tokenIds }),
    dispose: () => {
      pendingRequests.forEach((request) => {
        request.reject(new Error("Model disposed"));
      });
      pendingRequests.clear();
      worker.terminate();
    },
  };
}