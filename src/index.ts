import { ModelType } from "./types";

export async function loadModel(
  modelType: ModelType,
  options?: {
    onnxExecutionProviders?: string[];
    tokenizer?: string;
  }
) {
  const path = `./models/${modelType}.mjs`,
    worker = new Worker(new URL(path, import.meta.url), {
      type: "module",
    }),
    pendingRequests = new Map();

  let messageId = 0;

  worker.onmessage = (e) => {
    console.log("Worker message received:", e.data);
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

  const loadId = ++messageId;

  worker.postMessage({
    type: "action",
    name: "loadModel",
    data: {
      id: loadId,
      options: options,
    },
  });

  await new Promise((resolve, reject) => {
    pendingRequests.set(loadId, { resolve, reject });
  });

  return {
    forward: (input: string, options?: { bosToken?: boolean }) => {
      const fwdPassId = ++messageId;

      worker.postMessage({
        type: "action",
        name: "forward",
        data: {
          id: fwdPassId,
          input,
          options,
        },
      });

      return new Promise((resolve, reject) => {
        pendingRequests.set(fwdPassId, { resolve, reject });
      });
    },
    sample: (
      logits: Float32Array,
      options?: {
        temperature?: number;
        topP?: number;
        topK?: number;
      }
    ) => {
      const sampleId = ++messageId;

      worker.postMessage({
        type: "action",
        name: "sample",
        data: {
          id: sampleId,
          logits,
          options,
        },
      });

      return new Promise((resolve, reject) => {
        pendingRequests.set(sampleId, { resolve, reject });
      });
    },
  };
}
