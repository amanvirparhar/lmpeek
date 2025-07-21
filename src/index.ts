import { ModelType } from "./types";

export async function loadModel(modelType: ModelType, tokenizerRepo?: string) {
  const worker = new Worker(`./workers/${modelType}.js`, { type: "module" });
  const pendingRequests = new Map();

  let messageId = 0;

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

  const loadId = ++messageId;

  worker.postMessage({
    id: loadId,
    type: "action",
    name: "loadModel",
    data: { tokenizerRepo },
  });

  await new Promise((resolve, reject) => {
    pendingRequests.set(loadId, { resolve, reject });
  });

  return (input: string, options?: { bosToken?: boolean }) => {
    const fwdPassId = ++messageId;

    worker.postMessage({
      id: fwdPassId,
      type: "action",
      name: "forward",
      data: {
        input,
        options,
      },
    });

    return new Promise((resolve, reject) => {
      pendingRequests.set(fwdPassId, { resolve, reject });
    });
  };
}
