import type { ModelType } from "./types";

// model configurations
export const MODEL_CONFIGS: Record<
  ModelType,
  {
    url: string;
  }
> = {
  "gpt-2": {
    url: "https://huggingface.co/Amanvir/gpt-2/resolve/main/model.onnx",
  },
};
