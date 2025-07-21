import torch
from model import GPT  # import model
import os


class wrapper(torch.nn.Module):
    def __init__(self, model):
        super(wrapper, self).__init__()
        self.model = model
        self.layer_num = self.model.config.n_layer
        self.head_num = self.model.config.n_head
        self.extracted_outputs = []

    def forward(self, input):
        outputs = self.model(input)

        # extract values from dictionary and return them as separate outputs
        self.extracted_outputs = []

        # add embedding outputs <aman>
        self.extracted_outputs.extend([
            outputs["embedding"]["tok_emb"],
            outputs["embedding"]["pos_emb"],
            outputs["embedding"]["input_emb"]
        ])

        for i in range(self.layer_num):
            # add layer norm 1 output <aman>
            self.extracted_outputs.extend([
                outputs["block"][f"block_{i}"]["ln_1"]["output"]
            ])

            # add q, k, v outputs for each head <aman>
            for j in range(self.head_num):
                self.extracted_outputs.extend([
                    outputs["block"][f"block_{i}"]["attn"][f"head_{j}"]["q"],
                    outputs["block"][f"block_{i}"]["attn"][f"head_{j}"]["k"],
                    outputs["block"][f"block_{i}"]["attn"][f"head_{j}"]["v"]
                ])

            for j in range(self.head_num):
                self.extracted_outputs.extend([
                    outputs["block"][f"block_{i}"]["attn"][f"head_{j}"]["attn"],
                    # added scaled and masked attention outputs <aman>
                    outputs["block"][f"block_{i}"]["attn"][f"head_{j}"]["attn_scaled"],
                    outputs["block"][f"block_{i}"]["attn"][f"head_{j}"]["attn_masked"],
                    outputs["block"][f"block_{i}"]["attn"][f"head_{j}"]["attn_softmax"],
                    # dropout is 0 in eval, so we can skip it <aman>
                    # outputs["block"][f"block_{i}"]["attn"][f"head_{j}"]["attn_dropout"]
                ])

            # add attention output, residual, layer norm 2, mlp outputs <aman>
            self.extracted_outputs.extend([
                outputs["block"][f"block_{i}"]["attn"]["attn_output"],
                outputs["block"][f"block_{i}"]["res_1"],
                outputs["block"][f"block_{i}"]["ln_2"]["output"],
                outputs["block"][f"block_{i}"]["mlp"]["linear_1_output"],
                outputs["block"][f"block_{i}"]["mlp"]["gelu_output"],
                outputs["block"][f"block_{i}"]["mlp"]["linear_2_output"],
                outputs["block"][f"block_{i}"]["mlp"]["output"],
                outputs["block"][f"block_{i}"]["res_2"]
            ])

        # add final layer norm and linear output <aman>
        self.extracted_outputs.extend([
            outputs["ln_f"]["output"],
            outputs["linear"]["output"]
        ])

        return tuple(self.extracted_outputs)


# initialize model
model = GPT.from_pretrained("gpt2")
model.eval()
wrapped_model = wrapper(model)

# initialize with embedding outputs <aman>
output_names = ["tok_emb", "pos_emb", "input_emb"]

for i in range(model.config.n_layer):
    output_names.append(f"block_{i}_ln_1_output")

    for j in range(model.config.n_head):
        output_names.extend([
            f"block_{i}_attn_head_{j}_q",
            f"block_{i}_attn_head_{j}_k",
            f"block_{i}_attn_head_{j}_v"
        ])

    for j in range(model.config.n_head):
        output_names.extend([
            f"block_{i}_attn_head_{j}_attn",
            f"block_{i}_attn_head_{j}_attn_scaled",
            f"block_{i}_attn_head_{j}_attn_masked",
            f"block_{i}_attn_head_{j}_attn_softmax",
            # dropout is 0 in eval, so we can skip it <aman>
            # f"block_{i}_attn_head_{j}_attn_dropout"
        ])

    output_names.extend([
        f"block_{i}_attn_attn_output",
        f"block_{i}_res_1",
        f"block_{i}_ln_2_output",
        f"block_{i}_mlp_linear_1_output",
        f"block_{i}_mlp_gelu_output",
        f"block_{i}_mlp_linear_2_output",
        f"block_{i}_mlp_output",
        f"block_{i}_res_2"
    ])

output_names.extend(["ln_f_output", "linear_output"])

# do a dummy forward pass, and assert length of output_names matches actual number of self.extracted_outputs <aman>
# with torch.no_grad():
#     wrapped_model(torch.tensor([[6601, 32704, 795, 30132, 2985, 284]]))
#     assert len(output_names) == len(wrapped_model.extracted_outputs), \
#         f"Output names length does not match extracted outputs length: {len(output_names)} != {len(wrapped_model.extracted_outputs)}"
# wrapped_model = wrapper(model)

dummy_input = torch.tensor([[6601, 32704, 795, 30132, 2985, 284]])

torch.onnx.export(
    wrapped_model,
    dummy_input,
    "gpt2.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=output_names,
    dynamic_axes={
        'input': {0: '0', 1: '1'},
        **{
            name: {0: '0', 1: '1', 2: '2'}
            for name in output_names if name != "linear_output"
        },
        'linear_output': {0: '0', 1: '1', 2: '2'}
    }
)

print("Model has been successfully exported to ONNX format.")
