#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "name_conversion.h"

int main() {
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"lora.transformer_blocks.7.attn.to_q.lora_A.weight",
         "lora.model.diffusion_model.blocks.7.attn.qkv_proj.weight.lora_down"},
        {"lora.transformer_blocks.7.attn.to_k.lora_B.weight",
         "lora.model.diffusion_model.blocks.7.attn.qkv_proj.weight.1.lora_up"},
        {"lora.transformer_blocks.7.attn.to_v.lora_B.weight",
         "lora.model.diffusion_model.blocks.7.attn.qkv_proj.weight.2.lora_up"},
        {"lora.transformer_blocks.7.attn.to_out.0.lora_A.weight",
         "lora.model.diffusion_model.blocks.7.attn.out_proj.weight.lora_down"},
        {"lora.transformer_blocks.7.ff.net.0.proj.lora_A.weight",
         "lora.model.diffusion_model.blocks.7.mlp.fc1.weight.lora_down"},
        {"lora.transformer_blocks.7.ff.net.2.lora_B.weight",
         "lora.model.diffusion_model.blocks.7.mlp.fc2.weight.lora_up"},
        {"lora.transformer_blocks.7.adaln_proj.linear.diff_b",
         "lora.model.diffusion_model.blocks.7.adaln_proj.linear.bias.diff"},
        {"lora.transformer_blocks.7.adaln_schedule.diff",
         "lora.model.diffusion_model.blocks.7.adaln_schedule.weight.diff"},
        {"lora.norm_out.adaln_schedule.diff",
         "lora.model.diffusion_model.final_layer.adaln_schedule.weight.diff"},
        {"lora.diffusion_model.blocks.0.adaln_schedule.diff",
         "lora.model.diffusion_model.blocks.0.adaln_schedule.weight.diff"},
        {"lora.diffusion_model.final_layer.adaln_schedule.diff",
         "lora.model.diffusion_model.final_layer.adaln_schedule.weight.diff"},
        {"lora.token_refiner.refiner_blocks.1.attn.to_v.lora_A.weight",
         "lora.model.diffusion_model.token_refiner.blocks.1.attn.qkv_proj.weight.2.lora_down"},
        {"lora.token_refiner.refiner_blocks.1.ff.net.2.lora_B.weight",
         "lora.model.diffusion_model.token_refiner.blocks.1.mlp.fc2.weight.lora_up"},
        {"lora.audio_proj_in.diff_b",
         "lora.model.diffusion_model.audio_patch_proj.bias.diff"},
        {"lora.audio_proj_out.diff",
         "lora.model.diffusion_model.final_layer.audio_out.weight.diff"},
        {"lora.context_embedder.diff",
         "lora.model.diffusion_model.condition_proj.weight.diff"},
        {"lora.norm_out.linear.diff",
         "lora.model.diffusion_model.final_layer.adaln_proj.linear.weight.diff"},
        {"lora.norm_out.norm.diff",
         "lora.model.diffusion_model.final_layer.norm.weight.diff"},
        {"lora.proj_in.diff",
         "lora.model.diffusion_model.video_patch_proj.weight.diff"},
        {"lora.proj_out.diff_b",
         "lora.model.diffusion_model.final_layer.video_out.bias.diff"},
        {"lora.time_embedder.linear_1.diff",
         "lora.model.diffusion_model.time_embedder.proj_in.weight.diff"},
        {"lora.diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight",
         "lora.model.diffusion_model.blocks.0.attn.qkv_proj.weight.lora_down"},
        {"model.diffusion_model.transformer_blocks.7.attn.qkv_proj.weight",
         "model.diffusion_model.blocks.7.attn.qkv_proj.weight"},
        {"model.diffusion_model.token_refiner.refiner_blocks.1.attn.qkv_proj.weight",
         "model.diffusion_model.token_refiner.blocks.1.attn.qkv_proj.weight"},
        {"model.diffusion_model.rope.inv_freq",
         "model.diffusion_model.rope.inv_freq"},
    };

    bool ok = true;
    for (const auto& [input, expected] : cases) {
        const std::string actual = convert_tensor_name(input, VERSION_MINIMAX_H3);
        if (actual == expected) {
            continue;
        }
        std::cerr << "name conversion mismatch\ninput:    " << input
                  << "\nexpected: " << expected
                  << "\nactual:   " << actual << '\n';
        ok = false;
    }

    return ok ? 0 : 1;
}
