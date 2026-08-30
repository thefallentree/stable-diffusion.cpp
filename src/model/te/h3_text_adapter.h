#ifndef __SD_MODEL_TE_H3_TEXT_ADAPTER_H__
#define __SD_MODEL_TE_H3_TEXT_ADAPTER_H__

#include "model/common/block.hpp"

namespace LLM {

class H3TextAdapter : public GGMLBlock {
public:
    static constexpr int64_t INPUT_DIM  = 2560;
    static constexpr int64_t HIDDEN_DIM = 4096;
    static constexpr int64_t OUTPUT_DIM = 5120;

    H3TextAdapter() {
        blocks["net.0"] = std::make_shared<Linear>(INPUT_DIM, HIDDEN_DIM, true, true, true);
        blocks["net.2"] = std::make_shared<Linear>(HIDDEN_DIM, OUTPUT_DIM, true, true, true);
    }

    ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
        GGML_ASSERT(x->ne[0] == INPUT_DIM);
        if (x->type != GGML_TYPE_F32) {
            x = ggml_ext_cast_f32(ctx->ggml_ctx, ctx->backend, x);
        }
        auto input_projection  = std::dynamic_pointer_cast<Linear>(blocks["net.0"]);
        auto output_projection = std::dynamic_pointer_cast<Linear>(blocks["net.2"]);

        x = input_projection->forward(ctx, x);
        x = ggml_gelu_erf_inplace(ctx->ggml_ctx, x);
        x = output_projection->forward(ctx, x);
        GGML_ASSERT(x->ne[0] == OUTPUT_DIM);
        return x;
    }
};

}  // namespace LLM

#endif  // __SD_MODEL_TE_H3_TEXT_ADAPTER_H__
