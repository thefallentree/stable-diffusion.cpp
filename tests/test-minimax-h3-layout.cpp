#include <cstdint>
#include <iostream>
#include <vector>

#include "model/diffusion/minimax_h3.hpp"

static TensorStorage tensor_storage(const std::string& name,
                                    int64_t columns,
                                    int64_t rows) {
    const int64_t shape[2] = {columns, rows};
    return TensorStorage(name, GGML_TYPE_F32, shape, 2, 0);
}

int main() {
    constexpr const char* prefix = "model.diffusion_model";

    String2TensorStorage direct_tensors;
    const std::string direct_name =
        "model.diffusion_model.blocks.0.adaln_schedule.weight";
    direct_tensors.insert({direct_name,
                           tensor_storage(direct_name,
                                          5376 * 6 * 3,
                                          MiniMaxH3::FASTH3_SCHEDULE_ROWS)});
    const auto direct = MiniMaxH3::Config::detect_from_weights(direct_tensors, prefix);
    if (!direct.uses_direct_adaln_schedule() || !direct.value_first_swiglu ||
        direct.uses_time_embedder()) {
        std::cerr << "FastH3 direct layout was not detected\n";
        return 1;
    }

    String2TensorStorage legacy_tensors;
    const std::string curve_name = "model.diffusion_model.adaln_t_table";
    legacy_tensors.insert({curve_name, tensor_storage(curve_name, 2688, 1000)});
    const auto legacy = MiniMaxH3::Config::detect_from_weights(legacy_tensors, prefix);
    if (!legacy.uses_adaln_curves() || legacy.uses_direct_adaln_schedule() ||
        legacy.value_first_swiglu || legacy.uses_time_embedder()) {
        std::cerr << "legacy H3 layout changed unexpectedly\n";
        return 1;
    }

    const std::vector<float> trained = {
        0.0000834098f,
        0.0003335557f,
        0.0271674432f,
        0.0769230798f,
        0.1004803851f,
        0.2000000030f,
        0.2500000000f,
        0.5000000000f,
    };
    const auto indices = MiniMaxH3::fasth3_schedule_indices(trained);
    if (indices.size() != trained.size()) {
        std::cerr << "FastH3 trained schedule was rejected\n";
        return 1;
    }
    for (size_t index = 0; index < indices.size(); ++index) {
        if (indices[index] != static_cast<int32_t>(index)) {
            std::cerr << "FastH3 schedule row mismatch\n";
            return 1;
        }
    }

    if (!MiniMaxH3::fasth3_schedule_indices({0.125f}).empty()) {
        std::cerr << "FastH3 accepted an untrained schedule row\n";
        return 1;
    }

    const auto video_sigmas = MiniMaxH3::fasth3_video_sigmas();
    if (video_sigmas.size() != 5 || video_sigmas.back() != 0.f) {
        std::cerr << "FastH3 default sigma schedule has the wrong shape\n";
        return 1;
    }
    for (size_t index = 0; index + 1 < video_sigmas.size(); ++index) {
        const float video_sigma = video_sigmas[index];
        const float audio_sigma = MiniMaxH3::time_shift_sigma(video_sigma, 12.f, 3.f);
        if (MiniMaxH3::fasth3_schedule_indices({1.f - video_sigma,
                                                1.f - audio_sigma})
                .size() != 2) {
            std::cerr << "FastH3 default sigmas do not address trained video/audio rows\n";
            return 1;
        }
    }

    return 0;
}
