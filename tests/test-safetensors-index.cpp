#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ggml.h"
#include "model_io/safetensors_io.h"
#include "model_loader.h"

namespace fs = std::filesystem;

struct TemporaryDirectory {
    fs::path path;

    ~TemporaryDirectory() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};

static TensorWriteInfo make_write_info(ggml_tensor* tensor) {
    TensorWriteInfo info;
    info.tensor = tensor;
    info.n_dims = ggml_n_dims(tensor);
    for (int dimension = 0; dimension < info.n_dims; ++dimension) {
        info.ne[dimension] = tensor->ne[dimension];
    }
    return info;
}

int main() {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    TemporaryDirectory temporary{
        fs::temp_directory_path() / ("sdcpp-safetensors-index-" + std::to_string(unique))};
    fs::create_directories(temporary.path);

    ggml_init_params init_params = {};
    init_params.mem_size         = 4096;
    ggml_context* context        = ggml_init(init_params);
    if (context == nullptr) {
        std::cerr << "failed to create ggml context\n";
        return 1;
    }

    ggml_tensor* keep = ggml_new_tensor_1d(context, GGML_TYPE_F32, 2);
    ggml_tensor* drop = ggml_new_tensor_1d(context, GGML_TYPE_F32, 2);
    ggml_tensor* h3_video_input = ggml_new_tensor_1d(context, GGML_TYPE_F32, 2);
    ggml_tensor* h3_audio_input = ggml_new_tensor_1d(context, GGML_TYPE_F32, 2);
    ggml_set_name(keep, "keep");
    ggml_set_name(drop, "drop");
    ggml_set_name(h3_video_input, "proj_in.weight");
    ggml_set_name(h3_audio_input, "audio_proj_in.weight");
    static_cast<float*>(keep->data)[0] = 1.f;
    static_cast<float*>(keep->data)[1] = 2.f;
    static_cast<float*>(drop->data)[0] = 3.f;
    static_cast<float*>(drop->data)[1] = 4.f;
    static_cast<float*>(h3_video_input->data)[0] = 5.f;
    static_cast<float*>(h3_video_input->data)[1] = 6.f;
    static_cast<float*>(h3_audio_input->data)[0] = 7.f;
    static_cast<float*>(h3_audio_input->data)[1] = 8.f;

    const fs::path shard = temporary.path / "shard.safetensors";
    std::string error;
    const std::vector<TensorWriteInfo> tensors = {
        make_write_info(keep),
        make_write_info(drop),
        make_write_info(h3_video_input),
        make_write_info(h3_audio_input),
    };
    if (!write_safetensors_file(shard.string(), tensors, &error)) {
        std::cerr << error << '\n';
        ggml_free(context);
        return 1;
    }
    ggml_free(context);

    const fs::path index = temporary.path / "model.safetensors.index.json";
    {
        std::ofstream output(index);
        output << R"({"metadata":{"total_size":8},"weight_map":{"keep":"shard.safetensors"}})";
    }

    ModelLoader loader;
    if (!loader.init_from_file(index.string(), "model.")) {
        std::cerr << "failed to load valid safetensors index\n";
        return 1;
    }
    const auto& storage = loader.get_tensor_storage_map();
    if (storage.size() != 1 || storage.find("model.keep") == storage.end() ||
        storage.find("model.drop") != storage.end()) {
        std::cerr << "safetensors index leaked an unindexed shard tensor\n";
        return 1;
    }

    const fs::path invalid_index = temporary.path / "invalid.safetensors.index.json";
    {
        std::ofstream output(invalid_index);
        output << R"({"weight_map":{"missing":"shard.safetensors"}})";
    }
    ModelLoader invalid_loader;
    if (invalid_loader.init_from_file(invalid_index.string())) {
        std::cerr << "safetensors index accepted a missing tensor\n";
        return 1;
    }

    const fs::path h3_index = temporary.path / "h3.safetensors.index.json";
    {
        std::ofstream output(h3_index);
        output << R"({"weight_map":{"proj_in.weight":"shard.safetensors","audio_proj_in.weight":"shard.safetensors"}})";
    }
    ModelLoader h3_loader;
    if (!h3_loader.init_from_file(h3_index.string(), "model.diffusion_model.")) {
        std::cerr << "failed to load MiniMax-H3 diffusers names\n";
        return 1;
    }
    h3_loader.convert_tensors_name();
    const auto& h3_storage = h3_loader.get_tensor_storage_map();
    if (h3_storage.find("model.diffusion_model.video_patch_proj.weight") == h3_storage.end() ||
        h3_storage.find("model.diffusion_model.audio_patch_proj.weight") == h3_storage.end()) {
        std::cerr << "failed to detect and convert MiniMax-H3 diffusers names\n";
        return 1;
    }

    const int64_t boundary_shape[2] = {5376, 96};
    const TensorStorage boundary(
        "model.diffusion_model.final_layer.video_out.weight",
        GGML_TYPE_BF16,
        boundary_shape,
        2,
        0);
    if (h3_loader.tensor_should_be_converted(boundary, GGML_TYPE_F16) ||
        !h3_loader.tensor_should_be_converted(boundary, GGML_TYPE_F16, true)) {
        std::cerr << "explicit tensor-type rule did not override conversion policy\n";
        return 1;
    }

    return 0;
}
