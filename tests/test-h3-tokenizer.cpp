#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "tokenizers/qwen2_tokenizer.h"

static bool expect_tokens(MiniMaxH3Tokenizer& tokenizer,
                          const std::string& text,
                          const std::vector<int>& expected) {
    const std::vector<int> actual = tokenizer.encode(text);
    if (actual == expected) {
        return true;
    }

    std::cerr << "token mismatch for '" << text << "'\nexpected:";
    for (int token : expected) {
        std::cerr << ' ' << token;
    }
    std::cerr << "\nactual:";
    for (int token : actual) {
        std::cerr << ' ' << token;
    }
    std::cerr << '\n';
    return false;
}

int main() {
    MiniMaxH3Tokenizer tokenizer;
    bool ok = true;

    const std::vector<std::pair<std::string, int>> special_tokens = {
        {"<|endoftext|>", 151643},
        {"<|im_start|>", 151644},
        {"<|im_end|>", 151645},
        {"<think>", 151667},
        {"</think>", 151668},
        {"<d>", 151669},
        {"</d>", 151670},
        {"<|cutoff|>", 151671},
        {"<|lyrics_start|>", 151672},
        {"<|lyrics_end|>", 151673},
        {"<|caption_start|>", 151674},
        {"<|caption_end|>", 151675},
    };
    for (const auto& [text, token] : special_tokens) {
        ok = expect_tokens(tokenizer, text, {token}) && ok;
    }

    ok = expect_tokens(tokenizer,
                       "A cinematic fox runs through neon rain.",
                       {32, 64665, 38835, 8473, 1526, 46652, 11174, 13}) && ok;
    ok = expect_tokens(tokenizer,
                       "镜头切换到海边，风吹动她的头发。",
                       {105995, 110697, 26939, 113146, 3837, 109520, 27733, 104007, 104994, 1773}) && ok;
    ok = expect_tokens(tokenizer,
                       "Robot 🤖 says café at 00:03.500!",
                       {43374, 11162, 97, 244, 2727, 51950, 518, 220, 15, 15, 25, 15, 18, 13, 20, 15, 15, 0}) && ok;
    ok = expect_tokens(tokenizer,
                       "<|lyrics_start|>la la la<|lyrics_end|>",
                       {151672, 4260, 1187, 1187, 151673}) && ok;
    ok = expect_tokens(tokenizer,
                       "Use <d>00:01.250</d> then <|cutoff|>.",
                       {10253, 220, 151669, 15, 15, 25, 15, 16, 13, 17, 20, 15, 151670, 1221, 220, 151671, 13}) && ok;

    if (tokenizer.EOS_TOKEN_ID != 151645 || tokenizer.PAD_TOKEN_ID != 151643) {
        std::cerr << "unexpected MiniMax-H3 EOS or PAD token id\n";
        ok = false;
    }

    Qwen2Tokenizer qwen_tokenizer;
    if (qwen_tokenizer.encode("<|boi_token|>") != std::vector<int>{151669}) {
        std::cerr << "default Qwen tokenizer special-token mapping regressed\n";
        ok = false;
    }

    return ok ? 0 : 1;
}
