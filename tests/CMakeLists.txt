add_executable(test-h3-tokenizer test-h3-tokenizer.cpp)
target_link_libraries(test-h3-tokenizer PRIVATE stable-diffusion)

add_test(NAME test-h3-tokenizer COMMAND test-h3-tokenizer)
