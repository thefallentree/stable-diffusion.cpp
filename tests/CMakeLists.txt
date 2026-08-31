add_executable(test-h3-tokenizer test-h3-tokenizer.cpp)
target_link_libraries(test-h3-tokenizer PRIVATE stable-diffusion)

add_test(NAME test-h3-tokenizer COMMAND test-h3-tokenizer)

add_executable(test-name-conversion test-name-conversion.cpp)
target_link_libraries(test-name-conversion PRIVATE stable-diffusion)

add_test(NAME test-name-conversion COMMAND test-name-conversion)

add_executable(test-safetensors-index test-safetensors-index.cpp)
target_link_libraries(test-safetensors-index PRIVATE stable-diffusion)

add_test(NAME test-safetensors-index COMMAND test-safetensors-index)

add_executable(test-minimax-h3-layout test-minimax-h3-layout.cpp)
target_link_libraries(test-minimax-h3-layout PRIVATE stable-diffusion)

add_test(NAME test-minimax-h3-layout COMMAND test-minimax-h3-layout)
