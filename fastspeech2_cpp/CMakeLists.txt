cmake_minimum_required(VERSION 3.15)
project(FastSpeech2Cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB FASTSPEECH2_SOURCES
    "${SRC_DIR}/*.cpp"
)

add_executable(fastspeech2 ${FASTSPEECH2_SOURCES})
target_include_directories(fastspeech2 PRIVATE "${SRC_DIR}")

if (MSVC)
    target_compile_options(fastspeech2 PRIVATE /W4)
else()
    target_compile_options(fastspeech2 PRIVATE -Wall -Wextra)
endif()

find_package(Python3 COMPONENTS Interpreter)
if (Python3_Interpreter_FOUND)
    set(PARITY_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tools/parity/compare_runtime.py")
    set(DEFAULT_PHONEMES "23,15,8,32,45,12")
    add_custom_target(parity-check
        COMMAND ${Python3_EXECUTABLE} "${PARITY_SCRIPT}"
            --checkpoint "${CMAKE_CURRENT_SOURCE_DIR}/../output/ckpt/LJSpeech/900000.pth.tar"
            --preprocess_config "${CMAKE_CURRENT_SOURCE_DIR}/../config/LJSpeech/preprocess.yaml"
            --model_config "${CMAKE_CURRENT_SOURCE_DIR}/../config/LJSpeech/model.yaml"
            --weights_dir "${CMAKE_CURRENT_SOURCE_DIR}/weights"
            --runtime "$<TARGET_FILE:fastspeech2>"
            --phonemes "${DEFAULT_PHONEMES}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Running PyTorch vs C++ parity check..."
    )
endif()
