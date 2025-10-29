# FastSpeech2 C++ Implementation Makefile

CC = g++
CFLAGS = -O3 -std=c++11 -Wall -Wextra
DEBUG_CFLAGS = -g -O0 -std=c++11 -Wall -Wextra -DDEBUG
LDFLAGS = -lm

# Directories
TEST_DIR = test
TOOLS_DIR = tools
DEBUG_DIR = debug
DATA_DIR = test/data

# Source files
SOURCES = fastspeech2.cpp op.cpp
HEADERS = fastspeech2.h op.h
MAIN = main.cpp

# Executables
EXEC = fastspeech2
EXEC_DEBUG = fastspeech2_debug
TEST_OPS = test_ops
TEST_COMPONENTS = test_components
TEST_INTEGRATION = test_integration

.PHONY: all clean build test-data test-ops test-components test-integration test debug help

# Default target
all: build

# Build main executable
build: $(MAIN) $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(MAIN) $(SOURCES) -o $(EXEC) $(LDFLAGS)
	@echo "✓ Built $(EXEC)"

# Build debug version
debug: $(MAIN) $(SOURCES) $(HEADERS)
	$(CC) $(DEBUG_CFLAGS) $(MAIN) $(SOURCES) -o $(EXEC_DEBUG) $(LDFLAGS)
	@echo "✓ Built $(EXEC_DEBUG) with debug symbols"

# Generate test data from PyTorch
test-data:
	@echo "Generating operator test data..."
	python3 $(TOOLS_DIR)/generate_op_test_data.py
	@echo "Extracting component outputs..."
	python3 $(TOOLS_DIR)/extract_component_outputs.py
	@echo "Generating integration test data..."
	python3 $(TOOLS_DIR)/generate_integration_test_data.py
	@echo "✓ All test data generated"

# Build and run operator unit tests
test-ops: $(TEST_DIR)/test_ops.cpp $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(TEST_DIR)/test_ops.cpp $(SOURCES) -o $(TEST_OPS) $(LDFLAGS)
	@echo "Running operator tests..."
	./$(TEST_OPS)

# Build and run component tests
test-components: $(TEST_DIR)/test_components.cpp $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(TEST_DIR)/test_components.cpp $(SOURCES) -o $(TEST_COMPONENTS) $(LDFLAGS)
	@echo "Running component tests..."
	./$(TEST_COMPONENTS)

# Build and run integration tests
test-integration: $(TEST_DIR)/test_integration.cpp $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(TEST_DIR)/test_integration.cpp $(SOURCES) -o $(TEST_INTEGRATION) $(LDFLAGS)
	@echo "Running integration tests..."
	./$(TEST_INTEGRATION)

# Run all tests
test: test-ops test-components test-integration
	@echo ""
	@echo "✓ All tests completed!"

# Convert PyTorch weights to binary format
convert-weights:
	@echo "Converting PyTorch weights to binary format..."
	python3 $(TOOLS_DIR)/convert_weights.py
	@echo "✓ Weights converted"

# Clean build artifacts
clean:
	rm -f $(EXEC) $(EXEC_DEBUG) $(TEST_OPS) $(TEST_COMPONENTS) $(TEST_INTEGRATION)
	rm -f *.o
	rm -rf $(DEBUG_DIR)/*.bin $(DEBUG_DIR)/*.png
	@echo "✓ Cleaned build artifacts"

# Clean everything including test data
clean-all: clean
	rm -rf $(DATA_DIR)/*.bin $(DATA_DIR)/*.txt
	@echo "✓ Cleaned all generated files"

# Help target
help:
	@echo "FastSpeech2 C++ Implementation - Available targets:"
	@echo ""
	@echo "  make build              - Build main executable"
	@echo "  make debug              - Build debug version with symbols"
	@echo "  make test-data          - Generate test data from PyTorch"
	@echo "  make test-ops           - Run operator unit tests"
	@echo "  make test-components    - Run component validation tests"
	@echo "  make test-integration   - Run end-to-end integration tests"
	@echo "  make test               - Run all tests"
	@echo "  make convert-weights    - Convert PyTorch weights to binary"
	@echo "  make clean              - Clean build artifacts"
	@echo "  make clean-all          - Clean everything including test data"
	@echo "  make help               - Show this help message"
	@echo ""
