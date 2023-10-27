# Compiler and compiler flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall -Wextra
NVCCFLAGS = -std=c++17

# Source files and target executable
SRCS = main.cpp readfile.cpp PricingEngine.cu
OBJS = $(SRCS:.cu=.o)
TARGET = myprogram

# Build rule for the target executable
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $(TARGET)

# Rules for compiling C++ source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rules for compiling CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean rule to remove the executable and object files
clean:
	rm -f $(TARGET) $(OBJS)