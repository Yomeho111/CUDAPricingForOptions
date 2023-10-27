# Makefile for a C++ application

# Compiler and compiler flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra
FLAGS = -pthread


# Source files and target executable
SRCS = main.cpp Readfile.cpp PricingEngine.cpp
TARGET = myprogram

# Build rule for the target executable
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) $(FLAGS) -o $(TARGET)

# Clean rule to remove the executable and object files
clean:
	rm -f $(TARGET)