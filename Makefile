CXX = g++
CXXFLAGS = -std=c++17 -O3 -DHOMEGROWN -mcx16 -pthread -march=native -DNDEBUG
INCLUDES = -IParlayANN/parlaylib/include -I/home/ubuntu/boost_1_83_0
LIBS = -lboost_program_options

SRCS = doubling_dpc.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = doubling_dpc

DEBUG_CXXFLAGS = -std=c++17 -DPARLAY_SEQUENTIAL -mcx16 -pthread -march=native -g -O0 -DDEBUG   # Debug-specific flags

.PHONY: all clean debug

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

debug: CXXFLAGS = $(DEBUG_CXXFLAGS)   # Add debug flags to CXXFLAGS for the debug target
debug: $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
