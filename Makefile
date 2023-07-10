CXX = g++
CXXFLAGS = -std=c++17 -O3 -DHOMEGROWN -mcx16 -pthread -march=native -DNDEBUG
INCLUDES = -IParlayANN/parlaylib/include -I/home/ubuntu/boost_1_82_0
LIBS = -lboost_program_options

SRCS = doubling_dpc.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = doubling_dpc

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
