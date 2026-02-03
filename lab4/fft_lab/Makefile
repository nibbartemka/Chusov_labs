CXX ?= g++
CXXFLAGS ?= -O3
override CXXFLAGS += -std=c++20 -fopenmp
LDFLAGS = -pthread -ldl

OUTDIR ?= ../bin/

SRCS := main.cpp dft.cpp fft.cpp fft_multiprocessing.cpp
OBJS := $(addprefix obj/,$(SRCS:.cpp=.o))
DEPS := $(OBJS:.o=.d)

TARGET := $(OUTDIR)run

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS) | $(OUTDIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

obj/%.o: %.cpp | obj
	$(CXX) $(CXXFLAGS) -c -o $@ -MMD -MF $(@:.o=.d) $<

obj:
	mkdir -p obj

$(OUTDIR):
	mkdir -p $(OUTDIR)

clean:
	rm -rf obj $(TARGET)

-include $(DEPS)