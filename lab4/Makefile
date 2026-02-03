ifeq ($(CXXFLAGS),)
CXXFLAGS = -O3
endif
ifeq ($(CXX),)
CXX = g++
endif

ENV = CXX=$(CXX) OUTDIR=../bin/ CXXFLAGS=$(CXXFLAGS)

.PHONY: all run clean

all: bin/run

bin/run:
	make -C fft_lab $(ENV)

run: bin/run
	./bin/run

clean:
	make -C fft_lab clean $(ENV)
	rm -rf bin/