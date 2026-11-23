SHELL := /bin/bash
.PHONY: test host dsp build clean

test:
	./run_tests.sh

host:
	LFM_SIM_MODES=host ./run_tests.sh

dsp:
	LFM_SIM_MODES=dsp ./run_tests.sh

build:
	./sim/build.sh

clean:
	rm -rf sim/build
