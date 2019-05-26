CXX=g++
CXXFLAGS=-std=c++11
PYINCLUDE=$(shell python3-config --includes)
PYLDFLAGS=$(shell python3-config --ldflags)


UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin) # Max OS X
    LDFLAGS=-undefined dynamic_lookup -dynamiclib
endif
ifeq ($(UNAME_S),Linux) # Linux default
    LDFLAGS=-shared $(PYLDFLAGS)
endif


all: src/select_levels.o src/fill_SMAWK.o src/fill_quadratic.o src/fill_log_linear.o src/dynamic_prog.o src/Ckmeans.1d.dp.o src/Ckmeans_wrap.o
	$(CXX) $(CXXFLAGS) -fPIC $(LDFLAGS) $^ -o src/_Ckmeans.so
	rm -f src/*.o

src/Ckmeans_wrap.cxx: src/Ckmeans.i src/Ckmeans.1d.dp.h
	swig -python -py3 -c++ $<

src/Ckmeans_wrap.o: src/Ckmeans_wrap.cxx
	$(CXX) $(CXXFLAGS) -fPIC $(PYINCLUDE) -c $< -o $@

src/Ckmeans.1d.dp.o: src/Ckmeans.1d.dp.cpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

src/dynamic_prog.o: src/dynamic_prog.cpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

src/fill_log_linear.o: src/fill_log_linear.cpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

src/fill_quadratic.o: src/fill_quadratic.cpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

src/fill_SMAWK.o: src/fill_SMAWK.cpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

src/select_levels.o: src/select_levels.cpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

clean:
	rm -f src/Ckmeans.py
	rm -f src/Ckmeans_wrap.cxx
	rm -f src/_Ckmeans.so
	rm -f src/*.o
