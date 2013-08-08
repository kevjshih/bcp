CC=g++
OBJ = src/classifier.o src/config.o src/hog_features.o src/hypothesis.o src/model.o src/object_hypothesis.o src/poselet.o src/poselet_api.o src/poselet_cluster.o src/poselet_detector.o src/poselet_hit.o include/gil_draw.o poselets_test/main.o

# MATLAB = /Applications/MATLAB_R2011a.app
BOOST = ../../third-party/src/boost/boost-1.48.0/boost_1_48_0

ifdef MATLAB
  MEXOBJ = poselets_mex/main.o
  MEXINC = -I$(MATLAB)/extern/include
  MEXEXT = $(shell $(MATLAB)/bin/mexext)
  MEXLDFLAGS = -L. -L/usr/lib/gcc/i686-apple-darwin10/4.2.1/x86_64 -lgomp
  MEXOUT = poselets_mex.$(MEXEXT)
  MEXSRC=$(MEXOBJ:.o=.cpp)
endif

CFLAGS  = -O3 -Wall -fPIC #-fopenmp
INCLUDES = -Isrc -Iinclude -Ithird_party/rapidxml-1.13 -I$(BOOST) -Ithird_party
TESTLDFLAGS = -Llib -ljpeg
OUT = lib/libposelet.a
DEPS =

SRC=$(OBJ:.o=.cpp)

ifdef MATLAB
$(MEXOUT) : $(OUT) $(MEXOBJ)
	$(MATLAB)/bin/mex $(INCLUDES) $(MEXINC) $(MEXLDFLAGS) -lposelet $(MEXSRC)

$(MEXOBJ) : $(MEXSRC)
	$(CC) $(CFLAGS) $(INCLUDES) $(MEXINC) -c -o $@ $<
endif

$(OUT): $(OBJ)
	ar rcs $(OUT) $(OBJ)

%.o: %.cpp $(DEPS)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

test : $(OUT)
	$(CC) $(CFLAGS) $(INCLUDES) $(TESTLDFLAGS) poselets_test/main.cpp include/gil_imdebug.cpp -lposelet -o poselets_test/poselets_test

clean:
	rm -f $(OBJ) $(OUT) $(MEXOUT)
