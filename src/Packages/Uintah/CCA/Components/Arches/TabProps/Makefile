# define some required dirs

#--------------------------------------------
# TabProps Lite for SCIRun
HDF5_INC = /opt/hdf5-1.6.7/include
HDF5_LIB = /opt/hdf5-1.6.7/lib

BOOST_INC = /opt/boost-1.36.0
BOOST_LIB = /opt/boost-1.36.0

#CXX_FLAGS = -O3 -Wall -fexpensive-optimizations -funroll-loops
CXX_FLAGS = -g -Wall
CXX = g++
#--------------------------------------------

BASE = .
BASE_SRC = $(BASE)
BASE_INC = $(BASE)

INC  = -I$(BASE_INC) -I$(HDF5_INC) -I$(BOOST_INC) 
LIBS = -lhdf5 -lblas -llapack
COMPILE_CXX = $(CXX) $(CXX_FLAGS) $(INC) -c
LINK = $(CXX) $(CXX_FLAGS) -L$(HDF5_LIB) -L$(BOOST_LIB)



default: BSpline.o LU.o StateTable.o 

BSPLINE_OBJS = 		\
	BSpline.o	\
	LU.o

STATETBL_OBJS =		\
	StateTable.o


BSpline.o: $(BASE_SRC)/BSpline.cc $(BASE_INC)/BSpline.h
	$(LINK) $(INC) -c $(BASE_SRC)/BSpline.cc

LU.o: $(BASE_SRC)/LU.cc $(BASE_INC)/LU.h
	$(LINK) $(INC) -c $(BASE_SRC)/LU.cc

StateTable.o: $(BASE_SRC)/StateTable.cc $(BASE_INC)/StateTable.h
	$(LINK) $(INC) -c $(BASE_SRC)/StateTable.cc



clean:  ; @rm *.o *.a *.x
