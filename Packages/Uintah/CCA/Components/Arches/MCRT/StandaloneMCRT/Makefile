
CXX = g++
# CFLAGS = -g
CFLAGS = -pg -O3

CXX_OBJS = RMCRT.o RNG.o Surface.o RealSurface.o TopRealSurface.o BottomRealSurface.o \
	   FrontRealSurface.o BackRealSurface.o LeftRealSurface.o RightRealSurface.o \
	   VirtualSurface.o ray.o VolElement.o MakeTableFunction.o \
	   setupBenchmark.o

NoTableMCRT : $(CXX_OBJS) 
	$(CXX) $(CFLAGS) $(CXX_OBJS) -o NoTableMCRT

.cc.o:
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o NoTableMCRT gmon.out
