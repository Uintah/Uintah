
CXX = g++
# CFLAGS = -g -MD
CFLAGS = -pg -O3 -MD

SRCS = RMCRT.cc Surface.cc RealSurface.cc TopRealSurface.cc BottomRealSurface.cc \
	   FrontRealSurface.cc BackRealSurface.cc LeftRealSurface.cc RightRealSurface.cc \
	   VirtualSurface.cc ray.cc VolElement.cc MakeTableFunction.cc \
	   setupBenchmark.cc

OBJS := $(patsubst %.cc,%.o,$(filter %.cc,$(SRCS)))

NoTableMCRT : $(OBJS) 
	$(CXX) $(CFLAGS) $(OBJS) -o RMCRT

.cc.o: $<
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.d RMCRT vtk* *.out

-include $(SRCS:.cc=.d)

