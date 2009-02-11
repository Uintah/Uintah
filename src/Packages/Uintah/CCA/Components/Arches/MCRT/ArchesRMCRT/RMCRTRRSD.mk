CXX = g++
# CFLAGS = -g -MD
CFLAGS = -pg -O3 -MD

SRCS = RMCRTRRSD.cc Surface.cc RealSurface.cc TopRealSurface.cc BottomRealSurface.cc \
	   FrontRealSurface.cc BackRealSurface.cc LeftRealSurface.cc RightRealSurface.cc \
	   VirtualSurface.cc ray.cc VolElement.cc MakeTableFunction.cc \
	

OBJS := $(patsubst %.cc,%.o,$(filter %.cc,$(SRCS)))

RMCRTRR : $(OBJS) 
	$(CXX) $(CFLAGS) $(OBJS) -o RMCRTRRSD

.cc.o: $<
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.d vtk* RMCRTRRSD *.out

-include $(SRCS:.cc=.d)
