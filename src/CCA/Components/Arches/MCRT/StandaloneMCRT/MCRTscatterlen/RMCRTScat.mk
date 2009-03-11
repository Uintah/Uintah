CXX = g++
CFLAGS = -g
# CFLAGS = -pg -O3 -MD

SRCS = RMCRTScat.cc Surface.cc RealSurface.cc TopRealSurface.cc BottomRealSurface.cc \
	   FrontRealSurface.cc BackRealSurface.cc LeftRealSurface.cc RightRealSurface.cc \
	   VirtualSurface.cc ray.cc VolElement.cc MakeTableFunction.cc\
	

OBJS := $(patsubst %.cc,%.o,$(filter %.cc,$(SRCS)))

RMCRTScat : $(OBJS) 
			$(CXX) $(CFLAGS) $(OBJS) -o RMCRTScat

.cc.o: $<
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.d vtk* RMCRTScat *.out

-include $(SRCS:.cc=.d)
