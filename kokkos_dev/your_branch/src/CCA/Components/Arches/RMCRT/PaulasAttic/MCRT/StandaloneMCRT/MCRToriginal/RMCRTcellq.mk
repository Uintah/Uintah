CXX = g++
#CFLAGS = -g
CFLAGS = -pg -O3 -MD

SRCS = RMCRTcellq.cc Surface.cc RealSurface.cc TopRealSurface.cc BottomRealSurface.cc \
	   FrontRealSurface.cc BackRealSurface.cc LeftRealSurface.cc RightRealSurface.cc \
	   VirtualSurface.cc ray.cc VolElement.cc MakeTableFunction.cc\
	

OBJS := $(patsubst %.cc,%.o,$(filter %.cc,$(SRCS)))

RMCRTRR : $(OBJS) 
	$(CXX) $(CFLAGS) $(OBJS) -o RMCRTcellq

.cc.o: $<
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.d vtk* RMCRTcellq *.out

-include $(SRCS:.cc=.d)
