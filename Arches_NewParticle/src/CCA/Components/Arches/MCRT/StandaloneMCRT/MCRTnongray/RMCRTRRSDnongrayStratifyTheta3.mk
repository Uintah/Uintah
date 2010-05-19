CXX = g++
# CFLAGS = -g -MD
CFLAGS = -pg -O3 -MD

SRCS =  RMCRTRRSDnongrayStratifyTheta3.cc Surface.cc RealSurface.cc TopRealSurface.cc BottomRealSurface.cc \
	   FrontRealSurface.cc BackRealSurface.cc LeftRealSurface.cc RightRealSurface.cc \
	   VirtualSurface.cc ray.cc VolElement.cc MakeTableFunction.cc RadWsgg.cc RadCoeff.cc\
	

OBJS := $(patsubst %.cc,%.o,$(filter %.cc,$(SRCS)))

RMCRTRRSDnongrayStratifyTheta3 : $(OBJS) 
			$(CXX) $(CFLAGS) $(OBJS) -o  RMCRTRRSDnongrayStratifyTheta3

.cc.o: $<
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.d RMCRTRRSDnongrayStratifyTheta3 *.out

-include $(SRCS:.cc=.d)
