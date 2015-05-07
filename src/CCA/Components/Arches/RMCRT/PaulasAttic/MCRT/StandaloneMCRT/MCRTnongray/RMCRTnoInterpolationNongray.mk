CXX = g++
CFLAGS = -g
# CFLAGS = -pg -O3 -MD

SRCS = RMCRTnoInterpolationNongray.cc Surface.cc RealSurface.cc TopRealSurface.cc BottomRealSurface.cc \
	   FrontRealSurface.cc BackRealSurface.cc LeftRealSurface.cc RightRealSurface.cc \
	   VirtualSurface.cc ray.cc VolElement.cc MakeTableFunction.cc RadWsgg.cc RadCoeff.cc\


OBJS := $(patsubst %.cc,%.o,$(filter %.cc,$(SRCS)))

RMCRTnoInterpolationNongray : $(OBJS) 
			$(CXX) $(CFLAGS) $(OBJS) -o RMCRTnoInterpolationNongray

.cc.o: $<
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.d  RMCRTnoInterpolationNongray *.out

-include $(SRCS:.cc=.d)
