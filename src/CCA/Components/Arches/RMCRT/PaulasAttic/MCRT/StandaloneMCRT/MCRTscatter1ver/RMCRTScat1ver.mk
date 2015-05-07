CXX = g++
# CFLAGS = -g -MD
CFLAGS = -pg -O3 -MD

SRCS = RMCRTScat1ver.cc Surface.cc RealSurface.cc TopRealSurface.cc BottomRealSurface.cc \
	   FrontRealSurface.cc BackRealSurface.cc LeftRealSurface.cc RightRealSurface.cc \
	   VirtualSurface.cc ray.cc VolElement.cc MakeTableFunction.cc \
	

OBJS := $(patsubst %.cc,%.o,$(filter %.cc,$(SRCS)))

RMCRTScat1ver : $(OBJS) 
		$(CXX) $(CFLAGS) $(OBJS) -o RMCRTScat1ver

.cc.o: $<
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.d RMCRTScat1ver  *.out

-include $(SRCS:.cc=.d)
