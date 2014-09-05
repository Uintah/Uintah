CXX = /usr/bin/mpicxx
# CFLAGS = -g -MD
CFLAGS = -pg -O3 -MD

SRCS =  RMCRTRRSDnongrayStratifyTheta3MPI.cc Surface.cc RealSurface.cc TopRealSurface.cc BottomRealSurface.cc \
	   FrontRealSurface.cc BackRealSurface.cc LeftRealSurface.cc RightRealSurface.cc \
	   VirtualSurface.cc ray.cc VolElement.cc MakeTableFunction.cc RadWsgg.cc RadCoeff.cc\
	

OBJS := $(patsubst %.cc,%.o,$(filter %.cc,$(SRCS)))

RMCRT11133BressloffWSGG404040ray15001e-10 : $(OBJS) 
			$(CXX) $(CFLAGS) $(OBJS) -o  RMCRT11133BressloffWSGG404040ray15001e-10

.cc.o: $<
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.d  *.out

-include $(SRCS:.cc=.d)
