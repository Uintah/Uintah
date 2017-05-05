CXX = g++
CFLAGS =  -g
# CFLAGS = -pg -O3 -MD

SRCS = RMCRTnoInterpolationFSKRIetaSerial.cc Surface.cc RealSurface.cc TopRealSurface.cc BottomRealSurface.cc \
	   FrontRealSurface.cc BackRealSurface.cc LeftRealSurface.cc RightRealSurface.cc \
	   VirtualSurface.cc ray.cc VolElement.cc MakeTableFunction.cc RadWsgg.cc RadCoeff.cc BinarySearchTree.cc\


OBJS := $(patsubst %.cc,%.o,$(filter %.cc,$(SRCS)))

RMCRTnoInterpolationFSKRIetaSerial : $(OBJS) 
			$(CXX) $(CFLAGS) $(OBJS) -o RMCRTnoInterpolationFSKRIetaSerial

.cc.o: $<
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.d  RMCRTnoInterpolationFSKRIetaSerial *.out

-include $(SRCS:.cc=.d)
