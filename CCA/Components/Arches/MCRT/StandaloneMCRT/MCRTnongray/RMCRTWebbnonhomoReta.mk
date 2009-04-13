CXX = g++
CFLAGS =  -g
# CFLAGS = -pg -O3 -MD

SRCS = RMCRTnoInterpolationWebbnonhomoReta.cc Surface.cc RealSurface.cc TopRealSurface.cc BottomRealSurface.cc \
	   FrontRealSurface.cc BackRealSurface.cc LeftRealSurface.cc RightRealSurface.cc \
	   VirtualSurface.cc ray.cc VolElement.cc MakeTableFunction.cc RadWsgg.cc RadCoeff.cc BinarySearchTree.cc\


OBJS := $(patsubst %.cc,%.o,$(filter %.cc,$(SRCS)))

RMCRTnoInterpolationWebbnonhomoRetaTest : $(OBJS) 
			$(CXX) $(CFLAGS) $(OBJS) -o RMCRTnoInterpolationWebbnonhomoRetaTest

.cc.o: $<
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.d  RMCRTnoInterpolationWebbnonhomoRetaTest *.out

-include $(SRCS:.cc=.d)
