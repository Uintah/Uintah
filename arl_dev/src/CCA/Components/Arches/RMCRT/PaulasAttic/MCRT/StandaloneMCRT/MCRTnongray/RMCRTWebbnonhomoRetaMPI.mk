CXX = /usr/bin/mpicxx
CFLAGS =  -g
# CFLAGS = -pg -O3 -MD

SRCS = RMCRTnoInterpolationWebbnonhomoRetaMPI.cc Surface.cc RealSurface.cc TopRealSurface.cc BottomRealSurface.cc \
	   FrontRealSurface.cc BackRealSurface.cc LeftRealSurface.cc RightRealSurface.cc \
	   VirtualSurface.cc ray.cc VolElement.cc MakeTableFunction.cc RadWsgg.cc RadCoeff.cc BinarySearchTree.cc\


OBJS := $(patsubst %.cc,%.o,$(filter %.cc,$(SRCS)))

RMCRTWebbnonhomoRetaMPI10wan606060withIbk30 : $(OBJS) 
			$(CXX) $(CFLAGS) $(OBJS) -o RMCRTWebbnonhomoRetaMPI10wan606060withIbk30

.cc.o: $<
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.d   *.out

-include $(SRCS:.cc=.d)
