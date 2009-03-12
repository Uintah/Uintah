# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/SpatialOps/TransportEqns

SRCS     += $(SRCDIR)/EqnFactory.cc \
	$(SRCDIR)/EqnBase.cc \
	$(SRCDIR)/ScalarEqn.cc 

PSELIBS := \
	CCA/Ports \
	Core/Grid \
	Core/Parallel \
	Core/Exceptions \
	Core/Math \
	Core/Exceptions Core/Thread Core/Geometry 

LIBS	:= 


