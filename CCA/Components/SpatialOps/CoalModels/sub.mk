# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/SpatialOps/CoalModels

SRCS     += $(SRCDIR)/ModelFactory.cc \
	$(SRCDIR)/ModelBase.cc \
  $(SRCDIR)/BadHawkDevol.cc

PSELIBS := \
	CCA/Ports \
	Core/Grid \
	Core/Parallel \
	Core/Exceptions \
	Core/Math \
	Core/Exceptions Core/Thread Core/Geometry 

LIBS	:= 


