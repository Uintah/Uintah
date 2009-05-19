# Makefile fragment for this subdirectory

SRCDIR   := CCA/Components/SpatialOps/SourceTerms

SRCS     += $(SRCDIR)/SourceTermFactory.cc \
	$(SRCDIR)/SourceTermBase.cc \
	$(SRCDIR)/ConstSrcTerm.cc 

PSELIBS := \
	CCA/Ports \
	Core/Grid \
	Core/Parallel \
	Core/Exceptions \
	Core/Math \
	Core/Exceptions Core/Thread Core/Geometry 

LIBS	:= 


