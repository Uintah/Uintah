# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/ICE/Advection

SRCS     += $(SRCDIR)/Advector.cc \
 	$(SRCDIR)/FirstOrderAdvector.cc \
 	$(SRCDIR)/FirstOrderCEAdvector.cc \
	$(SRCDIR)/SecondOrderAdvector.cc \
 	$(SRCDIR)/SecondOrderCEAdvector.cc \
 	$(SRCDIR)/SecondOrderBase.cc \
	$(SRCDIR)/AdvectionFactory.cc

PSELIBS := \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Parallel \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Core/Math \
	Core/Exceptions Core/Thread Core/Geometry 

LIBS	:= 


