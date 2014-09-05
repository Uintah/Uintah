# Makefile fragment for this subdirectory

SRCDIR   := Packages/Uintah/CCA/Components/ICE/Combined

SRCS     += \
	$(SRCDIR)/CombinedFactory.cc

PSELIBS := \
	Packages/Uintah/CCA/Ports \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/Core/Parallel \
	Packages/Uintah/Core/Exceptions \
	Packages/Uintah/Core/Math \
	Core/Exceptions Core/Thread Core/Geometry 

LIBS	:= 


