# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/DaveW/Core/Datatypes/CS684

SRCS     += $(SRCDIR)/DRaytracer.cc $(SRCDIR)/ImageR.cc \
	$(SRCDIR)/Pixel.cc $(SRCDIR)/RTPrims.cc $(SRCDIR)/RadPrims.cc \
	$(SRCDIR)/Sample2D.cc $(SRCDIR)/Scene.cc $(SRCDIR)/Spectrum.cc

PSELIBS := Core/Persistent Core/Geometry Core/Math \
	Core/Thread Core/Exceptions Core/Geom \
	Core/Containers Core/Datatypes
LIBS := $(M_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

