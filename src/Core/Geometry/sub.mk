# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/Geometry

SRCS     += $(SRCDIR)/BBox.cc $(SRCDIR)/Grid.cc $(SRCDIR)/IntVector.cc \
	$(SRCDIR)/Point.cc $(SRCDIR)/Tensor.cc \
	$(SRCDIR)/Transform.cc $(SRCDIR)/Plane.cc $(SRCDIR)/Vector.cc \
	$(SRCDIR)/Ray.cc  $(SRCDIR)/Quaternion.cc

PSELIBS := Core/Persistent Core/Containers Core/Exceptions Core/Tester
LIBS := $(DEFAULT_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk
