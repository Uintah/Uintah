#
# Makefile fragment for this subdirectory
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/Geometry

SRCS     += $(SRCDIR)/BBox.cc $(SRCDIR)/Grid.cc $(SRCDIR)/IntVector.cc \
	$(SRCDIR)/Point.cc \
	$(SRCDIR)/Transform.cc $(SRCDIR)/Plane.cc $(SRCDIR)/Vector.cc \
	$(SRCDIR)/Ray.cc  $(SRCDIR)/Quaternion.cc

PSELIBS := SCICore/Containers SCICore/Exceptions SCICore/Tester
LIBS := $(DEFAULT_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk
