#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Nektar/Datatypes

SRCS += $(SRCDIR)/NektarScalarField.cc \
	$(SRCDIR)/NektarVectorField.cc  

PSELIBS := SCICore/Persistent SCICore/Exceptions SCICore/Containers \
	SCICore/Thread SCICore/Geometry SCICore/Geom SCICore/TclInterface \
	SCICore/Math
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

clean::

