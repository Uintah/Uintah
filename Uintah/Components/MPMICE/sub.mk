#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Components/MPMICE

SRCS     += $(SRCDIR)/MPMICE.cc

PSELIBS := Uintah/Interface Uintah/Grid Uintah/Parallel \
	Uintah/Exceptions SCICore/Exceptions SCICore/Thread \
	SCICore/Geometry PSECore/XMLUtil Uintah/Math \
	Uintah/Components/MPM
LIBS := $(XML_LIBRARY) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

