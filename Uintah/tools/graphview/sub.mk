#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := Uintah/tools/graphview

PROGRAM := $(SRCDIR)/graphview

SRCS := $(SRCDIR)/graphview.cc $(SRCDIR)/TaskGraph.cc $(SRCDIR)/DaVinci.cc
PSELIBS := PSECore/XMLUtil
LIBS := $(XML_LIBRARY)

include $(SRCTOP)/scripts/program.mk

#
# $Log$
# Revision 1.1  2000/07/28 05:00:10  jehall
# - Initial checkin of work-in-progress graph viz tool
#
#
