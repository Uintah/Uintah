#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := Uintah/tools/graphview

PROGRAM := $(SRCDIR)/graphview

SRCS := $(SRCDIR)/graphview.cc	$(SRCDIR)/GV_TaskGraph.cc 	\
	$(SRCDIR)/DaVinci.cc

PSELIBS := PSECore/XMLUtil SCICore/Exceptions SCICore/Thread Uintah/Interface
LIBS := $(XML_LIBRARY)

include $(SRCTOP)/scripts/program.mk

#
# $Log$
# Revision 1.3  2000/11/22 02:29:35  witzel
# List timesteps using DataArchive queryTimesteps instead of listing
# the directories, and changed Task to GV_Task and TaskGraph to GV_TaskGraph
# to avoid conflicts.
#
# Revision 1.2  2000/08/03 23:34:42  witzel
# *** empty log message ***
#
# Revision 1.1  2000/07/28 05:00:10  jehall
# - Initial checkin of work-in-progress graph viz tool
#
#
