#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/TclInterface

SRCS     += $(SRCDIR)/DebugSettings.cc $(SRCDIR)/GuiManager.cc \
	$(SRCDIR)/GuiServer.cc $(SRCDIR)/Histogram.cc \
	$(SRCDIR)/MemStats.cc $(SRCDIR)/Remote.cc $(SRCDIR)/TCL.cc \
	$(SRCDIR)/TCLInit.cc $(SRCDIR)/TCLTask.cc $(SRCDIR)/TCLvar.cc \
	$(SRCDIR)/ThreadStats.cc

PSELIBS := SCICore/Exceptions SCICore/Util SCICore/Thread \
	SCICore/Containers SCICore/TkExtensions
LIBS := $(TCL_LIBRARY) $(X11_LIBS)

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:37:50  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:38  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
