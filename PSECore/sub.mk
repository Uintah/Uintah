#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/largeso_prologue.mk

SRCDIR := PSECore

SUBDIRS := $(SRCDIR)/Comm $(SRCDIR)/Constraints $(SRCDIR)/Dataflow \
	$(SRCDIR)/Datatypes $(SRCDIR)/Widgets $(SRCDIR)/GUI
ifeq ($(BUILD_PARALLEL),yes)
#SUBDIRS += $(SRCDIR)/Interface $(SRCDIR)/Controller $(SRCDIR)/Builder
endif

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := SCICore
LIBS := $(TCL_LIBRARY) $(XML_LIBRARY) -lm

include $(SRCTOP)/scripts/largeso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:37:12  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:49  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
