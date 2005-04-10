#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/largeso_prologue.mk

SRCDIR := SCICore

SUBDIRS := $(SRCDIR)/Containers $(SRCDIR)/Datatypes $(SRCDIR)/Exceptions \
	   $(SRCDIR)/Geom $(SRCDIR)/Geometry $(SRCDIR)/Malloc \
	   $(SRCDIR)/Math $(SRCDIR)/OS $(SRCDIR)/Process \
	   $(SRCDIR)/TclInterface $(SRCDIR)/Thread \
	   $(SRCDIR)/TkExtensions $(SRCDIR)/Tester \
	   $(SRCDIR)/Persistent $(SRCDIR)/Util $(SRCDIR)/GUI
ifeq ($(BUILD_PARALLEL),yes)
SUBDIRS := $(SUBDIRS) $(SRCDIR)/globus_threads
endif

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := 
LIBS := $(BLT_LIBRARY) $(ITCL_LIBRARY) $(TCL_LIBRARY) $(TK_LIBRARY) \
	$(GL_LIBS) $(GLOBUS_COMMON) $(THREAD_LIBS) -lm

include $(SRCTOP)/scripts/largeso_epilogue.mk

#
# $Log$
# Revision 1.3  2000/05/15 19:28:11  sparker
# New directory: OS for operating system interface classes
# Added a "Dir" class to create and iterate over directories (eventually)
#
# Revision 1.2  2000/03/20 19:37:31  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:15  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
