#
# Makefile fragment for this subdirectory
# $Id$

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Uintah/Datatypes

SUBDIRS := $(SRCDIR)/Particles

include $(SRCTOP)/scripts/recurse.mk

SRCS     += $(SRCDIR)/Archive.cc  $(SRCDIR)/ArchivePort.cc 

PSELIBS := SCICore/Exceptions SCICore/Geometry \
	SCICore/Persistent SCICore/Datatypes \
	SCICore/Containers SCICore/Thread Uintah/Grid Uintah/Interface \
        Uintah/Exceptions PSECore/XMLUtil 

LIBS := $(XML_LIBRARY)

ifeq ($(BUILD_PARALLEL),yes)
PSELIBS := $(PSELIBS) Component/CIA Component/PIDL
LIBS := $(LIBS) $(GLOBUS_LIBS) -lglobus_nexus
endif

include $(SRCTOP)/scripts/smallso_epilogue.mk


#
# $Log$
# Revision 1.3  2000/06/20 20:13:27  kuzimmer
# updated so that Archiver reader will compile in Uintah
#
# Revision 1.2  2000/03/20 19:38:29  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:29:49  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
#
# Makefile fragment for this subdirectory
# $Id$
#
