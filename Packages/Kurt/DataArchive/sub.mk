#
# Makefile fragment for this subdirectory
# $Id$
#
include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Kurt/DataArchive

SRCS     += $(SRCDIR)/VisParticleSet.cc \
		$(SRCDIR)/VisParticleSetPort.cc 

PSELIBS := SCICore/Exceptions SCICore/Geometry \
	SCICore/Persistent SCICore/Datatypes \
	SCICore/Containers Uintah/Grid Uintah/Interface \
	Uintah/Exceptions Kurt/Datatypes PSECore/XMLUtil 

LIBS := $(XML_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.4  2000/06/21 04:14:12  kuzimmer
# removed unneeded dependencies on Kurt
#
# Revision 1.3  2000/05/22 17:19:56  kuzimmer
# Updating new Viz tools
#
# Revision 1.2  2000/05/21 08:13:49  sparker
# Removed depend.mk - it doesn't belong in the repository
# Added xml libraries to sub.mk
#
# Revision 1.1  2000/05/20 02:31:47  kuzimmer
# Multiple changes for new vis tools
#
# Revision 1.4  2000/05/16 20:54:03  kuzimmer
# added new directory
#
# Revision 1.3  2000/03/21 17:33:28  kuzimmer
# updating volume renderer
#
# Revision 1.2  2000/03/20 19:36:41  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:35  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
