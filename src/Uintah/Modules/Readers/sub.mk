#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Modules/Readers

SRCS     += $(SRCDIR)/MPReader.cc $(SRCDIR)/MPReaderMultiFile.cc \
	$(SRCDIR)/TriangleReader.cc $(SRCDIR)/ParticleSetReader.cc

PSELIBS := Uintah/Datatypes/Particles PSECore/Datatypes PSECore/Dataflow \
	SCICore/Containers SCICore/Persistent SCICore/Exceptions \
	SCICore/TclInterface SCICore/Thread SCICore/Datatypes \
	SCICore/Geom
LIBS :=

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:38:43  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:30:16  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
