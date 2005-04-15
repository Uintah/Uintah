#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Modules/Readers

SRCS     += \
	$(SRCDIR)/MPReader.cc\
	$(SRCDIR)/MPReaderMultiFile.cc\
	$(SRCDIR)/TriangleReader.cc\
	$(SRCDIR)/ParticleSetReader.cc \
	$(SRCDIR)/ArchiveReader.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Uintah/Datatypes Uintah/Interface PSECore/Datatypes \
	PSECore/Dataflow SCICore/Containers SCICore/Persistent \
	SCICore/Exceptions SCICore/TclInterface SCICore/Thread \
	SCICore/Datatypes SCICore/Geom
LIBS := $(XML_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3.2.1  2000/10/26 10:06:23  moulding
# merge HEAD into FIELD_REDESIGN
#
# Revision 1.4  2000/10/24 05:57:57  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.3  2000/06/20 17:57:18  kuzimmer
# Moved GridVisualizer to Uintah
#
# Revision 1.2  2000/03/20 19:38:43  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:30:16  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
