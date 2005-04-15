#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Modules/Writers

SRCS     += \
	$(SRCDIR)/MPWriter.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Uintah/Datatypes PSECore/Dataflow \
	SCICore/TclInterface SCICore/Exceptions SCICore/Persistent \
	SCICore/Containers SCICore/Thread SCICore/Datatypes
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3.2.1  2000/10/26 10:06:30  moulding
# merge HEAD into FIELD_REDESIGN
#
# Revision 1.4  2000/10/24 05:57:59  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.3  2000/06/20 17:57:20  kuzimmer
# Moved GridVisualizer to Uintah
#
# Revision 1.2  2000/03/20 19:38:45  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:30:19  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
