#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := DaveW/Modules/Readers

SRCS     += $(SRCDIR)/ContourSetReader.cc $(SRCDIR)/PathReader.cc \
	$(SRCDIR)/SegFldReader.cc $(SRCDIR)/SigmaSetReader.cc \
	$(SRCDIR)/TensorFieldReader.cc

PSELIBS := DaveW/Datatypes/General PSECore/Dataflow PSECore/Datatypes \
	SCICore/Exceptions SCICore/Thread SCICore/Containers \
	SCICore/TclInterface SCICore/Persistent
LIBS := 

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:25:58  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
