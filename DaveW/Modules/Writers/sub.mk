#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := DaveW/Modules/Writers

SRCS     += $(SRCDIR)/ContourSetWriter.cc $(SRCDIR)/PathWriter.cc \
	$(SRCDIR)/SegFldWriter.cc $(SRCDIR)/SigmaSetWriter.cc \
	$(SRCDIR)/TensorFieldWriter.cc

PSELIBS := DaveW/Datatypes/General PSECore/Datatypes PSECore/Dataflow \
	SCICore/Persistent SCICore/Exceptions SCICore/Containers \
	SCICore/TclInterface SCICore/Thread 
LIBS := 

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:26:07  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
