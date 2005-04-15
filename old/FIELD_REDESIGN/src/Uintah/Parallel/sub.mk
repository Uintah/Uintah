#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Parallel

SRCS     += $(SRCDIR)/Parallel.cc $(SRCDIR)/ProcessorGroup.cc \
	$(SRCDIR)/UintahParallelComponent.cc $(SRCDIR)/UintahParallelPort.cc

PSELIBS := Uintah/Grid SCICore/Thread SCICore/Exceptions
LIBS := -lmpi

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.4  2000/06/17 07:06:50  sparker
# Changed ProcessorContext to ProcessorGroup
#
# Revision 1.3  2000/04/19 21:20:05  dav
# more MPI stuff
#
# Revision 1.2  2000/03/20 19:38:48  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:30:21  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
