#
# Makefile fragment for this subdirectory
# $Id$
#

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Parallel

SRCS     += $(SRCDIR)/Parallel.cc $(SRCDIR)/ProcessorContext.cc \
	$(SRCDIR)/UintahParallelComponent.cc $(SRCDIR)/UintahParallelPort.cc

PSELIBS := Uintah/Grid SCICore/Thread SCICore/Exceptions
LIBS :=

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/03/17 09:30:21  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#