#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Components/ProblemSpecification

SRCS	+= $(SRCDIR)/ProblemSpecReader.cc

PSELIBS := Uintah/Interface
LIBS 	:= $(XML_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

PROGRAM	:= $(SRCDIR)/testing
SRCS	:= $(SRCDIR)/testing.cc
include $(SRCTOP)/scripts/program.mk

PROGRAM	:= $(SRCDIR)/test2
SRCS	:= $(SRCDIR)/test2.cc 
PSELIBS := Uintah/Interface Uintah/Components/ProblemSpecification
LIBS 	:= $(XML_LIBRARY)
include $(SRCTOP)/scripts/program.mk


#
# $Log$
# Revision 1.3  2000/03/29 01:57:05  jas
# Added a problem specification reader.
#
# Revision 1.2  2000/03/20 19:38:24  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:29:43  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
