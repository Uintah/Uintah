#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Components/ProblemSpecification

SRCS	+= $(SRCDIR)/ProblemSpecReader.cc

PSELIBS := Uintah/Interface Uintah/Exceptions Uintah/Grid
LIBS 	:= $(XML_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

#PROGRAM	:= $(SRCDIR)/testing
#SRCS	:= $(SRCDIR)/testing.cc
#include $(SRCTOP)/scripts/program.mk
#
PROGRAM	:= $(SRCDIR)/test2
SRCS	:= $(SRCDIR)/test2.cc 
PSELIBS := Uintah/Interface Uintah/Components/ProblemSpecification \
	Uintah/Grid
LIBS 	:= $(XML_LIBRARY)
include $(SRCTOP)/scripts/program.mk


#
# $Log$
# Revision 1.8  2000/04/28 07:35:33  sparker
# Started implementation of DataWarehouse
# MPM particle initialization now works
#
# Revision 1.7  2000/04/12 22:59:42  sparker
# Added crude error handling capabilities to reader
#
# Revision 1.6  2000/04/06 18:10:31  jas
# Added Uintah/Grid to the PSELIB for test2 to compile.
#
# Revision 1.5  2000/03/30 20:23:39  sparker
# Fixed compile on SGI
#
# Revision 1.4  2000/03/29 23:45:22  jas
# Sample input file format using xml.
#
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
