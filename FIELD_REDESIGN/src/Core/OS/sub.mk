#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/OS

SRCS     += $(SRCDIR)/Dir.cc

PSELIBS := SCICore/Exceptions
LIBS :=

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/05/15 19:28:13  sparker
# New directory: OS for operating system interface classes
# Added a "Dir" class to create and iterate over directories (eventually)
#
#
