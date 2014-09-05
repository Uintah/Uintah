#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECore/XMLUtil

SRCS     += $(SRCDIR)/SimpleErrorHandler.cc $(SRCDIR)/XMLUtil.cc

PSELIBS := 
LIBS := $(XML_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2000/05/20 08:04:29  sparker
# Added XML helper library
#
#
