#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/Exceptions

SRCS     += $(SRCDIR)/Exceptions.cc $(SRCDIR)/InternalError.cc


PSELIBS := 
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:37:36  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:21  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
