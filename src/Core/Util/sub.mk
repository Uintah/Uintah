#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/Util

SRCS     += $(SRCDIR)/Debug.cc $(SRCDIR)/Timer.cc \
	$(SRCDIR)/soloader.cc $(SRCDIR)/DebugStream.cc

PSELIBS := SCICore/Containers SCICore/Exceptions
LIBS := $(DL_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3  2000/09/25 18:04:29  sparker
# Added DL_LIBRARY to link line
#
# Revision 1.2  2000/03/20 19:37:57  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:46  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
