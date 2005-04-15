#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/Persistent

SRCS     += $(SRCDIR)/Persistent.cc $(SRCDIR)/Pstreams.cc

PSELIBS := SCICore/Containers
LIBS := -lz

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3  2000/03/23 10:29:23  sparker
# Use new exceptions/ASSERT macros
# Fixed compiler warnings
#
# Revision 1.2  2000/03/20 19:37:46  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:35  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
