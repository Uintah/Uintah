#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Uintah/Components/ICE

SRCS     += $(SRCDIR)/ICE.cc

PSELIBS := Uintah/Interface Uintah/Grid SCICore/Exceptions
LIBS := $(XML_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.4  2000/05/30 19:36:40  dav
# added SCICore/Exceptions to PSELIBS
#
# Revision 1.3  2000/04/12 22:58:43  sparker
# Added xerces to link line
#
# Revision 1.2  2000/03/20 19:38:21  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:29:30  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
