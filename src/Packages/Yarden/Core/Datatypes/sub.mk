#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Yarden/Datatypes

SRCS     += \
	$(SRCDIR)/TensorField.cc \
	$(SRCDIR)/TensorFieldPort.cc

PSELIBS :=
LIBS :=

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3  2000/10/23 23:39:39  yarden
# Tensor and Tensor Field definitions
#
# Revision 1.2  2000/03/20 19:38:52  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:30:27  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#

