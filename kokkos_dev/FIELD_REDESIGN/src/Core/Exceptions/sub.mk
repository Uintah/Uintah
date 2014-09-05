#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/Exceptions

SRCS     += $(SRCDIR)/ArrayIndexOutOfBounds.cc \
	    $(SRCDIR)/AssertionFailed.cc \
	    $(SRCDIR)/DimensionMismatch.cc \
	    $(SRCDIR)/ErrnoException.cc \
	    $(SRCDIR)/Exception.cc \
	    $(SRCDIR)/FileNotFound.cc \
	    $(SRCDIR)/InternalError.cc

PSELIBS := 
LIBS := $(TRACEBACK_LIB)

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.5.2.1  2000/08/08 01:06:57  michaelc
# Accel SAttrib classes
#
# Revision 1.5  2000/05/20 08:06:14  sparker
# New exception classes
#
# Revision 1.4  2000/05/15 19:25:57  sparker
# Exception class for system calls (ones that use the errno variable)
#
# Revision 1.3  2000/03/23 10:25:41  sparker
# New exception facility - retired old "Exception.h" classes
#
# Revision 1.2  2000/03/20 19:37:36  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:21  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
