#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/TkExtensions

SRCS     += $(SRCDIR)/tk3d2.c $(SRCDIR)/tkAppInit.c $(SRCDIR)/tkBevel.c \
	$(SRCDIR)/tkCanvBLine.c $(SRCDIR)/tkCursor.c $(SRCDIR)/tkOpenGL.c \
	$(SRCDIR)/tkRange.c $(SRCDIR)/tkUnixRange.c $(SRCDIR)/tclTimer.c \
	$(SRCDIR)/tclUnixNotfy.c $(SRCDIR)/tk3d.c

PSELIBS := 
LIBS := $(BLT_LIBRARY) $(ITCL_LIBRARY) $(TK_LIBRARY) $(TCL_LIBRARY) \
	$(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:37:55  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:44  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
