#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/TkExtensions

SRCS     += $(SRCDIR)/tk3d2.c $(SRCDIR)/tkAppInit.c $(SRCDIR)/tkBevel.c \
	$(SRCDIR)/tkCanvBLine.c $(SRCDIR)/tkCursor.c $(SRCDIR)/tkOpenGL.c \
	$(SRCDIR)/tk3daux.c

SRCS += $(SRCDIR)/tclUnixNotify-$(TK_VERSION).c

#	$(SRCDIR)/tclTimer.c \
#	$(SRCDIR)/tkRange.c $(SRCDIR)/tkUnixRange.c \


INCLUDES += -I$(TCL_SRC_DIR) -I$(TCL_SRC_DIR)/generic \
	 -I$(TK_SRC_DIR) -I$(TK_SRC_DIR)/generic -I$(TK_SRC_DIR)/unix \
	 -I$(ITCL_SRC_DIR) -I$(ITCL_SRC_DIR)/generic \
	 -I$(ITK_SRC_DIR) -I$(ITK_SRC_DIR)/generic

PSELIBS := 
LIBS := $(BLT_LIBRARY) \
	$(ITK_LIBRARY) \
	$(ITCL_LIBRARY) \
	$(TK_LIBRARY) \
	$(TCL_LIBRARY) \
	$(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3  2000/11/12 20:51:10  yarden
# add support for itcltk-8.3
#
# tk3d.c which was a modified version of the tk package
# is replaced by tk3daux which contain only the extra functionality
#
# tclTimer.c which was also a tk package modification was removed.
#
# tclUnixNotify.c is replaced by two version one for 8.0 and one for 8.3
# as this file is different in the two distributions.
#
# Revision 1.2  2000/03/20 19:37:55  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:44  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
