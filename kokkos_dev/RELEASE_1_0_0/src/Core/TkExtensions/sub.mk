#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/TkExtensions

SRCS     += $(SRCDIR)/tk3d2.c $(SRCDIR)/tkAppInit.c $(SRCDIR)/tkBevel.c \
	$(SRCDIR)/tkCanvBLine.c $(SRCDIR)/tkCursor.c $(SRCDIR)/tkOpenGL.c \
	$(SRCDIR)/tk3daux.c

SRCS += $(SRCDIR)/tclUnixNotify-$(TK_VERSION).c

#	$(SRCDIR)/tclTimer.c \
#	$(SRCDIR)/tkRange.c $(SRCDIR)/tkUnixRange.c \


INCLUDES := -I$(TCL_SRC_DIR) -I$(TCL_SRC_DIR)/generic \
	 -I$(TK_SRC_DIR) -I$(TK_SRC_DIR)/generic -I$(TK_SRC_DIR)/unix \
	 -I$(ITCL_SRC_DIR) -I$(ITCL_SRC_DIR)/generic \
	 -I$(ITK_SRC_DIR) -I$(ITK_SRC_DIR)/generic $(INCLUDES)

PSELIBS := 
LIBS := $(BLT_LIBRARY) \
	$(ITK_LIBRARY) \
	$(ITCL_LIBRARY) \
	$(TK_LIBRARY) \
	$(TCL_LIBRARY) \
	$(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

