#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := Remote/remoteViewer

FLEX := /usr/local/gnu/lib
GLUT := /usr/local/contrib/moulding/glut

ifeq ($(LARGESOS),yes)
PSELIBS := SCICore Remote
else
PSELIBS := SCICore/OS SCICore/Thread Remote/Tools Remote/Modules/remoteSalmon
endif
LIBS := -L$(GLUT)/lib -lglut -lGLU -lGL \
	-lXmu -lX11 -lXext -lm ### -L$(FLEX) -lfl -ltiff -ljpeg 

PROGRAM := $(SRCDIR)/VRMLView
SRCS := $(SRCDIR)/VRMLView.cc
include $(SRCTOP)/scripts/program.mk

#
# $Log$
# Revision 1.1  2000/07/10 20:44:26  dahart
# initial commit
#
# Revision 1.2  2000/03/20 19:36:34  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:24  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
