# Makefile fragment for this subdirectory

SRCDIR := Packages/Remote/remoteViewer

FLEX := /usr/local/gnu/lib
GLUT := /usr/local/contrib/moulding/glut

ifeq ($(LARGESOS),yes)
PSELIBS := Core Remote
else
PSELIBS := Core/OS Core/Thread Remote/Tools Remote/Modules/remoteSalmon
endif
LIBS := -L$(GLUT)/lib -lglut  $(XML_LIBRARY) $(TK_LIBRARY) $(GL_LIBS) \
	 -lm

# -lXmu -lX11 -lXext -lm ### -L$(FLEX) -lfl -ltiff -ljpeg 

PROGRAM := $(SRCDIR)/VRMLView
SRCS := $(SRCDIR)/VRMLView.cc
include $(SRCTOP)/scripts/program.mk

