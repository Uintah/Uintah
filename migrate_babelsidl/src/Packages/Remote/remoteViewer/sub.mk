# Makefile fragment for this subdirectory

SRCDIR := Packages/Remote/remoteViewer

ifeq ($(LARGESOS),yes)
PSELIBS := Core Remote
else
PSELIBS := Core/OS Core/Thread Remote/Tools Remote/Modules/remoteSalmon
endif
LIBS := $(XML_LIBRARY) $(TK_LIBRARY) $(GLUT_LIBRARY) $(GL_LIBS) $(M_LIBRARY)

# $(XMU_LIBRARY) $(X_LIBRARY) $(M_LIBRARY) ### $(LEX_LIBRARY) $(TIFF_LIBRARY) $(JPEG_LIBRARY)

PROGRAM := $(SRCDIR)/VRMLView
SRCS := $(SRCDIR)/VRMLView.cc
include $(SRCTOP)/scripts/program.mk

