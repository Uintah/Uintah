# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Yarden/Core/Algorithms/Visualization

SRCS     += \
	$(SRCDIR)/NoiseMCube.cc \
	$(SRCDIR)/Phase.cc \
	$(SRCDIR)/Screen.cc 

PSELIBS := Packages/Yarden/Core/Datatypes \
	   Core/Persistent Core/Containers Core/Util \
	   Core/Exceptions Core/Thread Core/GuiInterface \
	   Core/Geom Core/Datatypes Core/Geometry \
	   Core/TkExtensions

LIBS := $(TK_LIBRARY) $(GL_LIBS) $(M_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

