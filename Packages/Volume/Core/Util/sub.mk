# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Volume/Core/Util

SRCS     += \
	$(SRCDIR)/SliceTable.cc \
	$(SRCDIR)/Utils.cc
#	$(SRCDIR)/Pbuffer.cc \
#	$(SRCDIR)/Shader.cc \
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Geometry \
	   Core/Util \
	   Core/Exceptions \

LIBS := $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
