# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Volume/Core/Util

SRCS     += \
	$(SRCDIR)/Utils.cc \
	$(SRCDIR)/ShaderProgramARB.cc \
	$(SRCDIR)/Pbuffer.cc \
	$(SRCDIR)/VideoCardInfo.c \
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Geometry \
	   Core/Util \
	   Core/Exceptions \

LIBS := $(GL_LIBRARY) $(M_LIBRARY)

ifeq ($(OS_NAME),Darwin)
LIBS := $(LIBS) -framework AGL
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
