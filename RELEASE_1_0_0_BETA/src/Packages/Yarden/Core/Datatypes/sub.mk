# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Packages/Yarden/Core/Datatypes

SRCS     += \
	$(SRCDIR)/Clock.cc \
	$(SRCDIR)/SpanSpace.cc \
	$(SRCDIR)/SpanTree.cc \
#	$(SRCDIR)/TensorField.cc \

PSELIBS := Core/Datatypes
LIBS :=

include $(SRCTOP)/scripts/smallso_epilogue.mk


