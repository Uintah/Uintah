# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Packages/Yarden/Core/Datatypes

SRCS     += \
	$(SRCDIR)/TensorField.cc \
	$(SRCDIR)/TensorFieldPort.cc

PSELIBS :=
LIBS :=

include $(SRCTOP)/scripts/smallso_epilogue.mk


