# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR := Dataflow/Modules/ManipFields

SRCS := \
	$(SRCDIR)/Rescale.cc\
#[INSERT NEW CODE FILE HERE]

VOLATILE_LIB := ManipFields

PSELIBS := Dataflow/Network Core/GuiInterface Core/Containers

LIBS 	:= 

include $(SRCTOP)/scripts/smallso_epilogue.mk


