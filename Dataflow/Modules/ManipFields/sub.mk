# Makefile fragment for this subdirectory

SRCDIR := Dataflow/Modules/ManipFields

include $(SRCTOP)/scripts/largeso_epilogue.mk

SRCS := \
	$(SRCDIR)/Rescale.cc\
#[INSERT NEW CODE FILE HERE]

VOLATILE_LIB := ManipFields

PSELIBS := 

LIBS 	:= 

include $(SRCTOP)/scripts/volatile_lib.mk

