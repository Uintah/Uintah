# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Yarden/Dataflow/Ports

SRCS     += \
	$(SRCDIR)/SpanPort.cc\
#	$(SRCDIR)/TensorFieldPort.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Dataflow/Network Dataflow/Ports \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread \
        Core/Geom Core/Datatypes Core/Geometry 
LIBS :=  $(M_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk
