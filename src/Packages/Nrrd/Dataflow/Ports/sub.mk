# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Nrrd/Dataflow/Ports

SRCS     += $(SRCDIR)/NrrdPort.cc  \
#[INSERT NEW CODE FILE HERE]



PSELIBS := Dataflow/Network Dataflow/Ports Core/Containers \
	Core/Thread Core/Geom Core/Geometry Core/Exceptions \
	Core/Persistent Core/Datatypes Core/Util \
	Packages/Nrrd/Core/Datatypes

include $(SRCTOP)/scripts/smallso_epilogue.mk

