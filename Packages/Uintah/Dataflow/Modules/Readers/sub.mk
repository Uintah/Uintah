# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Dataflow/Modules/Readers

SRCS     += \
	$(SRCDIR)/ArchiveReader.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Uintah/Datatypes Uintah/Interface Dataflow/Ports \
	Dataflow/Network Core/Containers Core/Persistent \
	Core/Exceptions Core/TclInterface Core/Thread \
	Core/Datatypes Core/Geom
LIBS := $(XML_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

