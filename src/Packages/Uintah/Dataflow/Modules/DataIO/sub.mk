# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Uintah/Dataflow/Modules/DataIO

SRCS     += \
	$(SRCDIR)/ArchiveReader.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := \
	Packages/Uintah/Core/Datatypes \
	Packages/Uintah/Core/DataArchive \
	Packages/Uintah/Core/Math \
	Packages/Uintah/Core/Grid \
	Packages/Uintah/CCA/Ports \
	Dataflow/Network \
	Dataflow/Ports \
	Core/Containers \
	Core/Persistent \
	Core/Exceptions \
	Core/GuiInterface \
	Core/Thread \
	Core/Datatypes \
	Core/Geom \
	Core/GeomInterface
LIBS := $(XML_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


ifeq ($(LARGESOS),no)
UINTAH_SCIRUN := $(UINTAH_SCIRUN) $(LIBNAME)
endif
