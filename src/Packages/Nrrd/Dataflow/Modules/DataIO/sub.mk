# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the Core/CCA/Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Core/CCA/Component"
# documentation on how to do it correctly.

include $(SRCTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := Packages/Nrrd/Dataflow/Modules/DataIO


SRCS     += \
	$(SRCDIR)/NrrdReader.cc\
	$(SRCDIR)/NrrdWriter.cc\
	$(SRCDIR)/NrrdToField.cc\
	$(SRCDIR)/FieldToNrrd.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Packages/Nrrd/Core/Datatypes Core/Datatypes \
	Dataflow/Network Dataflow/Ports \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/TkExtensions Packages/Nrrd/Core/Datatypes \
	Packages/Nrrd/Dataflow/Ports

LIBS := $(NRRD_LIBRARY) -lnrrd -lbiff -lair

include $(SRCTOP_ABS)/scripts/smallso_epilogue.mk
