# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/Modules/Writers

SRCS     += \
	$(SRCDIR)/TiffWriter.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := SCIRun/Datatypes/Image Dataflow/Network PSECore/Datatypes \
	Core/Datatypes Core/Persistent Core/Exceptions \
	Core/TclInterface Core/Containers Core/Thread \
	Core/Math Core/TkExtensions
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

