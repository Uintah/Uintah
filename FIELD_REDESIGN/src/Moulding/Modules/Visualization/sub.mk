# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

include $(OBJTOP_ABS)/scripts/smallso_prologue.mk

SRCDIR   := Moulding/Modules/Visualization

SRCS     += \
	$(SRCDIR)/VolumeRender.cc\
	$(SRCDIR)/StreamLines.cc\
	$(SRCDIR)/GenField.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := PSECore/Datatypes PSECore/Dataflow \
        SCICore/Persistent SCICore/Containers SCICore/Util \
        SCICore/Exceptions SCICore/Thread SCICore/TclInterface \
        SCICore/Geom SCICore/Datatypes SCICore/Geometry \
        SCICore/TkExtensions
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(OBJTOP_ABS)/scripts/smallso_epilogue.mk


