# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/Modules/Surface

SRCS     += \
	$(SRCDIR)/GenSurface.cc\
	$(SRCDIR)/LabelSurface.cc\
	$(SRCDIR)/LookupSurface.cc\
	$(SRCDIR)/LookupSplitSurface.cc\
	$(SRCDIR)/SurfGen.cc\
	$(SRCDIR)/SurfInterpVals.cc\
	$(SRCDIR)/SurfNewVals.cc\
	$(SRCDIR)/SurfToGeom.cc\
	$(SRCDIR)/TransformSurface.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Dataflow/Network Core/Datatypes Dataflow/Widgets \
	Core/Persistent Core/Exceptions Core/Thread \
	Core/Containers Core/TclInterface Core/Geometry \
	Core/Datatypes Core/Geom
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk
