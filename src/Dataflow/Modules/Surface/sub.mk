#
# Makefile fragment for this subdirectory
#

# *** NOTE ***
# 
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECommon/Modules/Surface

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

PSELIBS := PSECore/Dataflow PSECore/Datatypes PSECore/Widgets \
	SCICore/Persistent SCICore/Exceptions SCICore/Thread \
	SCICore/Containers SCICore/TclInterface SCICore/Geometry \
	SCICore/Datatypes SCICore/Geom
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk
