#
# Makefile fragment for this subdirectory
# $Id$
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

SRCDIR   := DaveW/Modules/EEG

SRCS     += \
	$(SRCDIR)/BldEEGMesh.cc\
	$(SRCDIR)/Coregister.cc\
	$(SRCDIR)/RescaleSegFld.cc\
	$(SRCDIR)/SFRGtoSFUG.cc\
	$(SRCDIR)/STreeExtractSurf.cc\
	$(SRCDIR)/SegFldOps.cc\
	$(SRCDIR)/SegFldToSurfTree.cc\
	$(SRCDIR)/SelectSurfNodes.cc\
	$(SRCDIR)/Taubin.cc\
	$(SRCDIR)/Thermal.cc\
	$(SRCDIR)/TopoSurfToGeom.cc\
	$(SRCDIR)/SliceMaker.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := DaveW/Datatypes/General PSECore/Datatypes PSECore/Widgets \
	PSECore/Dataflow SCICore/Persistent SCICore/Exceptions \
	SCICore/Geom SCICore/Thread SCICore/Geometry SCICore/Math \
	SCICore/TclInterface SCICore/Datatypes SCICore/Containers
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.6  2000/10/24 05:57:10  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.5  2000/09/07 20:40:16  zyp
# Added the SliceMaker module (it creates a disc for a demo)
#
# Revision 1.4  2000/06/08 22:46:13  moulding
# Added a comment note about not messing with the module maker comment lines,
# and how to edit this file by hand.
#
# Revision 1.3  2000/06/07 20:54:55  moulding
# made changes to allow the module maker to add to and edit this file
#
# Revision 1.2  2000/03/20 19:36:05  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:25:36  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
