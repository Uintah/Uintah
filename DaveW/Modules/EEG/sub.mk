#
# Makefile fragment for this subdirectory
# $Id$
#

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
#[INSERT NEW MODULE HERE]

PSELIBS := DaveW/Datatypes/General PSECore/Datatypes PSECore/Widgets \
	PSECore/Dataflow SCICore/Persistent SCICore/Exceptions \
	SCICore/Geom SCICore/Thread SCICore/Geometry SCICore/Math \
	SCICore/TclInterface SCICore/Datatypes SCICore/Containers
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
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
