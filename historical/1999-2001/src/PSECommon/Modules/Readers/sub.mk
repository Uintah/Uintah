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

SRCDIR   := PSECommon/Modules/Readers

SRCS     += \
	$(SRCDIR)/ColorMapReader.cc\
	$(SRCDIR)/ColumnMatrixReader.cc\
	$(SRCDIR)/DukeRawRead.cc\
	$(SRCDIR)/GeomReader.cc\
	$(SRCDIR)/GeometryReader.cc\
	$(SRCDIR)/ImageReader.cc\
	$(SRCDIR)/MatrixReader.cc\
	$(SRCDIR)/MeshReader.cc\
	$(SRCDIR)/MultiSFRGReader.cc\
	$(SRCDIR)/PointsReader.cc\
	$(SRCDIR)/ScalarFieldReader.cc\
	$(SRCDIR)/SurfaceReader.cc\
	$(SRCDIR)/VectorFieldReader.cc\
	$(SRCDIR)/VoidStarReader.cc\
	$(SRCDIR)/PathReader.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := PSECore/Dataflow PSECore/Datatypes SCICore/Datatypes \
	SCICore/Persistent SCICore/Exceptions SCICore/Thread \
	SCICore/Containers SCICore/TclInterface SCICore/Geom
LIBS := 

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.6  2000/10/24 05:57:36  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.5  2000/07/18 23:12:57  samsonov
# Added PathReader module
#
# Revision 1.4  2000/06/08 22:46:28  moulding
# Added a comment note about not messing with the module maker comment lines,
# and how to edit this file by hand.
#
# Revision 1.3  2000/06/07 00:11:39  moulding
# made some modifications that will allow the module make to edit and add
# to this file
#
# Revision 1.2  2000/03/20 19:37:02  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:14  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
