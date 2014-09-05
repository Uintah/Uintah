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

#
# $Log$
# Revision 1.2.2.4  2000/11/01 23:02:58  mcole
# Fix for previous merge from trunk
#
# Revision 1.2.2.2  2000/10/26 10:03:44  moulding
# merge HEAD into FIELD_REDESIGN
#
# Revision 1.2.2.1  2000/09/28 03:15:30  mcole
# merge trunk into FIELD_REDESIGN branch
#
# Revision 1.7  2000/10/29 04:34:57  dmw
# BuildFEMatrix -- ground an arbitrary node
# SolveMatrix -- when preconditioning, be careful with 0's on diagonal
# MeshReader -- build the grid when reading
# SurfToGeom -- support node normals
# IsoSurface -- fixed tet mesh bug
# MatrixWriter -- support split file (header + raw data)
#
# LookupSplitSurface -- split a surface across a place and lookup values
# LookupSurface -- find surface nodes in a sfug and copy values
# Current -- compute the current of a potential field (- grad sigma phi)
# LocalMinMax -- look find local min max points in a scalar field
#
# Revision 1.6  2000/10/24 05:57:38  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.5  2000/08/04 19:19:45  dmw
# adding TransformSurface.cc to makefile
#
# Revision 1.4  2000/06/08 22:46:30  moulding
# Added a comment note about not messing with the module maker comment lines,
# and how to edit this file by hand.
#
# Revision 1.3  2000/06/07 00:11:40  moulding
# made some modifications that will allow the module make to edit and add
# to this file
#
# Revision 1.2  2000/03/20 19:37:05  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:23  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
