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

SRCDIR   := PSECommon/Modules/Fields

SRCS     += \
	$(SRCDIR)/Downsample.cc\
	$(SRCDIR)/ExtractSurfs.cc\
        $(SRCDIR)/FieldFilter.cc\
	$(SRCDIR)/FieldGainCorrect.cc\
	$(SRCDIR)/FieldMedianFilter.cc\
	$(SRCDIR)/FieldRGAug.cc\
	$(SRCDIR)/FieldSeed.cc\
	$(SRCDIR)/GenField.cc\
	$(SRCDIR)/Gradient.cc\
	$(SRCDIR)/GradientMagnitude.cc\
	$(SRCDIR)/LocalMinMax.cc\
	$(SRCDIR)/MergeTensor.cc\
	$(SRCDIR)/OpenGL_Ex.cc\
	$(SRCDIR)/SFRGfile.cc\
	$(SRCDIR)/ShowGeometry.cc\
	$(SRCDIR)/TracePath.cc\
	$(SRCDIR)/TrainSeg2.cc\
	$(SRCDIR)/TrainSegment.cc\
	$(SRCDIR)/TransformField.cc\
	$(SRCDIR)/ScalarFieldProbe.cc\
	$(SRCDIR)/GenVectorField.cc\
	$(SRCDIR)/GenScalarField.cc\
#[INSERT NEW CODE FILE HERE]

#       $(SRCDIR)/ClipField.cc\

PSELIBS := PSECore/Dataflow PSECore/Datatypes PSECore/Widgets \
	SCICore/Persistent SCICore/Exceptions SCICore/Thread \
	SCICore/Containers SCICore/TclInterface SCICore/Geom \
	SCICore/Datatypes SCICore/Geometry SCICore/TkExtensions \
	SCICore/Math
LIBS := $(TK_LIBRARY) $(GL_LIBS) $(FLEX_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2.2.10  2000/11/01 23:02:55  mcole
# Fix for previous merge from trunk
#
# Revision 1.2.2.6  2000/10/27 16:29:21  mcole
# add back removed modules to compile
#
# Revision 1.2.2.5  2000/10/26 10:03:31  moulding
# merge HEAD into FIELD_REDESIGN
#
# Revision 1.2.2.4  2000/09/28 03:16:53  mcole
# merge trunk into FIELD_REDESIGN branch
#
# Revision 1.2.2.3  2000/09/21 04:34:29  mcole
# initial checkin of showGeometry module
#
# Revision 1.2.2.2  2000/09/11 16:17:49  kuehne
# updates to field redesign
#
# Revision 1.2.2.1  2000/06/07 17:28:46  kuehne
# Added GenField module.  Creates a scalar field from a specified equation and bounds.
#
# Revision 1.8  2000/10/29 04:34:52  dmw
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
# Revision 1.7  2000/10/24 05:57:33  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.6  2000/07/23 18:30:11  dahart
# Initial commit / Modules to generate scalar & vector fields from
# symbolic functions
#
# Revision 1.5  2000/06/16 04:17:06  samsonov
# *** empty log message ***
#
# Revision 1.4  2000/06/08 22:46:26  moulding
# Added a comment note about not messing with the module maker comment lines,
# and how to edit this file by hand.
#
# Revision 1.3  2000/06/07 00:11:36  moulding
# made some modifications that will allow the module make to edit and add
# to this file
#
# Revision 1.2  2000/03/20 19:36:57  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:01  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
