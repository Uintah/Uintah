#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk


SRCDIR   := SCICore/Datatypes

GENSRCS := $(SRCDIR)/ScalarFieldRG.cc $(SRCDIR)/ScalarFieldRGchar.cc \
	$(SRCDIR)/ScalarFieldRGuchar.cc $(SRCDIR)/ScalarFieldRGshort.cc \
	$(SRCDIR)/ScalarFieldRGushort.cc \
	$(SRCDIR)/ScalarFieldRGint.cc $(SRCDIR)/ScalarFieldRGfloat.cc \
	$(SRCDIR)/ScalarFieldRGdouble.cc

GENHDRS := $(patsubst %.cc,%.h,$(GENSRCS))

SRCS += $(GENSRCS) $(SRCDIR)/TriSurface.cc $(SRCDIR)/BasicSurfaces.cc \
	$(SRCDIR)/Boolean.cc $(SRCDIR)/ColorMap.cc \
	$(SRCDIR)/ColumnMatrix.cc $(SRCDIR)/Datatype.cc \
	$(SRCDIR)/DenseMatrix.cc $(SRCDIR)/HexMesh.cc \
	$(SRCDIR)/Image.cc $(SRCDIR)/Interval.cc $(SRCDIR)/Matrix.cc \
	$(SRCDIR)/Mesh.cc $(SRCDIR)/ScalarField.cc \
	$(SRCDIR)/ScalarFieldHUG.cc $(SRCDIR)/ScalarFieldRGBase.cc \
	$(SRCDIR)/ScalarFieldUG.cc $(SRCDIR)/ScalarFieldZone.cc \
	$(SRCDIR)/SparseRowMatrix.cc $(SRCDIR)/Surface.cc \
	$(SRCDIR)/SymSparseRowMatrix.cc $(SRCDIR)/TriDiagonalMatrix.cc \
	$(SRCDIR)/VectorField.cc $(SRCDIR)/VectorFieldHUG.cc \
	$(SRCDIR)/VectorFieldOcean.cc $(SRCDIR)/VectorFieldRG.cc \
	$(SRCDIR)/VectorFieldUG.cc $(SRCDIR)/VectorFieldZone.cc \
	$(SRCDIR)/VoidStar.cc $(SRCDIR)/cDMatrix.cc \
	$(SRCDIR)/cMatrix.cc $(SRCDIR)/cSMatrix.cc $(SRCDIR)/cVector.cc \
	$(SRCDIR)/SurfTree.cc $(SRCDIR)/ScalarFieldRGCC.cc \
	$(SRCDIR)/VectorFieldRGCC.cc $(SRCDIR)/templates.cc \
	$(SRCDIR)/Geom.cc $(SRCDIR)/Attrib.cc \
	$(SRCDIR)/Field.cc \
	$(SRCDIR)/FieldWrapper.cc $(SRCDIR)/Domain.cc \
	$(SRCDIR)/SField.cc $(SRCDIR)/VField.cc \
	$(SRCDIR)/TField.cc $(SRCDIR)/Lattice3Geom.cc \
	$(SRCDIR)/StructuredGeom.cc $(SRCDIR)/UnstructuredGeom.cc \
	$(SRCDIR)/MeshGeom.cc $(SRCDIR)/PointCloudGeom.cc \
	$(SRCDIR)/ContourGeom.cc $(SRCDIR)/TetMeshGeom.cc \
	$(SRCDIR)/SurfaceGeom.cc $(SRCDIR)/TriSurfaceGeom.cc \
	$(SRCDIR)/Path.cc $(SRCDIR)/CameraView.cc $(SRCDIR)/GenFunction.cc\
	$(SRCDIR)/VectorFieldRGCC.cc  \
	$(SRCDIR)/Path.cc $(SRCDIR)/CameraView.cc $(SRCDIR)/templates.cc

$(SRCDIR)/ScalarFieldRG.h: $(SRCDIR)/ScalarFieldRGTYPE.h
	sed -e 's/RGTYPE/RG/g' -e 's/TYPE/double/g' < $< > $@

$(SRCDIR)/ScalarFieldRGchar.h: $(SRCDIR)/ScalarFieldRGTYPE.h
	sed 's/TYPE/char/g' < $< > $@

$(SRCDIR)/ScalarFieldRGuchar.h: $(SRCDIR)/ScalarFieldRGTYPE.h
	sed 's/TYPE/uchar/g' < $< > $@

$(SRCDIR)/ScalarFieldRGshort.h: $(SRCDIR)/ScalarFieldRGTYPE.h
	sed 's/TYPE/short/g' < $< > $@

$(SRCDIR)/ScalarFieldRGushort.h: $(SRCDIR)/ScalarFieldRGTYPE.h
	sed 's/TYPE/ushort/g' < $< > $@

$(SRCDIR)/ScalarFieldRGint.h: $(SRCDIR)/ScalarFieldRGTYPE.h
	sed 's/TYPE/int/g' < $< > $@

$(SRCDIR)/ScalarFieldRGfloat.h: $(SRCDIR)/ScalarFieldRGTYPE.h
	sed 's/TYPE/float/g' < $< > $@

$(SRCDIR)/ScalarFieldRGdouble.h: $(SRCDIR)/ScalarFieldRGTYPE.h
	sed 's/TYPE/double/g' < $< > $@


# .CC

$(SRCDIR)/ScalarFieldRG.cc: $(SRCDIR)/ScalarFieldRGdouble.cc $(SRCDIR)/ScalarFieldRG.h
	sed 's/RGdouble/RG/g' < $< > $@

$(SRCDIR)/ScalarFieldRGchar.cc: $(SRCDIR)/ScalarFieldRGTYPE.cc $(SRCDIR)/ScalarFieldRGchar.h
	sed 's/TYPE/char/g' < $< > $@

$(SRCDIR)/ScalarFieldRGuchar.cc: $(SRCDIR)/ScalarFieldRGTYPE.cc $(SRCDIR)/ScalarFieldRGuchar.h
	sed 's/TYPE/uchar/g' < $< > $@

$(SRCDIR)/ScalarFieldRGshort.cc: $(SRCDIR)/ScalarFieldRGTYPE.cc $(SRCDIR)/ScalarFieldRGshort.h
	sed 's/TYPE/short/g' < $< >$@

$(SRCDIR)/ScalarFieldRGushort.cc: $(SRCDIR)/ScalarFieldRGTYPE.cc $(SRCDIR)/ScalarFieldRGushort.h
	sed 's/TYPE/ushort/g' < $< >$@

$(SRCDIR)/ScalarFieldRGint.cc: $(SRCDIR)/ScalarFieldRGTYPE.cc $(SRCDIR)/ScalarFieldRGint.h
	sed 's/TYPE/int/g' < $< > $@

$(SRCDIR)/ScalarFieldRGfloat.cc: $(SRCDIR)/ScalarFieldRGTYPE.cc $(SRCDIR)/ScalarFieldRGfloat.h
	sed 's/TYPE/float/g' < $< > $@

$(SRCDIR)/ScalarFieldRGdouble.cc: $(SRCDIR)/ScalarFieldRGTYPE.cc $(SRCDIR)/ScalarFieldRGdouble.h
	sed 's/TYPE/double/g' < $< > $@

PSELIBS := SCICore/Persistent SCICore/Exceptions SCICore/Containers \
	SCICore/Thread SCICore/Geometry SCICore/Geom SCICore/TclInterface \
	SCICore/Math
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

clean::
	rm -f $(GENSRCS)
	rm -f $(patsubst %.cc,%.h,$(GENSRCS))

#
# $Log$
# Revision 1.3.2.13  2000/10/26 10:04:22  moulding
# merge HEAD into FIELD_REDESIGN
#
# Revision 1.3.2.12  2000/10/10 23:02:17  michaelc
# move from {SVT}Field ports to Field ports
#
# Revision 1.3.2.11  2000/09/29 22:45:25  samsonov
# added GenFunction.cc into build
#
# Revision 1.3.2.10  2000/09/28 03:13:32  mcole
# merge trunk into FIELD_REDESIGN branch
#
# Revision 1.3.2.9  2000/09/25 21:52:08  yarden
# move GenSField.cc into GenSField.h
# (template implementation should be in .h file
# for portablity)
#
# Revision 1.3.2.8  2000/09/22 17:31:23  michaelc
# Cleanup data members, rearrange LatticeGeom
#
# Revision 1.3.2.7  2000/09/11 16:11:57  kuehne
# updates to field redesign
#
# Revision 1.3.2.6  2000/08/31 23:09:08  mcole
# fix build, bad comment
#
# Revision 1.3.2.5  2000/08/24 16:36:57  mcole
# remove GenFunction from build until it compiles
#
# Revision 1.3.2.4  2000/08/22 20:32:30  samsonov
# *** empty log message ***
#
# Revision 1.3.2.3  2000/08/14 21:29:45  michaelc
# Attributes as general acceleration class
#
# Revision 1.3.2.2  2000/08/03 16:52:52  kuehne
# Initial commit
#
# Revision 1.3.2.1  2000/06/07 17:42:25  kuehne
# Added datastructures used by the new fields
#
# Revision 1.9  2000/08/20 04:19:50  samsonov
# path to CameraView.cc
#
# Revision 1.8  2000/08/09 07:15:55  samsonov
# final version and Cocoon comments
#
# Revision 1.7  2000/07/19 06:39:38  samsonov
# Path datatype moved form DaveW
#
# Revision 1.6  2000/07/17 18:33:38  dmw
# deleted ushort and added it to sub.mk
#
# Revision 1.5  2000/07/17 16:29:40  bigler
# Removed reference to ScalarFieldRGushort.cc which did not exist.
#
# Revision 1.4  2000/07/12 15:45:11  dmw
# Added Yarden's raw output thing to matrices, added neighborhood accessors to meshes, added ScalarFieldRGushort
#
# Revision 1.3  2000/03/21 03:01:28  sparker
# Partially fixed special_get method in SimplePort
# Pre-instantiated a few key template types, in an attempt to reduce
#   initial compile time and reduce code bloat.
# Manually instantiated templates are in */*/templates.cc
#
# Revision 1.2  2000/03/20 19:37:35  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:19  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
