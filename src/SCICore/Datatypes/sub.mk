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
	$(SRCDIR)/VectorFieldRGCC.cc  \
	$(SRCDIR)/Path.cc $(SRCDIR)/CameraView.cc $(SRCDIR)/templates.cc \
         $(SRCDIR)/Brick.cc \
         $(SRCDIR)/GLTexture3D.cc \
         $(SRCDIR)/GLTextureIterator.cc \
         $(SRCDIR)/GLTexRenState.cc \
         $(SRCDIR)/GLOverOp.cc \
         $(SRCDIR)/GLMIP.cc \
         $(SRCDIR)/GLVolRenState.cc \
         $(SRCDIR)/GLAttenuate.cc \
         $(SRCDIR)/GLPlanes.cc \
         $(SRCDIR)/FullRes.cc \
         $(SRCDIR)/FullResIterator.cc \
         $(SRCDIR)/LOS.cc \
         $(SRCDIR)/LOSIterator.cc \
         $(SRCDIR)/ROI.cc \
         $(SRCDIR)/ROIIterator.cc \
         $(SRCDIR)/TexPlanes.cc \
         $(SRCDIR)/GLVolumeRenderer.cc \
         $(SRCDIR)/Polygon.cc \
         $(SRCDIR)/SliceTable.cc \
         $(SRCDIR)/VolumeUtils.cc \


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
LIBS := $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

clean::
	rm -f $(GENSRCS)
	rm -f $(patsubst %.cc,%.h,$(GENSRCS))

#
# $Log$
# Revision 1.10  2000/12/09 17:36:55  moulding
# Port Kurt' volume rendering stuff to linux and move it to PSECommon.
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
