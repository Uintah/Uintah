# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/Datatypes

GENSRCS := $(SRCDIR)/ScalarFieldRG.cc $(SRCDIR)/ScalarFieldRGchar.cc \
	$(SRCDIR)/ScalarFieldRGuchar.cc $(SRCDIR)/ScalarFieldRGshort.cc \
	$(SRCDIR)/ScalarFieldRGushort.cc \
	$(SRCDIR)/ScalarFieldRGint.cc $(SRCDIR)/ScalarFieldRGfloat.cc \
	$(SRCDIR)/ScalarFieldRGdouble.cc

GENHDRS := $(patsubst %.cc,%.h,$(GENSRCS))


SRCS += $(GENSRCS) \
        $(SRCDIR)/Attrib.cc                 \
        $(SRCDIR)/BasicSurfaces.cc	    \
        $(SRCDIR)/Boolean.cc		    \
	$(SRCDIR)/Brick.cc		    \
        $(SRCDIR)/CameraView.cc		    \
        $(SRCDIR)/ColorMap.cc		    \
        $(SRCDIR)/ColumnMatrix.cc	    \
        $(SRCDIR)/ContourGeom.cc	    \
        $(SRCDIR)/Datatype.cc		    \
        $(SRCDIR)/DenseMatrix.cc	    \
        $(SRCDIR)/Domain.cc		    \
        $(SRCDIR)/Field.cc		    \
        $(SRCDIR)/FieldWrapper.cc	    \
        $(SRCDIR)/GenFunction.cc	    \
        $(SRCDIR)/Geom.cc		    \
        $(SRCDIR)/HexMesh.cc		    \
        $(SRCDIR)/Image.cc		    \
        $(SRCDIR)/Interval.cc		    \
        $(SRCDIR)/LatticeGeom.cc	    \
        $(SRCDIR)/Matrix.cc		    \
        $(SRCDIR)/Mesh.cc		    \
        $(SRCDIR)/MeshGeom.cc		    \
        $(SRCDIR)/Path.cc		    \
        $(SRCDIR)/PointCloudGeom.cc	    \
        $(SRCDIR)/SField.cc		    \
        $(SRCDIR)/ScalarField.cc	    \
        $(SRCDIR)/ScalarFieldHUG.cc	    \
        $(SRCDIR)/ScalarFieldRGBase.cc      \
        $(SRCDIR)/ScalarFieldRGCC.cc        \
        $(SRCDIR)/ScalarFieldUG.cc	    \
        $(SRCDIR)/ScalarFieldZone.cc	    \
        $(SRCDIR)/SparseRowMatrix.cc	    \
        $(SRCDIR)/StructuredGeom.cc	    \
        $(SRCDIR)/SurfTree.cc		    \
        $(SRCDIR)/Surface.cc		    \
        $(SRCDIR)/SurfaceGeom.cc	    \
        $(SRCDIR)/SymSparseRowMatrix.cc	    \
        $(SRCDIR)/TField.cc		    \
        $(SRCDIR)/TetMeshGeom.cc	    \
        $(SRCDIR)/TriDiagonalMatrix.cc	    \
        $(SRCDIR)/TriSurface.cc		    \
        $(SRCDIR)/TriSurfaceGeom.cc	    \
	$(SRCDIR)/TypeName.cc		    \
        $(SRCDIR)/UnstructuredGeom.cc	    \
        $(SRCDIR)/VField.cc		    \
        $(SRCDIR)/VectorField.cc	    \
        $(SRCDIR)/VectorFieldHUG.cc	    \
        $(SRCDIR)/VectorFieldOcean.cc	    \
        $(SRCDIR)/VectorFieldRG.cc	    \
        $(SRCDIR)/VectorFieldRGCC.cc	    \
        $(SRCDIR)/VectorFieldUG.cc	    \
        $(SRCDIR)/VectorFieldZone.cc	    \
        $(SRCDIR)/VoidStar.cc		    \
        $(SRCDIR)/cDMatrix.cc		    \
        $(SRCDIR)/cMatrix.cc		    \
        $(SRCDIR)/cSMatrix.cc		    \
        $(SRCDIR)/cVector.cc		    \
        $(SRCDIR)/templates.cc		    \
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

PSELIBS := Core/Persistent Core/Exceptions Core/Containers \
	Core/Thread Core/Geometry Core/Geom Core/TclInterface \
	Core/Math Core/Util
LIBS := $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

clean::
	rm -f $(GENSRCS)
	rm -f $(patsubst %.cc,%.h,$(GENSRCS))

