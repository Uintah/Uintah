# Makefile fragment for this subdirectory

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/Datatypes

SRCS += $(SRCDIR)/BasicSurfaces.cc	    \
	$(SRCDIR)/Brick.cc		    \
        $(SRCDIR)/ColorMap.cc		    \
        $(SRCDIR)/ColumnMatrix.cc	    \
        $(SRCDIR)/ContourGeom.cc	    \
        $(SRCDIR)/Datatype.cc		    \
        $(SRCDIR)/DenseMatrix.cc	    \
        $(SRCDIR)/Field.cc		    \
        $(SRCDIR)/GenFunction.cc	    \
        $(SRCDIR)/Geom.cc		    \
        $(SRCDIR)/HexMesh.cc		    \
        $(SRCDIR)/Image.cc		    \
        $(SRCDIR)/LatticeGeom.cc	    \
        $(SRCDIR)/Matrix.cc		    \
        $(SRCDIR)/Mesh.cc		    \
        $(SRCDIR)/MeshGeom.cc		    \
        $(SRCDIR)/MeshRG.cc		    \
        $(SRCDIR)/Path.cc		    \
        $(SRCDIR)/PointCloudGeom.cc	    \
        $(SRCDIR)/PropertyManager.cc	    \
        $(SRCDIR)/ScalarField.cc	    \
        $(SRCDIR)/ScalarFieldHUG.cc	    \
        $(SRCDIR)/ScalarFieldRG.cc          \
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
	$(SRCDIR)/TriSurfGeom.cc	    \
        $(SRCDIR)/TetMeshGeom.cc	    \
        $(SRCDIR)/TriDiagonalMatrix.cc	    \
        $(SRCDIR)/TriSurface.cc		    \
        $(SRCDIR)/TriSurfGeom.cc	    \
	$(SRCDIR)/TypeName.cc		    \
        $(SRCDIR)/UnstructuredGeom.cc	    \
        $(SRCDIR)/VectorField.cc	    \
        $(SRCDIR)/VectorFieldHUG.cc	    \
        $(SRCDIR)/VectorFieldRG.cc	    \
        $(SRCDIR)/VectorFieldRGCC.cc	    \
        $(SRCDIR)/VectorFieldUG.cc	    \
        $(SRCDIR)/VectorFieldZone.cc	    \
        $(SRCDIR)/VoidStar.cc		    \
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

PSELIBS := Core/Persistent Core/Exceptions Core/Containers \
	Core/Thread Core/Geometry Core/Geom Core/TclInterface \
	Core/Math Core/Util
LIBS := $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

clean::
	rm -f $(GENSRCS)
	rm -f $(patsubst %.cc,%.h,$(GENSRCS))

