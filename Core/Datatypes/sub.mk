#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Core/Datatypes

SRCS += $(SRCDIR)/Brick.cc		    \
        $(SRCDIR)/ColorMap.cc		    \
        $(SRCDIR)/ColumnMatrix.cc	    \
	$(SRCDIR)/ContourMesh.cc            \
        $(SRCDIR)/cd_templates.cc	    \
        $(SRCDIR)/Datatype.cc		    \
        $(SRCDIR)/DenseMatrix.cc	    \
        $(SRCDIR)/Field.cc		    \
        $(SRCDIR)/FieldSet.cc		    \
        $(SRCDIR)/GenFunction.cc	    \
        $(SRCDIR)/Geom.c          	    \
        $(SRCDIR)/Image.cc		    \
        $(SRCDIR)/ImageMesh.cc		    \
        $(SRCDIR)/Matrix.cc		    \
        $(SRCDIR)/MeshBase.cc		    \
        $(SRCDIR)/Path.cc		    \
	$(SRCDIR)/PointCloudMesh.cc         \
        $(SRCDIR)/PropertyManager.cc	    \
	$(SRCDIR)/ScanlineMesh.cc           \
        $(SRCDIR)/SparseRowMatrix.cc	    \
        $(SRCDIR)/SymSparseRowMatrix.cc	    \
        $(SRCDIR)/TetVolMesh.cc 	    \
        $(SRCDIR)/TriDiagonalMatrix.cc	    \
        $(SRCDIR)/TriSurfMesh.cc	    \
	$(SRCDIR)/TypeName.cc		    \
        $(SRCDIR)/VoidStar.cc		    \
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
	$(SRCDIR)/LatVolMesh.cc \



PSELIBS := Core/Persistent Core/Exceptions Core/Containers \
	Core/Thread Core/Geometry Core/Geom Core/GuiInterface \
	Core/Math Core/Util
LIBS := $(GL_LIBS) -lm

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

clean::
	rm -f $(GENSRCS)
	rm -f $(patsubst %.cc,%.h,$(GENSRCS))

