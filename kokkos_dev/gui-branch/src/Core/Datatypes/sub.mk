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

SRCS += $(SRCDIR)/Clipper.cc		    	\
	$(SRCDIR)/ColorMap.cc		    	\
        $(SRCDIR)/ColumnMatrix.cc	    	\
	$(SRCDIR)/ContourMesh.cc            	\
        $(SRCDIR)/cd_templates.cc	    	\
        $(SRCDIR)/cd_templates_fields_0.cc    	\
        $(SRCDIR)/cd_templates_fields_1.cc    	\
        $(SRCDIR)/cd_templates_fields_2.cc    	\
        $(SRCDIR)/cd_templates_fields_3.cc    	\
        $(SRCDIR)/cd_templates_fields_4.cc    	\
        $(SRCDIR)/cd_templates_fields_5.cc    	\
        $(SRCDIR)/Datatype.cc		    	\
        $(SRCDIR)/DenseMatrix.cc	    	\
        $(SRCDIR)/Field.cc		    	\
        $(SRCDIR)/FieldSet.cc		    	\
        $(SRCDIR)/GenFunction.cc	    	\
        $(SRCDIR)/Geom.c          	    	\
	$(SRCDIR)/HexVolMesh.cc			\
        $(SRCDIR)/Image.cc		    	\
        $(SRCDIR)/ImageMesh.cc		    	\
	$(SRCDIR)/LatVolMesh.cc 		\
        $(SRCDIR)/Matrix.cc		    	\
        $(SRCDIR)/Mesh.cc		    	\
        $(SRCDIR)/Path.cc		    	\
	$(SRCDIR)/PointCloudMesh.cc         	\
        $(SRCDIR)/PropertyManager.cc	    	\
	$(SRCDIR)/QuadSurfMesh.cc               \
	$(SRCDIR)/ScanlineMesh.cc           	\
        $(SRCDIR)/SparseRowMatrix.cc	    	\
        $(SRCDIR)/TetVolMesh.cc 	    	\
        $(SRCDIR)/TriSurfMesh.cc	    	\
	$(SRCDIR)/TypeName.cc		    	\
	$(SRCDIR)/Brick.cc		    	\
	$(SRCDIR)/FullRes.cc 			\
	$(SRCDIR)/FullResIterator.cc 		\
	$(SRCDIR)/GLAttenuate.cc 		\
	$(SRCDIR)/GLMIP.cc 			\
	$(SRCDIR)/GLOverOp.cc 			\
	$(SRCDIR)/GLPlanes.cc 			\
        $(SRCDIR)/GLTexture3D.cc 		\
	$(SRCDIR)/GLTextureIterator.cc 		\
	$(SRCDIR)/GLTexRenState.cc 		\
	$(SRCDIR)/GLVolRenState.cc 		\
	$(SRCDIR)/GLVolumeRenderer.cc 		\
	$(SRCDIR)/LOS.cc 			\
	$(SRCDIR)/LOSIterator.cc 		\
	$(SRCDIR)/Polygon.cc 			\
	$(SRCDIR)/ROI.cc 			\
	$(SRCDIR)/ROIIterator.cc 		\
	$(SRCDIR)/SliceTable.cc 		\
	$(SRCDIR)/TexPlanes.cc 			\
	$(SRCDIR)/VolumeUtils.cc 		\



PSELIBS := Core/Persistent Core/Disclosure Core/Exceptions Core/Containers \
	Core/Thread Core/Geometry Core/Geom Core/GuiInterface \
	Core/Math Core/Util
LIBS := $(GL_LIBS) -lm

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

clean::
	rm -f $(GENSRCS)
	rm -f $(patsubst %.cc,%.h,$(GENSRCS))

