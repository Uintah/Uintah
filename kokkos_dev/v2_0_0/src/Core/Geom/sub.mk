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

SRCDIR   := Core/Geom

SRCS +=	$(SRCDIR)/BBoxCache.cc		                \
	$(SRCDIR)/ColorMap.cc		                \
	$(SRCDIR)/ColorMapTex.cc	                \
	$(SRCDIR)/DirectionalLight.cc	                \
	$(SRCDIR)/GeomArrows.cc		                \
	$(SRCDIR)/GeomBillboard.cc	                \
	$(SRCDIR)/GeomBox.cc		                \
	$(SRCDIR)/GeomColorMap.cc	                \
	$(SRCDIR)/GeomCone.cc		                \
	$(SRCDIR)/GeomContainer.cc	                \
	$(SRCDIR)/GeomCylinder.cc	                \
	$(SRCDIR)/GeomDL.cc                             \
	$(SRCDIR)/GeomDisc.cc		                \
	$(SRCDIR)/GeomEllipsoid.cc	                \
	$(SRCDIR)/GeomGrid.cc		                \
	$(SRCDIR)/GeomGroup.cc		                \
	$(SRCDIR)/GeomLine.cc		                \
	$(SRCDIR)/GeomObj.cc                            \
	$(SRCDIR)/GeomOpenGL.cc		                \
	$(SRCDIR)/Path.cc		    	        \
	$(SRCDIR)/GeomPick.cc		                \
	$(SRCDIR)/GeomPoint.cc		    		\
	$(SRCDIR)/GeomPolyline.cc	                \
	$(SRCDIR)/GeomQMesh.cc		                \
	$(SRCDIR)/GeomQuads.cc		                \
	$(SRCDIR)/GeomRenderMode.cc	                \
	$(SRCDIR)/GeomScene.cc		                \
	$(SRCDIR)/GeomSphere.cc		                \
	$(SRCDIR)/GeomSticky.cc		                \
	$(SRCDIR)/GeomSwitch.cc		                \
	$(SRCDIR)/GeomTetra.cc		                \
	$(SRCDIR)/GeomTexSlices.cc	                \
	$(SRCDIR)/GeomText.cc		                \
	$(SRCDIR)/GeomTimeGroup.cc	                \
	$(SRCDIR)/GeomTorus.cc		                \
	$(SRCDIR)/GeomTransform.cc	                \
	$(SRCDIR)/GeomTri.cc		                \
	$(SRCDIR)/GeomTriStrip.cc	                \
	$(SRCDIR)/GeomTriangles.cc	                \
	$(SRCDIR)/GeomTube.cc		                \
	$(SRCDIR)/GeomVertexPrim.cc	                \
	$(SRCDIR)/GuiGeom.cc		    		\
	$(SRCDIR)/GuiView.cc		    		\
	$(SRCDIR)/HeadLight.cc		    		\
	$(SRCDIR)/HistogramTex.cc	    		\
	$(SRCDIR)/IndexedGroup.cc	    		\
	$(SRCDIR)/Light.cc		    		\
	$(SRCDIR)/Lighting.cc		    		\
	$(SRCDIR)/Material.cc		    		\
	$(SRCDIR)/Path.cc		    		\
	$(SRCDIR)/PointLight.cc		    		\
	$(SRCDIR)/SpotLight.cc		    		\
	$(SRCDIR)/TexSquare.cc		    		\
	$(SRCDIR)/TimeGrid.cc		    		\
	$(SRCDIR)/View.cc		    		\
	$(SRCDIR)/tGrid.cc		    		\
	$(SRCDIR)/templates.cc

PSELIBS := Core/Persistent Core/Geometry Core/Exceptions \
	Core/Datatypes Core/Math Core/Containers Core/Thread \
	Core/GuiInterface

LIBS := $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

