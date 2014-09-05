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

SRCS     += $(SRCDIR)/GeomObj.cc $(SRCDIR)/GeomOpenGL.cc \
	$(SRCDIR)/GeomArrows.cc $(SRCDIR)/Pt.cc $(SRCDIR)/GeomText.cc \
	$(SRCDIR)/BBoxCache.cc $(SRCDIR)/ColorMapTex.cc \
	$(SRCDIR)/GeomGrid.cc $(SRCDIR)/HistogramTex.cc \
	$(SRCDIR)/GeomQMesh.cc $(SRCDIR)/TimeGrid.cc \
	$(SRCDIR)/GeomBillboard.cc $(SRCDIR)/GeomGroup.cc \
	$(SRCDIR)/GeomRenderMode.cc $(SRCDIR)/GeomTimeGroup.cc \
	$(SRCDIR)/GeomBox.cc $(SRCDIR)/HeadLight.cc $(SRCDIR)/GeomScene.cc \
	$(SRCDIR)/GeomTorus.cc $(SRCDIR)/Color.cc \
	$(SRCDIR)/IndexedGroup.cc $(SRCDIR)/GeomSphere.cc \
	$(SRCDIR)/GeomEllipsoid.cc \
	$(SRCDIR)/GeomTransform.cc $(SRCDIR)/GeomCone.cc \
	$(SRCDIR)/Light.cc $(SRCDIR)/Sticky.cc $(SRCDIR)/GeomTri.cc \
	$(SRCDIR)/GeomContainer.cc $(SRCDIR)/Lighting.cc \
	$(SRCDIR)/Switch.cc $(SRCDIR)/GeomTriStrip.cc \
	$(SRCDIR)/GeomCylinder.cc $(SRCDIR)/GeomLine.cc \
	$(SRCDIR)/GeomTriangles.cc $(SRCDIR)/GeomDisc.cc \
	$(SRCDIR)/Material.cc $(SRCDIR)/GeomTube.cc $(SRCDIR)/GeomTetra.cc \
	$(SRCDIR)/GeomVertexPrim.cc $(SRCDIR)/PointLight.cc \
	$(SRCDIR)/GeomTexSlices.cc $(SRCDIR)/View.cc \
	$(SRCDIR)/GeomPolyline.cc $(SRCDIR)/TexSquare.cc \
	$(SRCDIR)/tGrid.cc $(SRCDIR)/GeomPick.cc $(SRCDIR)/Pickable.cc \
	$(SRCDIR)/templates.cc 	$(SRCDIR)/GeomDL.cc
#	$(SRCDIR)/GuiGeom.cc $(SRCDIR)/GuiView.cc \

PSELIBS := Core/Persistent Core/Geometry Core/Exceptions \
	Core/Math Core/Containers Core/Thread 
LIBS := $(GL_LIBS) -lm

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

