#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCICore/Geom

SRCS     += $(SRCDIR)/GeomObj.cc $(SRCDIR)/GeomOpenGL.cc \
	$(SRCDIR)/GeomArrows.cc $(SRCDIR)/Pt.cc $(SRCDIR)/GeomText.cc \
	$(SRCDIR)/BBoxCache.cc $(SRCDIR)/GeomGrid.cc \
	$(SRCDIR)/GeomQMesh.cc $(SRCDIR)/TimeGrid.cc \
	$(SRCDIR)/GeomBillboard.cc $(SRCDIR)/GeomGroup.cc \
	$(SRCDIR)/GeomRenderMode.cc $(SRCDIR)/GeomTimeGroup.cc \
	$(SRCDIR)/GeomBox.cc $(SRCDIR)/HeadLight.cc $(SRCDIR)/GeomScene.cc \
	$(SRCDIR)/GeomTorus.cc $(SRCDIR)/Color.cc \
	$(SRCDIR)/IndexedGroup.cc $(SRCDIR)/GeomSphere.cc \
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
	$(SRCDIR)/TCLGeom.cc $(SRCDIR)/TCLView.cc

PSELIBS := SCICore/Persistent SCICore/Geometry SCICore/Exceptions \
	SCICore/Math SCICore/Containers SCICore/Thread \
	SCICore/TclInterface
LIBS := $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:37:40  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:25  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
