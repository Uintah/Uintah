#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Core/Geom

SRCS +=	$(SRCDIR)/BBoxCache.cc		                \
	$(SRCDIR)/ColorMap.cc		                \
	$(SRCDIR)/ColorMapTex.cc	                \
	$(SRCDIR)/DirectionalLight.cc	                \
	$(SRCDIR)/FreeType.cc		                \
	$(SRCDIR)/GeomArrows.cc		                \
	$(SRCDIR)/GeomBillboard.cc	                \
	$(SRCDIR)/GeomBox.cc		                \
	$(SRCDIR)/GeomColorMap.cc	                \
	$(SRCDIR)/GeomCone.cc		                \
	$(SRCDIR)/GeomContainer.cc	                \
	$(SRCDIR)/GeomCull.cc	                \
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
	$(SRCDIR)/GeomTexRectangle.cc	    		\
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
	$(SRCDIR)/OpenGLContext.cc	    		\
	$(SRCDIR)/OpenGLViewport.cc	    		\
	$(SRCDIR)/Path.cc		    		\
	$(SRCDIR)/PBuffer.cc		    		\
	$(SRCDIR)/PointLight.cc		    		\
	$(SRCDIR)/SpotLight.cc		    		\
	$(SRCDIR)/ShaderProgramARB.cc                   \
	$(SRCDIR)/TexSquare.cc		    		\
	$(SRCDIR)/TimeGrid.cc		    		\
	$(SRCDIR)/TkOpenGLContext.cc	    		\
	$(SRCDIR)/View.cc		    		\
	$(SRCDIR)/tGrid.cc		    		\
	$(SRCDIR)/templates.cc

PSELIBS := Core/Persistent Core/Geometry Core/Exceptions \
	Core/Datatypes Core/Math Core/Containers Core/Thread \
	Core/GuiInterface Core/Util Core/TkExtensions

LIBS := $(FTGL_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY) $(TK_LIBRARY) $(FREETYPE_LIBRARY) $(PNG_LIBRARY) $(Z_LIBRARY)

INCLUDES += $(FTGL_INCLUDE) $(FREETYPE_INCLUDE)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

