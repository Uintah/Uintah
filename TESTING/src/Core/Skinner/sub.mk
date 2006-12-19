#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
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

SRCDIR   := Core/Skinner

SRCS     += \
	$(SRCDIR)/Animation.cc	    	\
	$(SRCDIR)/Arc.cc	    	\
	$(SRCDIR)/Arrow.cc	    	\
	$(SRCDIR)/Arithmetic.cc	    	\
	$(SRCDIR)/Box.cc		\
	$(SRCDIR)/Collection.cc	    	\
	$(SRCDIR)/Color.cc	    	\
	$(SRCDIR)/ColorMap1D.cc	    	\
	$(SRCDIR)/Drawable.cc    	\
	$(SRCDIR)/Frame.cc	    	\
	$(SRCDIR)/FocusRegion.cc    	\
	$(SRCDIR)/FocusGrab.cc    	\
	$(SRCDIR)/Gradient.cc	    	\
	$(SRCDIR)/GeomSkinnerVarSwitch.cc	    	\
	$(SRCDIR)/Graph2D.cc	    	\
	$(SRCDIR)/Grid.cc	    	\
	$(SRCDIR)/Layout.cc	    	\
	$(SRCDIR)/MenuList.cc	    	\
	$(SRCDIR)/MenuManager.cc	\
	$(SRCDIR)/MenuButton.cc	\
	$(SRCDIR)/Parent.cc	    	\
	$(SRCDIR)/Progress.cc	    	\
	$(SRCDIR)/RectRegion.cc    	\
	$(SRCDIR)/Root.cc        	\
	$(SRCDIR)/SceneGraph.cc  	\
	$(SRCDIR)/Signals.cc    	\
	$(SRCDIR)/Skinner.cc    	\
	$(SRCDIR)/Text.cc       	\
	$(SRCDIR)/TextEntry.cc       	\
	$(SRCDIR)/Texture.cc    	\
	$(SRCDIR)/Window.cc      	\
	$(SRCDIR)/Variables.cc      	\
	$(SRCDIR)/XMLIO.cc              \
	$(SRCDIR)/ViewSubRegion.cc    	\
	$(SRCDIR)/VisibilityGroup.cc    \
#	$(SRCDIR)/Colormap2D.cc	    	\


#	$(SRCDIR)/Histogram.cc	    	\
#	$(SRCDIR)/MenuItem.cc	    	\



PSELIBS := Core/Containers  \
           Core/Datatypes \
           Core/Events \
           Core/Exceptions \
	   Core/Geom  \
	   Core/Geometry  \
	   Core/Math  \
	   Core/Persistent  \
	   Core/Thread \
	   Core/Util \
	   Core/XMLUtil

LIBS := $(DL_LIBRARY) $(THREAD_LIBRARY) $(GL_LIBRARY) $(TEEM_LIBRARY) $(XML2_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

