# 
# 
# The MIT License
# 
# Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
# Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
# University of Utah.
# 
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
# 
# 
# 
# 
# Makefile fragment for this subdirectory
include $(SRCTOP)/scripts/smallso_prologue.mk


SRCDIR   := Packages/Kurt/Dataflow/Modules/Visualization

SRCS     += \
	$(SRCDIR)/GLSLShader.cc \
	$(SRCDIR)/ParticleFlowRenderer.cc \
	$(SRCDIR)/ParticleFlow.cc\
#[INSERT NEW CODE FILE HERE]
# 	$(SRCDIR)/GridVolVis.cc \
# 	$(SRCDIR)/GridSliceVis.cc \
#	$(SRCDIR)/HarvardVis.cc \
# 	$(SRCDIR)/SCIRex.cc \
#	$(SRCDIR)/ParticleColorMapKey.cc \

PSELIBS := \
	Dataflow/Network \
	Dataflow/Modules/Visualization Core/Datatypes \
        Core/Thread Core/Persistent Core/Exceptions \
        Core/GuiInterface Core/Containers Core/Datatypes \
        Core/Geom Core/GeomInterface \
	Core/Geometry Dataflow/Widgets Core/XMLUtil \
	Core/Util \
	Packages/Uintah/Core/Grid        \
	Packages/Uintah/CCA/Ports       \
	Packages/Uintah/Core/ProblemSpec \
	Packages/Uintah/Core/Exceptions  \
	Packages/Uintah/Core/Datatypes

#Core/GLVolumeRenderer \
#Packages/Uintah/Dataflow \
#Packages/Kurt/Core/Geom \


LIBS := $(XML_LIBRARY)  $(GL_LIBRARY) $(M_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
KURT_MODULES := $(KURT_MODULES) $(LIBNAME)
endif


