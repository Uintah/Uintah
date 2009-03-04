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

SRCDIR   := Packages/Uintah/Dataflow/Modules/Operators

SRCS     += \
	$(SRCDIR)/CompareMMS.cc \
	$(SRCDIR)/EigenEvaluator.cc \
	$(SRCDIR)/ParticleEigenEvaluator.cc \
	$(SRCDIR)/ScalarFieldAverage.cc \
	$(SRCDIR)/ScalarFieldBinaryOperator.cc \
	$(SRCDIR)/ScalarFieldNormalize.cc \
	$(SRCDIR)/ScalarFieldOperator.cc \
	$(SRCDIR)/ScalarMinMax.cc \
	$(SRCDIR)/Schlieren.cc \
	$(SRCDIR)/TensorFieldOperator.cc \
	$(SRCDIR)/TensorParticlesOperator.cc \
	$(SRCDIR)/TensorToTensorConvertor.cc \
	$(SRCDIR)/VectorFieldOperator.cc \
	$(SRCDIR)/VectorParticlesOperator.cc \
	$(SRCDIR)/UdaScale.cc\
#[INSERT NEW CODE FILE HERE]

SUBDIRS := $(SRCDIR)/MMS
include $(SCIRUN_SCRIPTS)/recurse.mk          

PSELIBS := \
	Packages/Uintah/Core/Datatypes     \
	Packages/Uintah/Core/DataArchive   \
	Packages/Uintah/Core/Disclosure    \
	Packages/Uintah/CCA/Ports          \
	Packages/Uintah/Core/Grid          \
	Packages/Uintah/Core/Math          \
	Packages/Uintah/Core/Parallel      \
	Packages/Uintah/Core/Util          \
	Packages/Uintah/Core/ProblemSpec   \
	Packages/Uintah/Core/Exceptions    \
	Dataflow/Network   \
	Core/Basis         \
	Core/Containers    \
	Core/Persistent    \
	Core/Exceptions    \
	Core/GuiInterface  \
	Core/Thread        \
	Core/Datatypes     \
	Core/Geom          \
	Core/Geometry      \
	Core/GeomInterface \
	Core/Util 

LIBS := $(XML2_LIBRARY) $(M_LIBRARY) $(MPI_LIBRARY)

include $(SRCTOP)/scripts/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
UINTAH_SCIRUN := $(UINTAH_SCIRUN) $(LIBNAME)
endif

