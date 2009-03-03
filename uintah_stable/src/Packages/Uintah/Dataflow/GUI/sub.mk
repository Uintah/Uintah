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

SRCDIR := Packages/Uintah/Dataflow/GUI

SRCS := \
	$(SRCDIR)/CompareMMS.tcl \
	$(SRCDIR)/application.tcl \
	$(SRCDIR)/test.tcl \
	$(SRCDIR)/ScalarFieldAverage.tcl \
	$(SRCDIR)/SubFieldHistogram.tcl \
	$(SRCDIR)/FieldExtractor.tcl \
	$(SRCDIR)/ScalarFieldExtractor.tcl $(SRCDIR)/TimestepSelector.tcl \
	$(SRCDIR)/VectorFieldExtractor.tcl \
	$(SRCDIR)/TensorFieldExtractor.tcl \
	$(SRCDIR)/ParticleFieldExtractor.tcl \
	$(SRCDIR)/RescaleColorMapForParticles.tcl $(SRCDIR)/ParticleVis.tcl \
	$(SRCDIR)/NodeHedgehog.tcl \
	$(SRCDIR)/ArchiveReader.tcl \
	$(SRCDIR)/GridVisualizer.tcl\
	$(SRCDIR)/PatchVisualizer.tcl\
	$(SRCDIR)/PatchDataVisualizer.tcl\
	$(SRCDIR)/FaceCuttingPlane.tcl\
	$(SRCDIR)/Hedgehog.tcl\
	$(SRCDIR)/ScalarOperator.tcl\
	$(SRCDIR)/ScalarFieldOperator.tcl\
	$(SRCDIR)/ScalarFieldBinaryOperator.tcl\
	$(SRCDIR)/ScalarMinMax.tcl \
	$(SRCDIR)/ScalarFieldNormalize.tcl \
	$(SRCDIR)/Schlieren.tcl \
	$(SRCDIR)/TensorOperator.tcl\
	$(SRCDIR)/TensorFieldOperator.tcl\
	$(SRCDIR)/TensorParticlesOperator.tcl\
	$(SRCDIR)/VectorOperator.tcl\
	$(SRCDIR)/VectorFieldOperator.tcl\
	$(SRCDIR)/VectorParticlesOperator.tcl\
	$(SRCDIR)/AnimatedStreams.tcl\
	$(SRCDIR)/VariablePlotter.tcl\
	$(SRCDIR)/TensorToTensorConvertor.tcl\
	$(SRCDIR)/UTextureBuilder.tcl\
	$(SRCDIR)/UdaScale.tcl\
#[INSERT NEW TCL FILE HERE]
#	$(SRCDIR)/EigenEvaluator.tcl\
#	$(SRCDIR)/ParticleEigenEvaluator.tcl\

include $(SCIRUN_SCRIPTS)/tclIndex.mk

UINTAH_SCIRUN := $(UINTAH_SCIRUN) $(TCLINDEX)
