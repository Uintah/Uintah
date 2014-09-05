#
#  The MIT License
#
#  Copyright (c) 2010-2012 The University of Utah
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#  IN THE SOFTWARE.
# 
# 
# 
# 
# 
# Makefile fragment for this subdirectory 

SRCDIR   := CCA/Components/Wasatch/Expressions

SRCS     +=                         \
	$(SRCDIR)/BasicExprBuilder.cc		\
	$(SRCDIR)/ConvectiveFlux.cc	    	\
	$(SRCDIR)/DiffusiveFlux.cc	    	\
	$(SRCDIR)/DiffusiveVelocity.cc		\
	$(SRCDIR)/Dilatation.cc			\
	$(SRCDIR)/MomentumPartialRHS.cc 	\
	$(SRCDIR)/MomentumRHS.cc 		\
	$(SRCDIR)/MonolithicRHS.cc		\
	$(SRCDIR)/PrimVar.cc			\
	$(SRCDIR)/ScalarRHS.cc			\
	$(SRCDIR)/ScalabilityTestSrc.cc		\
	$(SRCDIR)/SetCurrentTime.cc		\
	$(SRCDIR)/Strain.cc 			\
	$(SRCDIR)/VelocityMagnitude.cc 		\
	$(SRCDIR)/Vorticity.cc 			\
	$(SRCDIR)/Pressure.cc     \
 $(SRCDIR)/PoissonExpression.cc

SUBDIRS := \
        $(SRCDIR)/MMS \
        $(SRCDIR)/PBE \
        $(SRCDIR)/PostProcessing \
        $(SRCDIR)/Turbulence  \
        $(SRCDIR)/EmbeddedGeometry

include $(SCIRUN_SCRIPTS)/recurse.mk
