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

SRCDIR   := Core/Math

FNSRCDIR  := $(SRCTOP)/$(SRCDIR)

# TARGDIR is not really 'srcdir'... however, it is given the 
# same value... but since it is used based on the top of the compilation
# tree, it has the "same value".
TARGDIR := $(SRCDIR)

$(TARGDIR)/fnparser.cc $(TARGDIR)/fnparser.h:	$(FNSRCDIR)/fnparser.y
	$(YACC) -o $(TARGDIR)/y.tab.c -p fn $(FNSRCDIR)/fnparser.y
	mv -f $(TARGDIR)/y.tab.h $(TARGDIR)/fnparser.h
	mv -f $(TARGDIR)/y.tab.c $(TARGDIR)/fnparser.cc

$(TARGDIR)/fnscanner.cc: $(FNSRCDIR)/fnscanner.l $(TARGDIR)/fnparser.cc
	$(LEX) -Pfn -o$(TARGDIR)/fnscanner.cc $(FNSRCDIR)/fnscanner.l

SRCS     += $(SRCDIR)/CubicPWI.cc              \
            $(SRCDIR)/Gaussian.cc	       \
            $(SRCDIR)/LinAlg.c		       \
            $(SRCDIR)/LinearPWI.cc	       \
            $(SRCDIR)/Mat.c		       \
            $(SRCDIR)/MiscMath.cc	       \
            $(SRCDIR)/MusilRNG.cc	       \
            $(SRCDIR)/PiecewiseInterp.cc       \
            $(SRCDIR)/TrigTable.cc	       \
            $(SRCDIR)/sci_lapack.cc	       \
            $(SRCDIR)/fft.c		       \
            $(SRCDIR)/ssmult.c

PSELIBS := Core/Exceptions Core/Containers
LIBS := $(M_LIBRARY) $(DL_LIBRARY) $(LAPACK_LIBRARY) $(BLAS_LIBRARY) $(F_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
