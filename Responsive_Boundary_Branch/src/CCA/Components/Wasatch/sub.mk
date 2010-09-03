# 
# 
# The MIT License
# 
# Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := CCA/Components/Wasatch

SRCS     +=                             \
        $(SRCDIR)/GraphHelperTools.cc	\
        $(SRCDIR)/ParseTools.cc		\
	$(SRCDIR)/Properties.cc		\
	$(SRCDIR)/StringNames.cc	\
        $(SRCDIR)/TaskInterface.cc	\
	$(SRCDIR)/TimeStepper.cc	\
	$(SRCDIR)/Wasatch.cc

PSELIBS :=              \
	Core/Geometry   \
	Core/Util       \
	Core/Exceptions   \
	CCA/Ports         \
	Core/Grid         \
	Core/Util         \
	Core/ProblemSpec  \
	Core/GeometryPiece  \
	Core/Exceptions   \
	Core/Disclosure   \
	Core/Math         \
	Core/Parallel

#------------------------------------------------------------------------
#jcs these are currently hacked in and work only for my Mac laptop.
#    eventually we need to get them working from the configure script.
SPATIALOPS_INCLUDE := -I/jcs/apps/include
SPATIALOPS_LIBRARY := /jcs/apps/lib/spatialops/libspatialops.a

EXPRLIB_INCLUDE    := -I/Users/u0033394/projects/ExprLib/buildUintah/install_dir/include
EXPRLIB_LIBRARY    := /Users/u0033394/projects/ExprLib/buildUintah/install_dir/lib/expression/libexprlib.a

BOOST_INCLUDE      := -I/jcs/apps/include/boost-1.41.0
BOOST_LIBRARY      := /jcs/apps/lib/boost-1.41.0/libboost_thread-mt.a /jcs/apps/lib/boost-1.41.0/libboost_signals-mt.a
#------------------------------------------------------------------------

LIBS := $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY) 	\
        $(EXPRLIB_LIBRARY) $(SPATIALOPS_LIBRARY) 	\
        $(TABPROPS_LIBRARY) $(HDF5_LIBRARY)		\
        $(BOOST_LIBRARY)

INCLUDES := $(INCLUDES) $(SPATIALOPS_INCLUDE) $(EXPRLIB_INCLUDE) \
            $(HDF5_INCLUDE) $(TABPROPS_INCLUDE) $(BOOST_INCLUDE)

SUBDIRS := \
        $(SRCDIR)/Operators	\
        $(SRCDIR)/Expressions	\
	$(SRCDIR)/transport

include $(SCIRUN_SCRIPTS)/recurse.mk

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
