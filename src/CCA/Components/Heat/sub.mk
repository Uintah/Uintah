#
#  The MIT License
#
#  Copyright (c) 1997-2017 The University of Utah
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
# Makefile fragment for this subdirectory /

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := CCA/Components/Heat

SRCS +=                            \
         $(SRCDIR)/CCHeat2D.cc     \
         $(SRCDIR)/NCHeat2D.cc     \
         $(SRCDIR)/CCHeat3D.cc     \
         $(SRCDIR)/NCHeat3D.cc     \
         $(SRCDIR)/AMRCCHeat2D.cc  \
         $(SRCDIR)/AMRNCHeat2D.cc  \
         $(SRCDIR)/AMRCCHeat3D.cc  \
         $(SRCDIR)/AMRNCHeat3D.cc  \

VTK_SRCS :=                        \
         $(SRCDIR)/vtkfile.cpp     \

PSELIBS :=                         \
        CCA/Components/Schedulers \
        CCA/Components/Models     \
        CCA/Ports                 \
        Core/Disclosure           \
        Core/Exceptions           \
        Core/Geometry             \
        Core/GeometryPiece        \
        Core/Grid                 \
        Core/IO                   \
        Core/Math                 \
        Core/Parallel             \
        Core/ProblemSpec          \
        Core/Util

LIBS :=                                                              \
#        $(Z_LIBRARY) $(XML2_LIBRARY) $(MPI_LIBRARY) $(M_LIBRARY)    \
#        $(EXPRLIB_LIBRARY) $(SPATIALOPS_LIBRARY)                    \
#        $(RADPROPS_LIBRARY) $(TABPROPS_LIBRARY)                     \
#        $(NSCBC_LIBRARY)                                            \
#        $(POKITT_LIBRARY)                                           \
#        $(BOOST_LIBRARY) $(LAPACK_LIBRARY) $(BLAS_LIBRARY)

VTK_LIBS :=                                                          \
         -L/home/jonmatteo/Developer/vtk/6.1.0/lib                   \
         -Wl,-rpath,/home/jonmatteo/Developer/vtk/6.1.0/lib          \
         -lvtksys-6.1 -lvtkzlib-6.1 -lvtkjsoncpp-6.1 -lvtkexpat-6.1  \
         -lvtkCommonCore-6.1 -lvtkCommonExecutionModel-6.1           \
         -lvtkCommonDataModel-6.1 -lvtkCommonMisc-6.1                \
         -lvtkCommonSystem-6.1 -lvtkCommonTransforms-6.1             \
         -lvtkCommonMath-6.1 -lvtkIOCore-6.1 -lvtkIOGeometry-6.1     \
         -lvtkIOXMLParser-6.1 -lvtkIOXML-6.1                         \
#        -lvtkpng-6.1 -lvtktiff-6.1 -lvtkmetaio-6.1                  \
#        -lvtkDICOMParser-6.1 -lvtkjpeg-6.1 -lvtkIOImage-6.1         \
#        -lvtkexpat-6.1                                              \

INCLUDES :=                                                          \
         $(INCLUDES)                                                 \
#        $(SPATIALOPS_INCLUDE) $(EXPRLIB_INCLUDE)                    \
#        $(TABPROPS_INCLUDE) $(RADPROPS_INCLUDE) $(NSCBC_INCLUDE)    \
#        $(POKITT_INCLUDE) $(BOOST_INCLUDE) $(LAPACK_INCLUDE)        \

VTK_INCLUDES :=                                                      \
         -I/home/jonmatteo/Developer/vtk/6.1.0/include/vtk-6.1       \

ifeq ($(HAVE_HEAD_VTK),yes)
   SRCS += $(VTK_SRCS)
   LIBS += $(VTK_LIBS)
   INCLUDES += $(VTK_INCLUDES)
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

