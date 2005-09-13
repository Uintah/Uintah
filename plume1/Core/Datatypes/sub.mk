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

SRCDIR   := Core/Datatypes

SRCS +=	$(SRCDIR)/Clipper.cc		    	\
	$(SRCDIR)/Color.cc		    	\
	$(SRCDIR)/ColumnMatrix.cc	    	\
	$(SRCDIR)/CurveMesh.cc            	\
	$(SRCDIR)/Datatype.cc		    	\
	$(SRCDIR)/DenseColMajMatrix.cc	    	\
	$(SRCDIR)/DenseMatrix.cc	    	\
	$(SRCDIR)/Field.cc		    	\
	$(SRCDIR)/FieldInterfaceAux.cc	    	\
	$(SRCDIR)/HexVolMesh.cc			\
	$(SRCDIR)/Image.cc		    	\
	$(SRCDIR)/ImageMesh.cc		    	\
	$(SRCDIR)/LatVolMesh.cc 		\
	$(SRCDIR)/MaskedLatVolMesh.cc	    	\
	$(SRCDIR)/Matrix.cc		    	\
	$(SRCDIR)/MatrixOperations.cc	    	\
	$(SRCDIR)/Mesh.cc		    	\
	$(SRCDIR)/NrrdData.cc		    	\
	$(SRCDIR)/NrrdString.cc		    	\
	$(SRCDIR)/NrrdScalar.cc		    	\
	$(SRCDIR)/PointCloudMesh.cc         	\
	$(SRCDIR)/PropertyManager.cc	    	\
	$(SRCDIR)/PrismVolMesh.cc	    	\
	$(SRCDIR)/QuadSurfMesh.cc               \
	$(SRCDIR)/QuadraticTetVolMesh.cc        \
	$(SRCDIR)/QuadraticLatVolMesh.cc        \
	$(SRCDIR)/ScanlineMesh.cc           	\
	$(SRCDIR)/SearchGrid.cc           	\
	$(SRCDIR)/SparseRowMatrix.cc	    	\
        $(SRCDIR)/String.cc                     \
	$(SRCDIR)/StructCurveMesh.cc	    	\
	$(SRCDIR)/StructQuadSurfMesh.cc	    	\
	$(SRCDIR)/StructHexVolMesh.cc	    	\
	$(SRCDIR)/TetVolMesh.cc 	    	\
	$(SRCDIR)/TriSurfMesh.cc	    	\
	$(SRCDIR)/TypeName.cc		    	\
	$(SRCDIR)/cd_templates.cc	    	\
	$(SRCDIR)/cd_templates_fields_0.cc    	\
	$(SRCDIR)/cd_templates_fields_1.cc    	\
	$(SRCDIR)/cd_templates_fields_2.cc    	\
	$(SRCDIR)/cd_templates_fields_3.cc    	\
	$(SRCDIR)/cd_templates_fields_4.cc    	\
	$(SRCDIR)/cd_templates_fields_5.cc	\
	$(SRCDIR)/cd_templates_fields_6.cc	\
	$(SRCDIR)/cd_templates_fields_7.cc

PSELIBS := Core/Persistent Core/Exceptions Core/Containers \
	Core/Thread Core/Geometry Core/GuiInterface \
	Core/Math Core/Util 
LIBS := $(GL_LIBRARY) $(M_LIBRARY) $(BLAS_LIBRARY) $(F_LIBRARY) \
	$(TEEM_LIBRARY) $(PNG_LIBRARY) $(Z_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

INCLUDES += $(TEEM_INCLUDE)
INCLUDES += $(BLAS_INCLUDE)

