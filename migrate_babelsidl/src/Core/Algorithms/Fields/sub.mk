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

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Core/Algorithms/Fields


SRCS     += $(SRCDIR)/ApplyMappingMatrix.cc\
            $(SRCDIR)/ClearAndChangeFieldBasis.cc\
            $(SRCDIR)/ConvertToTetVol.cc\
            $(SRCDIR)/ConvertToTriSurf.cc\
            $(SRCDIR)/DomainBoundary.cc\
            $(SRCDIR)/ClipBySelectionMask.cc\
            $(SRCDIR)/DistanceField.cc\
            $(SRCDIR)/FieldsAlgo.cc\
            $(SRCDIR)/FieldBoundary.cc\
            $(SRCDIR)/FieldDataNodeToElem.cc\
            $(SRCDIR)/FieldDataElemToNode.cc\
            $(SRCDIR)/GetFieldData.cc\
            $(SRCDIR)/GetFieldDataMinMax.cc\
            $(SRCDIR)/GetFieldInfo.cc\
            $(SRCDIR)/IsInsideField.cc\
            $(SRCDIR)/LinkFieldBoundary.cc\
            $(SRCDIR)/LinkToCompGrid.cc\
            $(SRCDIR)/LinkToCompGridByDomain.cc\
            $(SRCDIR)/GatherFields.cc\
            $(SRCDIR)/MergeFields.cc\
            $(SRCDIR)/ScaleField.cc\
            $(SRCDIR)/SplitByConnectedRegion.cc\
            $(SRCDIR)/SplitFieldByDomain.cc\
            $(SRCDIR)/SetFieldData.cc\
            $(SRCDIR)/MappingMatrixToField.cc\
            $(SRCDIR)/TransformField.cc\
            $(SRCDIR)/ToPointCloud.cc\
            $(SRCDIR)/Unstructure.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS :=  Core/Datatypes Core/Util Core/Containers \
            Core/Exceptions Core/Thread Core/GuiInterface \
            Core/Geom Core/Geometry \
            Core/Algorithms/Converter \
            Core/Algorithms/Util \
            Core/Persistent \
            Core/Basis Core/Bundle

LIBS :=     $(TEEM_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
