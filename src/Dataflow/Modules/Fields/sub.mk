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

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Dataflow/Modules/Fields

SRCS     += \
	$(SRCDIR)/ApplyMappingMatrix.cc\
	$(SRCDIR)/AttractNormals.cc\
	$(SRCDIR)/BuildMappingMatrix.cc\
	$(SRCDIR)/CastMLVtoHV.cc\
	$(SRCDIR)/CastTVtoMLV.cc\
	$(SRCDIR)/Centroids.cc\
	$(SRCDIR)/ChangeCoordinates.cc\
	$(SRCDIR)/ChangeFieldBasis.cc\
	$(SRCDIR)/ChangeFieldDataType.cc\
	$(SRCDIR)/ChangeFieldBounds.cc\
	$(SRCDIR)/ChooseField.cc\
	$(SRCDIR)/ClipByFunction.cc\
	$(SRCDIR)/ClipField.cc\
	$(SRCDIR)/ClipLattice.cc\
	$(SRCDIR)/ConvertTet.cc\
	$(SRCDIR)/Coregister.cc\
	$(SRCDIR)/CreateMesh.cc\
	$(SRCDIR)/CubitInterface.cc\
	$(SRCDIR)/DirectMapping.cc\
	$(SRCDIR)/FieldBoundary.cc\
	$(SRCDIR)/FieldCage.cc\
	$(SRCDIR)/FieldFrenet.cc\
	$(SRCDIR)/FieldInfo.cc\
	$(SRCDIR)/FieldMeasures.cc\
	$(SRCDIR)/FieldSlicer.cc\
	$(SRCDIR)/FieldSubSample.cc\
	$(SRCDIR)/GatherFields.cc\
	$(SRCDIR)/Gradient.cc\
	$(SRCDIR)/HexToTet.cc\
	$(SRCDIR)/IsoClip.cc\
	$(SRCDIR)/ManageFieldData.cc\
	$(SRCDIR)/ManageFieldMesh.cc\
	$(SRCDIR)/MapDataToMeshCoord.cc\
	$(SRCDIR)/MaskLattice.cc\
	$(SRCDIR)/MaskLatVolWithTriSurf.cc\
	$(SRCDIR)/MoveElemToNode.cc\
	$(SRCDIR)/MoveNodeToElem.cc\
	$(SRCDIR)/NodeGradient.cc\
	$(SRCDIR)/QuadToTri.cc\
	$(SRCDIR)/PlanarTransformField.cc\
	$(SRCDIR)/PointLatticeMap.cc\
	$(SRCDIR)/Probe.cc\
	$(SRCDIR)/RefineTetVol.cc\
	$(SRCDIR)/ReplaceScalarDataValue.cc\
	$(SRCDIR)/SampleField.cc\
	$(SRCDIR)/SampleLattice.cc\
	$(SRCDIR)/SamplePlane.cc\
	$(SRCDIR)/SampleStructHex.cc\
	$(SRCDIR)/ScalarFieldStats.cc\
	$(SRCDIR)/ScaleFieldData.cc\
	$(SRCDIR)/SelectField.cc\
	$(SRCDIR)/SetProperty.cc\
	$(SRCDIR)/TetVol2QuadraticTetVol.cc\
	$(SRCDIR)/ToStructured.cc\
	$(SRCDIR)/TransformField.cc\
	$(SRCDIR)/TransformData.cc\
	$(SRCDIR)/TransformData2.cc\
	$(SRCDIR)/TransformData3.cc\
	$(SRCDIR)/TransformMesh.cc\
	$(SRCDIR)/Unstructure.cc\
	$(SRCDIR)/VectorMagnitude.cc\
#[INSERT NEW CODE FILE HERE]


PSELIBS := Dataflow/Network Dataflow/Ports Dataflow/XMLUtil Dataflow/Widgets \
	Core/Datatypes Core/Persistent Core/Exceptions Core/ImportExport \
	Core/Thread Core/Containers Core/GuiInterface Core/Geom \
	Core/Datatypes Core/Geometry Core/TkExtensions Core/ImportExport \
	Core/Math Core/Util Core/Algorithms/Geometry Core/GeomInterface
LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(LEX_LIBRARY) $(M_LIBRARY) $(XML_LIBRARY) $(THREAD_LIBRARY)

# Sandia Meshing Library
ifeq ($(HAVE_CAMAL),yes)
   SRCS += $(SRCDIR)/TetMesher.cc
   LIBS := $(LIBS) $(CAMAL_LIBRARY) $(F_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
SCIRUN_MODULES := $(SCIRUN_MODULES) $(LIBNAME)
endif

