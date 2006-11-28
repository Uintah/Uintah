# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Dataflow/Modules/DataArrayMath

SRCS     += \
        $(SRCDIR)/CreateDataArray.cc\
        $(SRCDIR)/CalculateDataArray.cc\
        $(SRCDIR)/CreateTensorArray.cc\
        $(SRCDIR)/CreateVectorArray.cc\
        $(SRCDIR)/DecomposeTensorArrayIntoEigenVectors.cc\
        $(SRCDIR)/SplitVectorArrayInXYZ.cc\
        $(SRCDIR)/CreateDataArrayFromIndices.cc\
        $(SRCDIR)/ReplicateDataArray.cc\
        $(SRCDIR)/AppendDataArrays.cc\
        $(SRCDIR)/ReportDataArrayMeasure.cc\
        $(SRCDIR)/ReportDataArrayInfo.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Dataflow/Network \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Dataflow/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/GeomInterface Dataflow/TkExtensions \
        Core/Algorithms/ArrayMath 
        
        
LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY) $(TEEM_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


