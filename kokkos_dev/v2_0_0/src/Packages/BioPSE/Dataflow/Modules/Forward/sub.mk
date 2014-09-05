#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the Core/CCA/Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Core/CCA/Component"
# documentation on how to do it correctly.

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/BioPSE/Dataflow/Modules/Forward


SRCS     += $(SRCDIR)/AnisoSphereModel.cc\
	    $(SRCDIR)/ApplyFEMCurrentSource.cc\
	    $(SRCDIR)/ApplyFEMVoltageSource.cc\
	    $(SRCDIR)/BuildFEMatrixQuadratic.cc\
	    $(SRCDIR)/ComputeCurrent.cc\
	    $(SRCDIR)/ConfigureElectrode.cc\
	    $(SRCDIR)/ConfigureWireElectrode.cc\
	    $(SRCDIR)/CreateDisAnisoSpheres.cc\
	    $(SRCDIR)/DipoleInSphere.cc\
	    $(SRCDIR)/DipoleInAnisoSpheres.cc\
	    $(SRCDIR)/DipoleMatrixSourceRHSQuadratic.cc\
	    $(SRCDIR)/ForwardMagneticField.cc\
	    $(SRCDIR)/IndicesToTensors.cc\
	    $(SRCDIR)/InsertElectrodes.cc\
	    $(SRCDIR)/InsertVoltageSource.cc\
	    $(SRCDIR)/IntegrateCurrent.cc\
	    $(SRCDIR)/SetupBEMatrix.cc\
	    $(SRCDIR)/SetupFEMatrix.cc\
	    $(SRCDIR)/SphereModel.cc\
	    $(SRCDIR)/TensorsToIndices.cc\
#[INSERT NEW CODE FILE HERE]	


PSELIBS := Packages/BioPSE/Core/Datatypes \
	Packages/BioPSE/Core/Algorithms/Forward \
	Packages/BioPSE/Core/Algorithms/NumApproximation \
	Dataflow/Network Dataflow/Ports \
	Core/Datatypes Core/Persistent Core/Containers \
	Core/Util Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Geometry Core/GeomInterface Core/TkExtensions

LIBS := $(TK_LIBRARY) $(GL_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
