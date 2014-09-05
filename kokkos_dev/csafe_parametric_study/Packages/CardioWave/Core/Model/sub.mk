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
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/CardioWave/Core/Model

SRCS     += \
            $(SRCDIR)/ModelAlgo.cc\
            $(SRCDIR)/BuildMembraneTable.cc\
            $(SRCDIR)/BuildStimulusTable.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Dataflow/Network \
        Core/Persistent Core/Containers Core/Util \
        Core/Exceptions Core/Thread Core/GuiInterface \
        Core/Geom Core/Datatypes Core/Geometry \
        Core/GeomInterface Core/Bundle \
        Core/Algorithms/Util \
        Core/Algorithms/Fields \
        Core/Algorithms/Converter \
        Core/Algorithms/DataIO \
        Core/Algorithms/Math \
                
LIBS :=

ifeq ($(HAVE_INSIGHT),yes)
  LIBS += $(INSIGHT_LIBRARY) $(GDCM_LIBRARY) $(TK_LIBRARY) $(GL_LIBRARY) 
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

