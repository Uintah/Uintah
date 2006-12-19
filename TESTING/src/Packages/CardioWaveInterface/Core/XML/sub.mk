#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  
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

SRCDIR   := Packages/CardioWaveInterface/Core/XML

SRCS     += \
            $(SRCDIR)/MembraneXML.cc\
            $(SRCDIR)/SynapseXML.cc\
            $(SRCDIR)/OutputXML.cc\
            $(SRCDIR)/CWSolverXML.cc\
            $(SRCDIR)/NWSolverXML.cc\
            $(SRCDIR)/CWTimeStepXML.cc\
            $(SRCDIR)/NWTimeStepXML.cc\
            $(SRCDIR)/CardioWaveXML.cc\
            $(SRCDIR)/NeuroWaveXML.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Core/Persistent Core/Containers Core/Util \
           Core/XMLUtil

LIBS := $(XML2_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

