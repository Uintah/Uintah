# The contents of this file are subject to the University of Utah Public
# License (the "License"); you may not use this file except in compliance
# with the License.
# 
# Software distributed under the License is distributed on an "AS IS"
# basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
# License for the specific language governing rights and limitations under
# the License.
# 
# The Original Source Code is SCIRun, released March 12, 2001.
# 
# The Original Source Code was developed by the University of Utah.
# Portions created by UNIVERSITY are Copyright (C) 2001, 1994
# University of Utah. All Rights Reserved.
# 
#   File   : sub.mk<2>
#   Author : Martin Cole
#   Date   : Wed Jun 20 17:45:24 2001

# Makefile fragment for this subdirectory

SRCDIR := Packages/BioPSE/StandAlone/convert

ifeq ($(LARGESOS),yes)
PSELIBS := Core
else
PSELIBS := Core/Datatypes Core/Containers Core/Persistent Core/Exceptions Core/Thread Core/Geometry Core/Util
endif
LIBS := $(XML_LIBRARY) $(M_LIBRARY)

PROGRAM := $(SRCDIR)/ContinuityToTetVolDouble
SRCS := $(SRCDIR)/ContinuityToTetVolDouble.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/EGItoMat
SRCS := $(SRCDIR)/EGItoMat.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/ElectrodeToCurve
SRCS := $(SRCDIR)/ElectrodeToCurve.cc
include $(SCIRUN_SCRIPTS)/program.mk

PROGRAM := $(SRCDIR)/TextToCurve
SRCS := $(SRCDIR)/TextToCurve.cc
include $(SCIRUN_SCRIPTS)/program.mk
