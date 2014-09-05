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

# Makefile fragment for this subdirectory

SRCDIR := Packages/Teem/StandAlone/convert

ifeq ($(LARGESOS),yes)
PSELIBS := Core Packages/Teem/Core
else
PSELIBS := Core/Datatypes Core/Containers Core/Persistent Core/Exceptions Core/Thread Core/Geometry Core/Math Packages/Teem/Core/Datatypes
endif
LIBS := $(TEEM_LIBRARY) -lell -lair -lbiff -lnrrd -lm

INCLUDES += $(TEEM_INCLUDE)

PROGRAM := $(SRCDIR)/MakeTensorsIsotropic
SRCS := $(SRCDIR)/MakeTensorsIsotropic.cc
include $(SCIRUN_SCRIPTS)/program.mk
