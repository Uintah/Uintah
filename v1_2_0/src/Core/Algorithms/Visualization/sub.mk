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

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Core/Algorithms/Visualization

SRCS     += \
        $(SRCDIR)/HexMC.cc	\
	$(SRCDIR)/MarchingCubes.cc	\
	$(SRCDIR)/Noise.cc		\
	$(SRCDIR)/RenderField.cc	\
	$(SRCDIR)/Sage.cc		\
	$(SRCDIR)/TetMC.cc	\



PSELIBS := Core/Datatypes Core/Containers Core/Thread \
	Core/Exceptions Core/Geom Core/Disclosure Core/Geometry

LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
