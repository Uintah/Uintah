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

SRCDIR   := Core/Geometry

SRCS     += $(SRCDIR)/BBox.cc \
	    $(SRCDIR)/Grid.cc \
	    $(SRCDIR)/IntVector.cc \
	    $(SRCDIR)/Plane.cc \
	    $(SRCDIR)/Point.cc \
	    $(SRCDIR)/Polygon.cc \
	    $(SRCDIR)/Quaternion.cc \
	    $(SRCDIR)/Ray.cc \
	    $(SRCDIR)/Tensor.cc \
	    $(SRCDIR)/Transform.cc \
	    $(SRCDIR)/Vector.cc

PSELIBS := Core/Persistent Core/Containers \
	   Core/Exceptions Core/Util
LIBS := $(DEFAULT_LIBRARY) $(M_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
