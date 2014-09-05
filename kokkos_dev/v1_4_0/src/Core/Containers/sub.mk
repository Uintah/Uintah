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

SRCDIR   := Core/Containers

SRCS     += $(SRCDIR)/ConsecutiveRangeSet.cc \
	    $(SRCDIR)/Sort.cc \
	    $(SRCDIR)/StringUtil.cc \
	    $(SRCDIR)/TrivialAllocator.cc \
	    $(SRCDIR)/templates.cc

PSELIBS := Core/Exceptions Core/Tester Core/Thread
LIBS :=

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

