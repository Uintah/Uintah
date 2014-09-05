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

include $(SRCTOP)/scripts/largeso_prologue.mk

SRCDIR := Core/CCA

SUBDIRS := 

ifeq ($(BUILD_PARALLEL),yes)
SUBDIRS := \
	$(SRCDIR)/tools \
	$(SRCDIR)/Component
endif

include $(SRCTOP)/scripts/recurse.mk

PSELIBS := 

include $(SRCTOP)/scripts/largeso_epilogue.mk

