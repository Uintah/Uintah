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

# The SRCDIR_STACK maintains the current state of the recursion.
# This is used to reset SRCDIR after the include below is processed.
SUBDIRS := $(patsubst %,$(SRCTOP)/%,$(SUBDIRS))
SRCDIR_STACK := $(SRCDIR) $(SRCDIR_STACK)
ALLSUBDIRS := $(ALLSUBDIRS) $(SUBDIRS)

TMPSUBDIRS := $(SUBDIRS)
SUBDIRS := SET SUBDIRS BEFORE CALLING RECURSE
include $(patsubst %,%/sub.mk,$(TMPSUBDIRS))

SRCDIR := $(firstword $(SRCDIR_STACK))
SRCDIR_STACK := $(wordlist 2,$(words $(SRCDIR_STACK)),$(SRCDIR_STACK))

