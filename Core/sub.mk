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

include $(SCIRUN_SCRIPTS)/largeso_prologue.mk

SRCDIR := Core

SUBDIRS := \
	$(SRCDIR)/Algorithms \
	$(SRCDIR)/Containers \
	$(SRCDIR)/Datatypes \
	$(SRCDIR)/Exceptions \
	$(SRCDIR)/GUI \
	$(SRCDIR)/Geom \
	$(SRCDIR)/GeomInterface \
	$(SRCDIR)/Geometry \
	$(SRCDIR)/GLVolumeRenderer \
	$(SRCDIR)/GuiInterface \
	$(SRCDIR)/Malloc \
	$(SRCDIR)/Math \
	$(SRCDIR)/OS \
	$(SRCDIR)/Persistent \
	$(SRCDIR)/Process \
	$(SRCDIR)/Thread \
	$(SRCDIR)/TkExtensions \
	$(SRCDIR)/Util \
	$(SRCDIR)/2d \
#	$(SRCDIR)/Parts \
#	$(SRCDIR)/PartsGui


ifeq ($(BUILD_SCIRUN2),yes)
SUBDIRS := \
	$(SUBDIRS) \
	$(SRCDIR)/CCA \
	$(SRCDIR)/Babel 
endif

ifeq ($(HAVE_GLOBUS),yes)
SUBDIRS+=$(SRCDIR)/globus_threads
endif

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := 
LIBS := $(PLPLOT_LIBRARY) $(BLT_LIBRARY) $(ITCL_LIBRARY) $(TCL_LIBRARY) $(TK_LIBRARY) \
	$(ITK_LIBRARY) $(GL_LIBRARY) $(THREAD_LIBRARY) \
	$(Z_LIBRARY) $(M_LIBRARY) 

include $(SCIRUN_SCRIPTS)/largeso_epilogue.mk

