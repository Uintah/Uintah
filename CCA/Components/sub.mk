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

SRCDIR   := CCA/Components



ifeq ($(HAVE_QT),yes)
SUBDIRS := $(SRCDIR)/Builder $(SRCDIR)/TxtBuilder $(SRCDIR)/Hello  $(SRCDIR)/ListPlotter \
	$(SRCDIR)/ZList $(SRCDIR)/Viewer $(SRCDIR)/LinSolver \
	$(SRCDIR)/FileReader $(SRCDIR)/FEM $(SRCDIR)/Tri $(SRCDIR)/TableTennis \
	$(SRCDIR)/TTClient $(SRCDIR)/World 

else
SUBDIRS :=$(SRCDIR)/TxtBuilder $(SRCDIR)/Hello
endif


ifeq ($(HAVE_MPI),yes)
SUBDIRS := $(SUBDIRS) $(SRCDIR)/PWorld $(SRCDIR)/PHello $(SRCDIR)/PLinSolver
endif

#ifeq ($(HAVE_BABEL),yes)
#SUBDIRS:= $(SUBDIRS) $(SRCDIR)/BabelTest
#endif

SUBDIRS := $(SRCDIR)/Builder  $(SRCDIR)/Viewer $(SRCDIR)/LinSolver \
	$(SRCDIR)/FileReader $(SRCDIR)/FEM $(SRCDIR)/Tri $(SRCDIR)/PLinSolver


include $(SCIRUN_SCRIPTS)/recurse.mk
