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

SRCDIR   := Dataflow/Network

SRCS     += \
	$(SRCDIR)/Connection.cc \
	$(SRCDIR)/Port.cc \
	$(SRCDIR)/Module.cc \
	$(SRCDIR)/ModuleHelper.cc \
	$(SRCDIR)/Network.cc \
	$(SRCDIR)/FileUtils.cc \
	$(SRCDIR)/Services.cc \
	$(SRCDIR)/Scheduler.cc \
#	$(SRCDIR)/GenFiles.cc\
#	$(SRCDIR)/ComponentNode.cc \
#	$(SRCDIR)/SkeletonFiles.cc\
#       $(SRCDIR)/PackageDBHandler.cc \
#	$(SRCDIR)/PackageDB.cc \


PSELIBS := Dataflow/Resources Dataflow/Comm Dataflow/XMLUtil \
	Core/Parts Core/Framework \
	Core/Exceptions Core/Thread \
	Core/Containers Core/GuiInterface Core/Util \
	Core/Geom 
LIBS := $(XML_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

