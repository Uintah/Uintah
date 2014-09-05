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

SRCDIR   := SCIRun

SRCS     += \
	$(SRCDIR)/SCIRunFramework.cc \
	$(SRCDIR)/ComponentDescription.cc \
	$(SRCDIR)/ComponentInstance.cc \
	$(SRCDIR)/ComponentModel.cc \
	$(SRCDIR)/PortDescription.cc \
	$(SRCDIR)/SCIRunErrorHandler.cc \
	$(SRCDIR)/PortInstance.cc \
	$(SRCDIR)/PortInstanceIterator.cc\
	$(SRCDIR)/CCACommunicator.cc

SUBDIRS := $(SRCDIR)/CCA $(SRCDIR)/Dataflow $(SRCDIR)/Internal $(SRCDIR)/Babel

include $(SCIRUN_SCRIPTS)/recurse.mk

PSELIBS := Core/OS Core/Containers Core/Util Dataflow/XMLUtil \
	Dataflow/Network Core/GuiInterface Core/CCA/spec \
	Core/CCA/Component/PIDL Core/CCA/Component/CIA \
	Core/Exceptions Core/TkExtensions Core/Thread

LIBS := $(XML_LIBRARY) -L${BABELDIR}/lib -lsidl

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

$(SRCDIR)/BuilderService.o: Core/CCA/spec/cca_sidl.h
$(SRCDIR)/SCIRunFramework.o: Core/CCA/spec/cca_sidl.h
$(SRCDIR)/Services.o: Core/CCA/spec/cca_sidl.h
$(SRCDIR)/ComponentInstance.o: Core/CCA/spec/cca_sidl.h
$(SRCDIR)/CCACommunicator.o: Core/CCA/spec/cca_sidl.h
