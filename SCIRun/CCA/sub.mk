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

SRCDIR   := SCIRun/CCA

SRCS     += \
	$(SRCDIR)/CCAComponentDescription.cc \
	$(SRCDIR)/CCAComponentModel.cc \
	$(SRCDIR)/CCAComponentInstance.cc \
	$(SRCDIR)/CCAPortDescription.cc \
	$(SRCDIR)/CCAPortInstance.cc \
	$(SRCDIR)/ComponentID.cc \
	$(SRCDIR)/CCAException.cc \
	$(SRCDIR)/ConnectionID.cc
$(SRCDIR)/CCAComponentModel.o: Core/CCA/spec/cca_sidl.h
$(SRCDIR)/CCAComponentDescription.o: Core/CCA/spec/cca_sidl.h
$(SRCDIR)/CCAComponentInstance.o: Core/CCA/spec/cca_sidl.h
$(SRCDIR)/CCAPortInstance.o: Core/CCA/spec/cca_sidl.h
$(SRCDIR)/ComponentID.o: Core/CCA/spec/cca_sidl.h
$(SRCDIR)/ConnectionID.o: Core/CCA/spec/cca_sidl.h

