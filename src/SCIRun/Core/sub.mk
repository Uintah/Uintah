#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


# Makefile fragment for this subdirectory

SRCDIR   := SCIRun/Core

SRCS     += \
	     $(SRCDIR)/CCAException.cc \
	     $(SRCDIR)/TypeMapImpl.cc \
	     $(SRCDIR)/ConnectionIDImpl.cc \
             $(SRCDIR)/ConnectionInfoImpl.cc \
             $(SRCDIR)/PortInfoImpl.cc \
             $(SRCDIR)/ComponentInfoImpl.cc \
	     $(SRCDIR)/FixedPortServiceFactoryImpl.cc \
	     $(SRCDIR)/ProviderServiceFactoryImpl.cc \
	     $(SRCDIR)/ServiceInfoImpl.cc \
             $(SRCDIR)/ComponentClassDescriptionImpl.cc \
             $(SRCDIR)/ComponentClassFactoryImpl.cc \
             $(SRCDIR)/UnknownComponentClassFactory.cc \
	     $(SRCDIR)/CoreServicesImpl.cc \
	     $(SRCDIR)/CoreFrameworkImpl.cc \
	     $(SRCDIR)/BuilderServiceImpl.cc \
#             $(SRCDIR)/CoreFrameworkException.cc \
#             $(SRCDIR)/ComponentEvent.cc \
#             $(SRCDIR)/ConnectionEvent.cc \
#             $(SRCDIR)/FrameworkServiceFactory.cc \
#             $(SRCDIR)/ComponentEventService.cc \
#             $(SRCDIR)/ConnectionEventService.cc \
#             $(SRCDIR)/ComponentRepositoryService.cc \
#             $(SRCDIR)/FrameworkPropertiesService.cc \

$(SRCDIR)/TypeMap.o: Core/CCA/spec/sci_sidl.h
$(SRCDIR)/ConnectionID.o: Core/CCA/spec/sci_sidl.h
$(SRCDIR)/FrameworkID.o: Core/CCA/spec/sci_sidl.h 
$(SRCDIR)/ConnectionInfo.o: Core/CCA/spec/sci_sidl.h 
$(SRCDIR)/ComponentInfo.o: Core/CCA/spec/sci_sidl.h 
$(SRCDIR)/PortInfo.o: Core/CCA/spec/sci_sidl.h 
$(SRCDIR)/CoreFramework.o: Core/CCA/spec/sci_sidl.h 
$(SRCDIR)/CoreFrameworkException.o: Core/CCA/spec/sci_sidl.h 
$(SRCDIR)/ComponentEvent.o: Core/CCA/spec/sci_sidl.h 
$(SRCDIR)/ConnectionEvent.o: Core/CCA/spec/sci_sidl.h 
$(SRCDIR)/FrameworkServiceFactory.o: Core/CCA/spec/sci_sidl.h 
$(SRCDIR)/ComponentClassDescription.o: Core/CCA/spec/sci_sidl.h 
$(SRCDIR)/ComponentEventService.o: Core/CCA/spec/sci_sidl.h 
$(SRCDIR)/ConnectionEventService.o: Core/CCA/spec/sci_sidl.h 
$(SRCDIR)/ComponentRepositoryService.o: Core/CCA/spec/sci_sidl.h 
$(SRCDIR)/FrameworkPropertiesService.o: Core/CCA/spec/sci_sidl.h 
$(SRCDIR)/BuilderService.o: Core/CCA/spec/sci_sidl.h \


 PSELIBS := Core/OS Core/Containers Core/Util Core/XMLUtil \
	    Core/GuiInterface Core/CCA/spec \
            Core/CCA/PIDL Core/CCA/SSIDL \
            Core/Exceptions Core/Thread \
            Core/TkExtensions Core/Init Core/CCA/Comm

LIBS := $(XML_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
