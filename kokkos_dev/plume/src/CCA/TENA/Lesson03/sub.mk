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

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := CCA/TENA/Lesson03

INCLUDES += $(TENA_INCLUDES) $(TENA_DEFINES) -I$(TENA_OM_DEF_DIR)/Lesson_03-v2 -I$(TENA_OM_IMPL_DIR)/Lesson_03-v2

SRCS     += $(SRCDIR)/PublishPerson.cc \
	$(SRCDIR)/SubscribePerson.cc \
	$(SRCDIR)/Lesson03.cc

PSELIBS := Core/CCA/SSIDL Core/CCA/PIDL Core/CCA/Comm \
           Core/CCA/spec Core/Thread Core/Containers Core/Exceptions

LIBS := $(TENA_LIBRARY) -lLesson_03-v2-$(TENA_PLATFORM)-v5.1_0 -L/opt/OM/impl/TENA/lib -lLesson_03-v2-basic-v1-$(TENA_PLATFORM)-v5.1

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

#include $(SCIRUN_SCRIPTS)/program.mk

$(SRCDIR)/TENAService.o: Core/CCA/spec/sci_sidl.h
