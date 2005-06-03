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


# *** NOTE ***
#
# Do not remove or modify the comment line:
#
# #[INSERT NEW ?????? HERE]
#
# It is required by the Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Component"
# documentation on how to do it correctly.

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Packages/Ptolemy/Core/Comm

SRCS     += \
            $(SRCDIR)/PtolemyServer.cc \
            $(SRCDIR)/NetworkHelper.cc
	         
#[INSERT NEW CODE FILE HERE]

ifeq ($(LARGESOS),yes)
    PSELIBS := Dataflow Core
else
    PSELIBS := Dataflow/Network Core/Containers Core/GuiInterface \
               Core/Thread Core/Exceptions Core/Util Core/TkExtensions Core/Comm \
               Core/Malloc Core/Services Dataflow/XMLUtil Core/SystemCall \
               Dataflow/Modules/Render \
                   Packages/Ptolemy/Core/PtolemyInterface \

endif
   
LIBS :=  $(LIBS) $(XML_LIBRARY) $(THREAD_LIBRARY) $(SEMAPHORE_LIBRARY)

SOFLAGS := \
            -shared -L$(LIBDIR) $(CFLAGS)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

