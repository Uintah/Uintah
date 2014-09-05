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

SRCDIR   := Packages/Ptolemy/Core/jni

SRCS     += \
            $(SRCDIR)/JNIUtil.cc \
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Core/Persistent \
           Core/Containers Core/Util \
           Core/Exceptions Core/Thread \
           Core/GuiInterface

# JAVA_PATH set by configure
JAVA_LIB := $(JAVA_PATH)/jre/lib/i386
JAVA_INC := $(JAVA_PATH)/include

LIBS := \
         -L$(JAVA_LIB) \
         -L$(JAVA_LIB)/server \
         -ljava \
         -ljvm
##-L$(JAVA_LIB)/client
##-L$(JAVA_LIB)/native_threads \
##-lverify \
##-lhpi
##-lawt
##-ljawt

LDFLAGS += \
           -Wl,-rpath -Wl,$(JAVA_LIB) \
           -Wl,-rpath -Wl,$(JAVA_LIB)/server
#-Wl,-rpath -Wl,$(JAVA_LIB)/client
#-Wl,-rpath -Wl,$(JAVA_LIB)/native_threads

INCLUDES += \
            -I$(JAVA_INC) \
            -I$(JAVA_INC)/linux

CFLAGS += -MD

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

