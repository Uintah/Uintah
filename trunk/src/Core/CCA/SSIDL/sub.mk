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

SRCDIR   := Core/CCA/SSIDL

SRCS     += \
            $(SRCDIR)/sidl_sidl.cc \
            $(SRCDIR)/BaseInterface.cc \
            $(SRCDIR)/BaseException.cc \
            $(SRCDIR)/BaseClass.cc \
            $(SRCDIR)/ClassInfo.cc \
            $(SRCDIR)/ClassInfoI.cc \
            $(SRCDIR)/Deserializer.cc \
            $(SRCDIR)/Serializer.cc \
            $(SRCDIR)/Serializeable.cc


#$(SRCDIR)/Object.cc
#$(SRCDIR)/DLL.cc

# We cannot use the implicit rule for SSIDL, since it needs that
# special -cia flag

# SSIDL has been replaced with sidl.sidl from Babel source
$(SRCDIR)/sidl_sidl.o: $(SRCDIR)/sidl_sidl.cc $(SRCDIR)/sidl_sidl.h

$(SRCDIR)/sidl_sidl.cc: $(SRCDIR)/sidl.sidl $(SIDL_EXE)
	$(SIDL_EXE) -cia -o $@ $<

$(SRCDIR)/sidl_sidl.h: $(SRCDIR)/sidl.sidl $(SIDL_EXE)
	$(SIDL_EXE) -cia -h -o $@ $<

GENHDRS := $(SRCDIR)/sidl_sidl.h

PSELIBS := Core/CCA/PIDL

LIBS := 

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

