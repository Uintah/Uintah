
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

SRCDIR   := CCA/Components/TableTennis

SRCS     += \
             $(SRCDIR)/TableTennis_sidl.cc \
             $(SRCDIR)/TableTennis.cc

# Hack until I get the guts to put this in Makefile.in  
$(SRCDIR)/TableTennis_sidl.o: $(SRCDIR)/TableTennis_sidl.cc $(SRCDIR)/TableTennis_sidl.h

$(SRCDIR)/TableTennis_sidl.cc: $(SRCDIR)/TableTennis.sidl $(SIDL_EXE)
	$(SIDL_EXE) -I $(SRCTOP_ABS)/Core/CCA/spec/cca.sidl -o $@ $<

$(SRCDIR)/TableTennis_sidl.h: $(SRCDIR)/TableTennis.sidl $(SIDL_EXE)
	$(SIDL_EXE) -I $(SRCTOP_ABS)/Core/CCA/spec/cca.sidl -h -o $@ $<

GENHDRS := $(SRCDIR)/TableTennis_sidl.h
PSELIBS := Core/CCA/SSIDL Core/CCA/PIDL Core/CCA/Comm \
           Core/CCA/spec Core/Thread Core/Containers Core/Exceptions

ifeq ($(HAVE_QT),yes)
  LIBS := $(QT_LIBRARY)
endif

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

#include $(SCIRUN_SCRIPTS)/program.mk

$(SRCDIR)/TableTennis.o: Core/CCA/spec/cca_sidl.h
