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


#
# Makefile fragment for this subdirectory
#

SRCDIR   := Core/CCA/tools/kwai
SIDL_SRCDIR := $(SRCDIR)

GENSRCS := $(SRCDIR)/parser.cc $(SRCDIR)/parser.h $(SRCDIR)/scanner.cc
SRCS := $(SRCDIR)/main.cc $(SRCDIR)/parser.y $(SRCDIR)/scanner.l $(SRCDIR)/symTable.cc 


PROGRAM := $(SRCDIR)/kwai
KWAI_EXE := $(PROGRAM)
PSELIBS :=
LIBS := $(UUID_LIB) $(XML_LIBRARY)

include $(SCIRUN_SCRIPTS)/program.mk

$(SRCDIR)/scanner.cc: $(SRCDIR)/scanner.l
	$(LEX) -t $< > $@

$(SRCDIR)/scanner.o: $(SRCDIR)/parser.h

$(SRCDIR)/parser.cc: $(SRCDIR)/parser.y
	$(YACC) -y -d $<
	sed 's/getenv()/xxgetenv()/g' < y.tab.c > $@
	rm y.tab.c y.tab.h

$(SRCDIR)/parser.h: $(SRCDIR)/parser.y
	$(YACC) -y -d $<
	rm y.tab.c
	mv y.tab.h $@

# This dependency is so that the parallel build will not try to build
# parser.cc and parser.h at the same time
$(SRCDIR)/parser.cc: $(SRCDIR)/parser.h

SIDL_BUILTINS := $(SRCTOP_ABS)/Core/CCA/SSIDL/
CFLAGS_SIDLMAIN   := $(CFLAGS) -DSIDL_BUILTINS=\"$(SIDL_BUILTINS)\"

$(SRCDIR)/main.o:       $(SRCDIR)/main.cc
	$(CXX) $(CFLAGS_SIDLMAIN) $(INCLUDES) $(CC_DEPEND_REGEN) -c $< -o $@

