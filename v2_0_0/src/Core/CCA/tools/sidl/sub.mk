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

#
# Makefile fragment for this subdirectory
#

SRCDIR   := Core/CCA/tools/sidl
SIDL_SRCDIR := $(SRCDIR)

GENSRCS := $(SRCDIR)/parser.cc $(SRCDIR)/parser.h $(SRCDIR)/scanner.cc
SRCS := $(SRCDIR)/Spec.cc $(SRCDIR)/SymbolTable.cc $(SRCDIR)/emit.cc \
        $(SRCDIR)/main.cc $(SRCDIR)/parser.y $(SRCDIR)/scanner.l \
        $(SRCDIR)/static_check.cc

PROGRAM := $(SRCDIR)/sidl
SIDL_EXE := $(PROGRAM)
PSELIBS :=
LIBS := $(UUID_LIB)

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

