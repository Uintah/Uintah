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

SRCDIR   := Core/Math

FNSRCDIR  := $(SRCTOP)/$(SRCDIR)

# TARGDIR is not really 'srcdir'... however, it is given the 
# same value... but since it is used based on the top of the compilation
# tree, it has the "same value".
TARGDIR := $(SRCDIR)

$(TARGDIR)/fnparser.cc $(TARGDIR)/fnparser.h:	$(FNSRCDIR)/fnparser.y
	$(YACC) -o $(TARGDIR)/y.tab.c -p fn $(FNSRCDIR)/fnparser.y
	mv -f $(TARGDIR)/y.tab.h $(TARGDIR)/fnparser.h
	mv -f $(TARGDIR)/y.tab.c $(TARGDIR)/fnparser.cc

$(TARGDIR)/fnscanner.cc: $(FNSRCDIR)/fnscanner.l $(TARGDIR)/fnparser.cc
	$(LEX) -Pfn -o$(TARGDIR)/fnscanner.cc $(FNSRCDIR)/fnscanner.l

SRCS     += $(SRCDIR)/CubicPWI.cc              \
            $(SRCDIR)/Gaussian.cc	       \
            $(SRCDIR)/LinAlg.c		       \
            $(SRCDIR)/LinearPWI.cc	       \
            $(SRCDIR)/Mat.c		       \
            $(SRCDIR)/MusilRNG.cc	       \
            $(SRCDIR)/PiecewiseInterp.cc       \
            $(SRCDIR)/TrigTable.cc	       \
            $(SRCDIR)/fft.c		       \
            $(TARGDIR)/fnparser.cc	       \
            $(TARGDIR)/fnscanner.cc	       \
            $(SRCDIR)/function.cc	       \
            $(SRCDIR)/ssmult.c


PSELIBS := Core/Exceptions Core/Containers
LIBS := $(M_LIBRARY) $(DL_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

