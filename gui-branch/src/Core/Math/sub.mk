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

FNSRCDIR	:= $(SRCTOP)/$(SRCDIR)

$(FNSRCDIR)/fnparser.cc \
$(FNSRCDIR)/fnparser.h:	$(FNSRCDIR)/fnparser.y;
	$(YACC) -p fn $(FNSRCDIR)/fnparser.y -o $(FNSRCDIR)/fnparser.cc
	mv -f $(FNSRCDIR)/fnparser.cc.h $(FNSRCDIR)/fnparser.h

$(FNSRCDIR)/fnscanner.cc: $(FNSRCDIR)/fnscanner.l $(FNSRCDIR)/fnparser.cc;
	$(LEX) -Pfn -o$(FNSRCDIR)/fnscanner.cc $(FNSRCDIR)/fnscanner.l

SRCS     += $(SRCDIR)/CubicPWI.cc              \
            $(SRCDIR)/LinAlg.c		       \
            $(SRCDIR)/LinearPWI.cc	       \
            $(SRCDIR)/Mat.c		       \
            $(SRCDIR)/MusilRNG.cc	       \
            $(SRCDIR)/PiecewiseInterp.cc       \
            $(SRCDIR)/TrigTable.cc	       \
            $(SRCDIR)/fft.c		       \
            $(SRCDIR)/fnparser.cc	       \
            $(SRCDIR)/fnscanner.cc	       \
            $(SRCDIR)/function.cc	       \
            $(SRCDIR)/hf.c                     \
            $(SRCDIR)/ssmult.c


PSELIBS := Core/Exceptions Core/Containers
LIBS := -lm

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

