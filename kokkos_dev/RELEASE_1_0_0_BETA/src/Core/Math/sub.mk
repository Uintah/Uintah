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

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Core/Math

FNSRCDIR	:= $(SRCTOP)/$(SRCDIR)

$(FNSRCDIR)/fnparser.cc\
$(FNSRCDIR)/fnparser.h:	$(FNSRCDIR)/fnparser.y;
	$(YACC) -p fn $(FNSRCDIR)/fnparser.y -o $(FNSRCDIR)/fnparser.cc
	mv -f $(FNSRCDIR)/fnparser.cc.h $(FNSRCDIR)/fnparser.h

$(FNSRCDIR)/fnscanner.cc: $(FNSRCDIR)/fnscanner.l $(FNSRCDIR)/fnparser.cc;
	$(LEX) -Pfn -o$(FNSRCDIR)/fnscanner.cc $(FNSRCDIR)/fnscanner.l

SRCS     += $(SRCDIR)/Mat.c $(SRCDIR)/MusilRNG.cc $(SRCDIR)/TrigTable.cc \
	$(SRCDIR)/LinAlg.c $(SRCDIR)/fft.c $(SRCDIR)/hf.c $(SRCDIR)/ssmult.c \
	$(SRCDIR)/PiecewiseInterp.cc $(SRCDIR)/LinearPWI.cc $(SRCDIR)/CubicPWI.cc \
	$(SRCDIR)/function.cc $(SRCDIR)/fnscanner.cc $(SRCDIR)/fnparser.cc


PSELIBS := Core/Exceptions Core/Containers
LIBS := -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

