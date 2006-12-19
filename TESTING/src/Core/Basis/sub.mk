# The contents of this file are subject to the University of Utah Public
# License (the "License"); you may not use this file except in compliance
# with the License.
# 
# Software distributed under the License is distributed on an "AS IS"
# basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
# 
# the License.
# 
# The Original Source Code is SCIRun, released March 12, 2001.
# 
# The Original Source Code was developed by the University of Utah.
# Portions created by UNIVERSITY are Copyright (C) 2001, 1994
# University of Utah. All Rights Reserved.
# 
#   File   : sub.mk<2>
#   Author : Martin Cole
#   Date   : Wed Apr 28 10:26:53 2004

# Makefile fragment for this subdirectory

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR   := Core/Basis

SRCS     +=      $(SRCDIR)/CrvLinearLgn.cc             \
                 $(SRCDIR)/CrvQuadraticLgn.cc          \
                 $(SRCDIR)/HexTrilinearLgn.cc          \
                 $(SRCDIR)/HexTriquadraticLgn.cc          \
                 $(SRCDIR)/Locate.cc                   \
                 $(SRCDIR)/PrismLinearLgn.cc           \
                 $(SRCDIR)/PrismQuadraticLgn.cc           \
                 $(SRCDIR)/QuadBilinearLgn.cc          \
                 $(SRCDIR)/QuadBiquadraticLgn.cc          \
                 $(SRCDIR)/TetLinearLgn.cc             \
                 $(SRCDIR)/TetQuadraticLgn.cc          \
                 $(SRCDIR)/TriLinearLgn.cc             \
                 $(SRCDIR)/TriQuadraticLgn.cc          \
                 $(SRCDIR)/TriCubicHmt.cc          \
								 $(SRCDIR)/NoData.cc \
								 $(SRCDIR)/Constant.cc \

PSELIBS :=
LIBS :=

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

