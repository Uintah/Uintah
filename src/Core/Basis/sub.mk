# 
# 
# The MIT License
# 
# Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
# Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
# University of Utah.
# 
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
# 
# 
# 
# 
# The contents of this file are subject to the University of Utah Public
# License (the "License"); you may not use this file except in compliance
# with the License.
# 
# Software distributed under the License is distributed on an "AS IS"
# basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
# License for the specific language governing rights and limitations under
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


PSELIBS :=
LIBS :=

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

