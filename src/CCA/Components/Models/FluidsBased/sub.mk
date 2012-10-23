#
#  The MIT License
#
#  Copyright (c) 1997-2012 The University of Utah
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to
#  deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#  IN THE SOFTWARE.
# 
# 
# 
# 
# 
# Makefile fragment for this subdirectory 

SRCDIR := CCA/Components/Models/FluidsBased

# Uncomment this like to compile with cantera
#CANTERA_DIR := /home/sci/sparker/canterataz
ifneq ($(CANTERA_DIR),)
 INCLUDES := $(INCLUDES) -I$(CANTERA_DIR)/include
 CANTERA_LIBRARY := -L$(CANTERA_DIR)/lib/cantera -loneD -lzeroD -ltransport -lconverters -lcantera -lrecipes -lcvode -lctlapack -lctmath -lctblas -lctcxx
endif

SRCS += \
       $(SRCDIR)/ArchesTable.cc    \
       $(SRCDIR)/TableInterface.cc \
       $(SRCDIR)/TableFactory.cc 

#       $(SRCDIR)/Mixing2.cc
#       $(SRCDIR)/Mixing2.cc \
#       $(SRCDIR)/Mixing3.cc

ifneq ($(BUILD_ICE),no) 
  SRCS += \
       $(SRCDIR)/AdiabaticTable.cc     \
       $(SRCDIR)/flameSheet_rxn.cc     \
       $(SRCDIR)/MaterialProperties.cc \
       $(SRCDIR)/Mixing.cc             \
       $(SRCDIR)/NonAdiabaticTable.cc  \
       $(SRCDIR)/PassiveScalar.cc      \
       $(SRCDIR)/SimpleRxn.cc          \
       $(SRCDIR)/TestModel.cc          \
       $(SRCDIR)/MassMomEng_src.cc
endif

