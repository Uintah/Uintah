#
#  The MIT License
#
#  Copyright (c) 1997-2018 The University of Utah
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

SRCDIR := CCA/Components/PhaseField/AMR

SRCS += \
  $(SRCDIR)/AMRFDViewHeatProblemCCP5FC0-bld.cc \
  $(SRCDIR)/AMRFDViewHeatProblemCCP5FC1-bld.cc \
  $(SRCDIR)/AMRFDViewHeatProblemCCP5FCSimple-bld.cc \
  $(SRCDIR)/AMRFDViewHeatProblemCCP5FCLinear-bld.cc \
  $(SRCDIR)/AMRFDViewHeatProblemCCP5FCBilinear-bld.cc \
  $(SRCDIR)/AMRFDViewHeatProblemCCP7FC0-bld.cc \
  $(SRCDIR)/AMRFDViewHeatProblemCCP7FC1-bld.cc \
  $(SRCDIR)/AMRFDViewHeatProblemNCP5FC0-bld.cc \
  $(SRCDIR)/AMRFDViewHeatProblemNCP5FC1-bld.cc \
  $(SRCDIR)/AMRFDViewHeatProblemNCP5FCSimple-bld.cc \
  $(SRCDIR)/AMRFDViewHeatProblemNCP5FCLinear-bld.cc \
  $(SRCDIR)/AMRFDViewHeatProblemNCP5FCBilinear-bld.cc \
  $(SRCDIR)/AMRFDViewHeatProblemNCP7FC0-bld.cc \
  $(SRCDIR)/AMRFDViewHeatProblemNCP7FC1-bld.cc \
  $(SRCDIR)/AMRFDViewPureMetalProblemCCP5FC0-bld.cc \
  $(SRCDIR)/AMRFDViewPureMetalProblemCCP5FC1-bld.cc \
  $(SRCDIR)/AMRFDViewPureMetalProblemCCP5FCSimple-bld.cc \
  $(SRCDIR)/AMRFDViewPureMetalProblemCCP5FCLinear-bld.cc \
  $(SRCDIR)/AMRFDViewPureMetalProblemCCP7FC0-bld.cc \
  $(SRCDIR)/AMRFDViewPureMetalProblemCCP7FC1-bld.cc \
  $(SRCDIR)/AMRFDViewPureMetalProblemCCP5FCBilinear-bld.cc \
  $(SRCDIR)/AMRFDViewPureMetalProblemNCP5FC0-bld.cc \
  $(SRCDIR)/AMRFDViewPureMetalProblemNCP5FC1-bld.cc \
  $(SRCDIR)/AMRFDViewPureMetalProblemNCP5FCSimple-bld.cc \
  $(SRCDIR)/AMRFDViewPureMetalProblemNCP5FCLinear-bld.cc \
  $(SRCDIR)/AMRFDViewPureMetalProblemNCP5FCBilinear-bld.cc \
  $(SRCDIR)/AMRFDViewPureMetalProblemNCP7FC0-bld.cc \
  $(SRCDIR)/AMRFDViewPureMetalProblemNCP7FC1-bld.cc \

BLDDIR := $(SRCTOP)/$(SRCDIR)

BLDSRCS += \
  $(BLDDIR)/AMRFDViewHeatProblemCCP5FC0-bld.cc \
  $(BLDDIR)/AMRFDViewHeatProblemCCP5FC1-bld.cc \
  $(BLDDIR)/AMRFDViewHeatProblemCCP5FCSimple-bld.cc \
  $(BLDDIR)/AMRFDViewHeatProblemCCP5FCLinear-bld.cc \
  $(BLDDIR)/AMRFDViewHeatProblemCCP5FCBilinear-bld.cc \
  $(BLDDIR)/AMRFDViewHeatProblemCCP7FC0-bld.cc \
  $(BLDDIR)/AMRFDViewHeatProblemCCP7FC1-bld.cc \
  $(BLDDIR)/AMRFDViewHeatProblemNCP5FC0-bld.cc \
  $(BLDDIR)/AMRFDViewHeatProblemNCP5FC1-bld.cc \
  $(BLDDIR)/AMRFDViewHeatProblemNCP5FCSimple-bld.cc \
  $(BLDDIR)/AMRFDViewHeatProblemNCP5FCLinear-bld.cc \
  $(BLDDIR)/AMRFDViewHeatProblemNCP5FCBilinear-bld.cc \
  $(BLDDIR)/AMRFDViewHeatProblemNCP7FC0-bld.cc \
  $(BLDDIR)/AMRFDViewHeatProblemNCP7FC1-bld.cc \
  $(BLDDIR)/AMRFDViewPureMetalProblemCCP5FC0-bld.cc \
  $(BLDDIR)/AMRFDViewPureMetalProblemCCP5FC1-bld.cc \
  $(BLDDIR)/AMRFDViewPureMetalProblemCCP5FCSimple-bld.cc \
  $(BLDDIR)/AMRFDViewPureMetalProblemCCP5FCLinear-bld.cc \
  $(BLDDIR)/AMRFDViewPureMetalProblemCCP7FC0-bld.cc \
  $(BLDDIR)/AMRFDViewPureMetalProblemCCP7FC1-bld.cc \
  $(BLDDIR)/AMRFDViewPureMetalProblemCCP5FCBilinear-bld.cc \
  $(BLDDIR)/AMRFDViewPureMetalProblemNCP5FC0-bld.cc \
  $(BLDDIR)/AMRFDViewPureMetalProblemNCP5FC1-bld.cc \
  $(BLDDIR)/AMRFDViewPureMetalProblemNCP5FCSimple-bld.cc \
  $(BLDDIR)/AMRFDViewPureMetalProblemNCP5FCLinear-bld.cc \
  $(BLDDIR)/AMRFDViewPureMetalProblemNCP5FCBilinear-bld.cc \
  $(BLDDIR)/AMRFDViewPureMetalProblemNCP7FC0-bld.cc \
  $(BLDDIR)/AMRFDViewPureMetalProblemNCP7FC1-bld.cc \
