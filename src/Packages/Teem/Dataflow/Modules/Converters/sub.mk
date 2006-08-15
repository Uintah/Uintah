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


# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the Core/CCA/Component Wizard to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Core/CCA/Component"
# documentation on how to do it correctly.

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

INCLUDES += $(TEEM_INCLUDE)



SRCDIR   := Packages/Teem/Dataflow/Modules/Converters


SRCS     += \
        $(SRCDIR)/ColorMap2ToNrrd.cc\
        $(SRCDIR)/ColorMapToNrrd.cc\
        $(SRCDIR)/ConvertToNrrd.cc\
        $(SRCDIR)/FieldToNrrd.cc\
        $(SRCDIR)/NrrdToColorMap2.cc\
        $(SRCDIR)/NrrdToField.cc\
        $(SRCDIR)/NrrdToMatrix.cc\
        $(SRCDIR)/MatrixToNrrd.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := \
        Core/Basis         \
        Core/Containers    \
        Core/Datatypes     \
        Core/Exceptions    \
        Core/Geom          \
        Core/Geometry      \
        Core/GeomInterface \
        Core/GuiInterface  \
        Core/Persistent    \
        Core/Thread        \
        Core/TkExtensions  \
        Core/Util          \
        Core/Volume        \
        Dataflow/Network   

LIBS := $(TEEM_LIBRARY) $(PNG_LIBRARY) $(Z_LIBRARY) $(MAGICK_LIBRARY)

ifeq ($(HAVE_INSIGHT),yes)
  PSELIBS += Core/Algorithms/DataIO

  LIBS += $(INSIGHT_LIBRARY)
endif


include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

ifeq ($(LARGESOS),no)
  TEEM_MODULES := $(TEEM_MODULES) $(LIBNAME)
endif
