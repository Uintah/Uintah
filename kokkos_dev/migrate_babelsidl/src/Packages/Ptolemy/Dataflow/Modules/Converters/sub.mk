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

SRCDIR   := Packages/Ptolemy/Dataflow/Modules/Converters

SRCS     += \
	$(SRCDIR)/PTIIDataToNrrd.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Core/Datatypes Core/Persistent Core/Containers \
           Core/Util Core/Exceptions Core/Thread Core/GuiInterface \
           Core/Geom Core/Geometry Core/GeomInterface Core/TkExtensions \
           Dataflow/Network Core/Volume \
           Dataflow/Modules/Render \
           Packages/Ptolemy/Core/Datatypes
#Core/Algorithms/DataIO

ifeq ($(HAVE_JAVA), yes)
 PSELIBS += Packages/Ptolemy/Core/PtolemyInterface
endif

LIBS := $(TK_LIBRARY) $(TEEM_LIBRARY) $(PNG_LIBRARY) $(Z_LIBRARY) $(MAGICK_LIBRARY)
#LIBS := $(LIBS) $(TK_LIBRARY) $(TEEM_LIBRARY)


include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

#ifeq ($(LARGESOS),no)
#TEEM_MODULES := $(TEEM_MODULES) $(LIBNAME)
#endif
