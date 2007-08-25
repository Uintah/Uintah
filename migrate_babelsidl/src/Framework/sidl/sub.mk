#
#  For more information, please see: http://software.sci.utah.edu
#
#  The MIT License
#
#  Copyright (c) 2007 Scientific Computing and Imaging Institute,
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


# Makefile fragment for this subdirectory

#include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCDIR := Framework/sidl

FWKSRCDIR_ABS :=  $(SRCTOP_ABS)/$(SRCDIR)
FWKIMPLDIR_ABS := $(SRCTOP_ABS)/$(SRCDIR)/Impl
FWKIMPLDIR := $(SRCDIR)/Impl
FWKGLUEDIR_ABS := $(SRCTOP_ABS)/$(SRCDIR)/Impl/glue
FWKGLUEDIR := $(SRCDIR)/Impl/glue

FWKSIDL := \
        $(FWKSRCDIR_ABS)/sci-cca.sidl \
        $(FWKSRCDIR_ABS)/scijump.sidl

FWKOUTPUTDIR_ABS := $(OBJTOP_ABS)/$(SRCDIR)
FWKOUTIMPLDIR_ABS := $(OBJTOP_ABS)/$(SRCDIR)/Impl
FWKOUTGLUEDIR_ABS := $(OBJTOP_ABS)/$(SRCDIR)/Impl/glue

$(FWKIMPLDIR_ABS)/serverbabel.make: $(FWKGLUEDIR_ABS)/serverbabel.make

#--generate-subdirs
$(FWKGLUEDIR_ABS)/serverbabel.make: $(FWKSIDL) Core/Babel/timestamp
	if ! test -d $(FWKOUTGLUEDIR_ABS); then mkdir -p $(FWKOUTGLUEDIR_ABS); fi
	$(BABEL) --server=C++ \
           --output-directory=$(FWKIMPLDIR_ABS) \
           --make-prefix=server \
           --hide-glue \
           --repository-path=$(BABEL_REPOSITORY) \
           --vpath=$(FWKIMPLDIR_ABS) $(filter %.sidl, $+)

$(FWKGLUEDIR_ABS)/clientbabel.make: $(CCASIDL)
	$(BABEL) --client=C++ --hide-glue --make-prefix=client --output-directory=$(FWKIMPLDIR_ABS) $<

serverIORSRCS :=
serverSTUBSRCS :=
serverSKELSRCS :=
include $(FWKGLUEDIR_ABS)/serverbabel.make
SRCS += $(patsubst %,$(FWKGLUEDIR)/%,$(serverIORSRCS) $(serverSTUBSRCS) $(serverSKELSRCS))

clientSTUBSRCS :=
include $(FWKGLUEDIR_ABS)/clientbabel.make
SRCS += $(patsubst %,$(FWKGLUEDIR)/%,$(clientSTUBSRCS))

serverIMPLSRCS :=
include $(FWKIMPLDIR_ABS)/serverbabel.make
SRCS += $(patsubst %,$(FWKIMPLDIR)/%,$(serverIMPLSRCS))

#PSELIBS := Core/Thread Core/Exceptions Framework/Core
#LIBS := $(BABEL_LIBRARY)

INCLUDES += -I$(FWKIMPLDIR_ABS) -I$(FWKGLUEDIR_ABS) $(BABEL_INCLUDE)

#include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
