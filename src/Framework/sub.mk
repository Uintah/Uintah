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


# Makefile fragment for this subdirectory

SRCDIR := Framework

SRCDIR_ABS :=  $(SRCTOP_ABS)/$(SRCDIR)
SIDL := \
        $(SRCDIR_ABS)/sci-cca.sidl \
        $(SRCDIR_ABS)/scijump.sidl

OUTPUTDIR_ABS := $(OBJTOP_ABS)/$(SRCDIR)
GLUEDIR := $(OUTPUTDIR_ABS)/glue

# Need a list ($(FWK_IMPLSRCS) of Babel impl files as target dependencies,
# which forces running Babel whenever the impl files are changed.
#
# The full list of Babel impl files is included from Babel-generated
# makefiles (below).

FWK_IMPLSRCS := \
                $(SRCDIR_ABS)/scijump_Topic_Impl.hxx \
                $(SRCDIR_ABS)/scijump_Topic_Impl.cxx

$(OUTPUTDIR_ABS)/babel.make: $(GLUEDIR)/server.make

#--generate-subdirs
$(GLUEDIR)/server.make: $(SIDL) $(FWK_IMPLSRCS) Core/Babel/timestamp
	if ! test -d $(OUTPUTDIR_ABS); then mkdir -p $(OUTPUTDIR_ABS); fi
	$(BABEL) --server=C++ --output-directory=$(OUTPUTDIR_ABS) --hide-glue --repository-path=$(BABEL_REPOSITORY) --vpath=$(SRCDIR_ABS) $(filter %.sidl, $+)
	mv $(dir $@)babel.make $@

$(GLUEDIR)/client.make: $(CCASIDL)
	$(BABEL) --client=C++ --hide-glue --output-directory=$(OUTPUTDIR_ABS) $<
	mv $(dir $@)babel.make $@

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

#SRCS :=

IORSRCS :=
STUBSRCS :=
SKELSRCS :=
include $(GLUEDIR)/server.make
SRCS += $(patsubst %,$(GLUEDIR)/%,$(IORSRCS) $(STUBSRCS) $(SKELSRCS))

STUBSRCS :=
include $(GLUEDIR)/client.make
SRCS += $(patsubst %,$(GLUEDIR)/%,$(STUBSRCS))

IMPLSRCS :=
include $(OUTPUTDIR_ABS)/babel.make
SRCS += $(patsubst %,$(OUTPUTDIR_ABS)/%,$(IMPLSRCS))

PSELIBS :=
INCLUDES += -I$(OUTPUTDIR_ABS) -I$(GLUEDIR) $(BABEL_INCLUDE)
LIBS := $(BABEL_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
