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

#OUTDIR := ${SRCTOP_ABS}/${SRCDIR}
#OUTDIR := ${OBJTOP_ABS}/${SRCDIR}

SRCDIR_ABS :=  ${SRCTOP_ABS}/${SRCDIR}
IMPLDIR_ABS := ${SRCTOP_ABS}/${SRCDIR}
IMPLDIR := ${SRCDIR}
GLUEDIR_ABS := ${SRCTOP_ABS}/${SRCDIR}/glue
GLUEDIR := ${SRCDIR}/glue

OUTPUTDIR_ABS := ${OBJTOP_ABS}/${SRCDIR}
OUTIMPLDIR_ABS := ${OBJTOP_ABS}/${SRCDIR}
OUTGLUEDIR_ABS := ${OBJTOP_ABS}/${SRCDIR}/glue

## C++ is the default
ifndef BABEL_LANGUAGE
  include $(SCIRUN_SCRIPTS)/babel_component_cxx.mk
endif

ifeq ($(strip $(MAKE_CLIENT)), yes)
  include $(SCIRUN_SCRIPTS)/babel_client.mk
endif

#${OUTDIR}/${COMPONENT}_stub.make: ${SRCDIR}/${COMPONENT}.sidl Core/Babel/timestamp
#$(BABEL) --client=C++ --output-directory=$(dir $@) --repository-path=${BABEL_REPOSITORY} $<
#mv  $(dir $@)babel.make $@

$(IMPLDIR_ABS)/$(COMPONENT)babel.make: $(GLUEDIR_ABS)/$(COMPONENT)babel.make

## maybe it would be good to make sure there's no directory component to the server sidl file names
$(GLUEDIR_ABS)/$(COMPONENT)babel.make: $(patsubst %,$(SRCDIR_ABS)/%,$(SERVER_SIDL)) Core/Babel/timestamp
	if ! test -d $(OUTGLUEDIR_ABS); then mkdir -p $(OUTGLUEDIR_ABS); fi
	$(BABEL) --server=$(BABEL_LANGUAGE) \
           --output-directory=$(IMPLDIR_ABS) \
           --make-prefix=$(COMPONENT) \
           --hide-glue \
           --repository-path=$(BABEL_REPOSITORY) \
           --vpath=$(IMPLDIR_ABS) $(filter %.sidl, $+)

$(COMPONENT)IORSRCS :=
$(COMPONENT)STUBSRCS :=
$(COMPONENT)SKELSRCS :=
include $(GLUEDIR_ABS)/$(COMPONENT)babel.make
SRCS += $(patsubst %,$(GLUEDIR)/%,$($(COMPONENT)IORSRCS) $($(COMPONENT)STUBSRCS) $($(COMPONENT)SKELSRCS))

$(COMPONENT)IMPLSRCS :=
include $(IMPLDIR_ABS)/$(COMPONENT)babel.make
SRCS += $(patsubst %,$(IMPLDIR)/%,$($(COMPONENT)IMPLSRCS))

PSELIBS := Framework Core/Exceptions
LIBS := $(BABEL_LIBRARY)

INCLUDES += -I$(IMPLDIR_ABS) -I$(GLUEDIR_ABS) $(BABEL_INCLUDE)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
