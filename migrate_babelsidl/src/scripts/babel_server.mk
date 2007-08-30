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

## C++ is the default
ifndef BABEL_LANGUAGE
  include $(SCIRUN_SCRIPTS)/babel_component_cxx.mk
endif

TMPSRCDIR := $(SRCDIR)
COMPONENT := $(notdir $(TMPSRCDIR))

.PHONY: $(SRCTOP_ABS)/$(TMPSRCDIR)/$(COMPONENT)babel.make.package $(OBJTOP_ABS)/$(TMPSRCDIR)/glue

$(SRCTOP_ABS)/$(TMPSRCDIR)/$(COMPONENT)babel.make.package: $(SRCTOP_ABS)/$(TMPSRCDIR)/$(COMPONENT)babel.make
#@echo "$(subst babel.make.package,,$(notdir $@)) SERVER package in $(dir $@)!!!"

$(SRCTOP_ABS)/$(TMPSRCDIR)/$(COMPONENT)babel.make: $(SRCTOP_ABS)/$(TMPSRCDIR)/glue/$(COMPONENT)babel.make

$(SRCTOP_ABS)/$(TMPSRCDIR)/glue/$(COMPONENT)babel.make: $(OBJTOP_ABS)/$(TMPSRCDIR)/glue $(patsubst %, $(SRCTOP_ABS)/%,$(SERVER_SIDL)) Core/Babel/timestamp
	$(BABEL) --server=$(BABEL_LANGUAGE) \
           --output-directory=$(subst glue/,,$(dir $@)) \
           --make-prefix=$(subst babel.make,,$(notdir $@)) \
           --hide-glue \
           --repository-path=$(BABEL_REPOSITORY) \
           --vpath=$(subst glue/,,$(dir $@)) $(filter %.sidl, $+)

$(OBJTOP_ABS)/$(TMPSRCDIR)/glue:
	if ! test -d $@; then mkdir -p $@; fi

$(COMPONENT)IMPL_SRC_DIRS :=
$(COMPONENT)IOR_SRC_DIRS :=
include $(SRCTOP_ABS)/$(TMPSRCDIR)/$(COMPONENT)babel.make.package
INCLUDES := $(INCLUDES) -I$($(COMPONENT)IMPL_SRC_DIRS) -I$($(COMPONENT)IOR_SRC_DIRS)

$(COMPONENT)IMPLSRCS :=
include $(SRCTOP_ABS)/$(TMPSRCDIR)/$(COMPONENT)babel.make
SRCS := $(SRCS) $(patsubst %,$(TMPSRCDIR)/%,$($(COMPONENT)IMPLSRCS))

$(COMPONENT)IORSRCS :=
$(COMPONENT)STUBSRCS :=
$(COMPONENT)SKELSRCS :=
include $(SRCTOP_ABS)/$(TMPSRCDIR)/glue/$(COMPONENT)babel.make
SRCS := $(SRCS) $(patsubst %,$(TMPSRCDIR)/glue/%,$($(COMPONENT)IORSRCS) $($(COMPONENT)STUBSRCS) $($(COMPONENT)SKELSRCS))
