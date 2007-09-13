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

.PHONY: $(SRCTOP_ABS)/$(TMPSRCDIR)/$(COMPONENT)clientbabel.make.package

$(SRCTOP_ABS)/$(TMPSRCDIR)/glue/$(COMPONENT)clientbabel.make: $(patsubst %, $(SRCTOP_ABS)/%,$(CLIENT_SIDL)) $(CCASIDL) Core/Babel/timestamp
	$(BABEL) --client=$(BABEL_LANGUAGE) \
           --hide-glue \
           --make-prefix=$(subst clientbabel.make,,$(notdir $@))client \
           --repository-path=$(BABEL_REPOSITORY) \
           --output-directory=$(subst glue/,,$(dir $@)) $<

$(SRCTOP_ABS)/$(TMPSRCDIR)/$(COMPONENT)clientbabel.make.package: $(SRCTOP_ABS)/$(TMPSRCDIR)/glue/$(COMPONENT)clientbabel.make
#@echo "$(subst babel.make.package,,$(notdir $@)) CLIENT package in $(dir $@)!!!"

$(COMPONENT)clientSTUB_SRC_DIRS :=
include $(SRCTOP_ABS)/$(TMPSRCDIR)/$(COMPONENT)clientbabel.make.package
INCLUDES := $(INCLUDES) -I$($(COMPONENT)clientSTUB_SRC_DIRS)

$(COMPONENT)clientSTUBSRCS :=
include $(SRCTOP_ABS)/$(TMPSRCDIR)/glue/$(COMPONENT)clientbabel.make
SRCS := $(SRCS) $(patsubst %,$(TMPSRCDIR)/glue/%,$($(COMPONENT)clientSTUBSRCS))


