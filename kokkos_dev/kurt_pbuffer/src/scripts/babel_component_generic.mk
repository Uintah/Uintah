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


include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCS := $(XSRCS)

COMPONENT:=$(notdir $(SRCDIR))

#OUTDIR := ${SRCTOP_ABS}/${SRCDIR}
OUTDIR := ${OBJTOP_ABS}/${SRCDIR}

$(COMPONENT)_LANGUAGE := $(BABEL_LANGUAGE)

# This rule uses the babel compiler to generate the language-specific glue and implementation
# files.  Since implementation code (*Impl*) already exists in the source tree, we first copy this existing
# code to the babel target output directory.  Babel will not overwrite critical sections of the 
# existing implementation code.
${OUTDIR}/${COMPONENT}.make: ${SRCDIR}/${COMPONENT}.sidl Core/Babel/timestamp
	if ! test -d $(dir $@); then mkdir -p $(dir $@); fi
	cp -u $(dir $<)*Impl.* $(dir $@)
	$(BABEL) --server=$($(basename $(notdir $@))_LANGUAGE) --output-directory=$(dir $@) --repository-path=${BABEL_REPOSITORY} --suppress-timestamp $<
	mv  $(dir $@)babel.make $@

${OUTDIR}/${COMPONENT}_stub.make: ${SRCDIR}/${COMPONENT}.sidl Core/Babel/timestamp
	$(BABEL) --client=C++ --output-directory=$(dir $@) --repository-path=${BABEL_REPOSITORY} --suppress-timestamp $< 
	mv  $(dir $@)babel.make $@

#${OUTDIR}/cca.make: ${CCASIDL} 
#	$(BABEL) -c$($(COMPONENT)_LANGUAGE) -o/${dir $@} $<
#	mv $(dir $@)babel.make $@

#${OUTDIR}/cca_stub.make: ${CCASIDL} 
#	$(BABEL) -cC++ -o/${dir $@} $<
#	mv $(dir $@)babel.make $@

IORSRCS :=
STUBSRCS :=
IMPLSRCS :=
SKELSRCS :=
#include ${OUTDIR}/cca.make

IORSRCS :=
STUBSRCS :=
IMPLSRCS :=
SKELSRCS :=
#SRCS := $(SRCS) $(patsubst %,$(SRCDIR)/%,$(IORSRCS) $(STUBSRCS) $(IMPLSRCS) $(SKELSRCS))

IORSRCS :=
STUBSRCS :=
IMPLSRCS :=
SKELSRCS :=
#include ${OUTDIR}/cca_stub.make
#SRCS := $(SRCS) $(patsubst %,$(SRCDIR)/%,$(IORSRCS) $(STUBSRCS) $(IMPLSRCS) $(SKELSRCS))

IORSRCS :=
STUBSRCS :=
IMPLSRCS :=
SKELSRCS :=
include ${OUTDIR}/${COMPONENT}.make
SRCS := $(SRCS) $(patsubst %,$(SRCDIR)/%,$(IORSRCS) $(STUBSRCS) $(IMPLSRCS) $(SKELSRCS))

IORSRCS :=
STUBSRCS :=
IMPLSRCS :=
SKELSRCS :=
#include ${OUTDIR}/${COMPONENT}_stub.make
#SRCS := $(SRCS) $(patsubst %,$(SRCDIR)/%,$(IORSRCS) $(STUBSRCS) $(IMPLSRCS) $(SKELSRCS))

PSELIBS := Core/CCA/SSIDL Core/CCA/PIDL \
	Core/CCA/Comm Core/CCA/spec Core/Thread \
	Core/Containers Core/Exceptions

LIBS := $(SIDL_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
