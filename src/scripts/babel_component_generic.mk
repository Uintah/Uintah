
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
	$(BABEL) --server=$($(basename $(notdir $@))_LANGUAGE) --output-directory=$(dir $@) --repository-path=${BABEL_REPOSITORY} --hide-glue --suppress-timestamp --language-subdir $<
	mv  $(dir $@)babel.make $@

${OUTDIR}/${COMPONENT}_stub.make: ${SRCDIR}/${COMPONENT}.sidl Core/Babel/timestamp
	$(BABEL) --client=C++ --output-directory=$(dir $@) --repository-path=${BABEL_REPOSITORY} $< 
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
