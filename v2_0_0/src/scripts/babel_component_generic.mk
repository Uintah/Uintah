
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

SRCS := $(XSRCS)

COMPONENT:=$(notdir $(SRCDIR))

OUTDIR := ${SRCTOP_ABS}/${SRCDIR}

$(COMPONENT)_LANGUAGE := $(BABEL_LANGUAGE)

${OUTDIR}/${COMPONENT}.make: ${OUTDIR}/${COMPONENT}.sidl Core/Babel/timestamp
	$(BABEL) -s$($(basename $(notdir $@))_LANGUAGE) -o$(dir $@) -R${BABEL_REPOSITORY} $<	
	mv  $(dir $@)babel.make $@

${OUTDIR}/${COMPONENT}_stub.make: ${OUTDIR}/${COMPONENT}.sidl Core/Babel/timestamp
	$(BABEL) -cC++ -o$(dir $@) -R${BABEL_REPOSITORY} $<	
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
