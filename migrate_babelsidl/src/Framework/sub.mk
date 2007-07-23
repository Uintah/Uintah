SRCDIR   := Framework
FWKSIDL := ${SRCDIR}/framework.sidl
OUTPUTDIR :=${OBJTOP_ABS}/$(SRCDIR)

INCLUDES := -I$(OUTPUTDIR) $(INCLUDES)

GLUEDIR := glue

ifeq ($(IS_OSX),yes)
  CP_FLAGS :=
else
  CP_FLAGS := -u
endif

## use babel --vpath<path> for Impl sources

${OUTPUTDIR}/babel.make: ${OUTPUTDIR}/${GLUEDIR}/server.make

${OUTPUTDIR}/${GLUEDIR}/server.make: ${FWKSIDL} Core/Babel/timestamp
#if ! test -d $(dir $@); then mkdir -p $(dir $@); fi
	if ! test -d ${OUTPUTDIR}; then mkdir -p ${OUTPUTDIR}; fi
#cp $(CP_FLAGS) $(dir $<)*Impl.* $(dir $@)
	$(BABEL) --server=C++ --output-directory=${OUTPUTDIR} --hide-glue --repository-path=${BABEL_REPOSITORY} $<
	mv $(dir $@)babel.make $@

${OUTPUTDIR}/${GLUEDIR}/client.make: ${CCASIDL}
	$(BABEL) --client=C++ --hide-glue --output-directory=${OUTPUTDIR} $<
	mv $(dir $@)babel.make $@

include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

#SRCS := $(SRCS) $(SRCDIR)/BabelComponentModel.cc \
#        $(SRCDIR)/BabelComponentInstance.cc \
#        $(SRCDIR)/BabelComponentDescription.cc \
#        $(SRCDIR)/BabelPortInstance.cc

SRCS :=

IORSRCS :=
STUBSRCS :=
SKELSRCS :=
include ${OUTPUTDIR}/${GLUEDIR}/server.make
SRCS += $(patsubst %,$(OUTPUTDIR)/$(GLUEDIR)/%,$(IORSRCS) $(STUBSRCS) $(SKELSRCS))

STUBSRCS :=
include ${OUTPUTDIR}/${GLUEDIR}/client.make
SRCS += $(patsubst %,$(OUTPUTDIR)/$(GLUEDIR)/%,$(STUBSRCS))

IMPLSRCS :=
include ${OUTPUTDIR}/babel.make
SRCS += $(patsubst %,$(OUTPUTDIR)/%,$(IMPLSRCS))

PSELIBS :=
INCLUDES += -I$(OUTPUTDIR)/$(GLUEDIR) $(BABEL_INCLUDE)
LIBS := $(BABEL_LIBRARY)

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
