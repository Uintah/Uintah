
SRCDIR   := SCIRun/Babel
FWKSIDL := ${SRCDIR}/framework.sidl

#OUTPUTDIR :=${SRCTOP_ABS}/$(SRCDIR)
OUTPUTDIR :=${OBJTOP_ABS}/$(SRCDIR)

INCLUDES := -I$(OUTPUTDIR) $(INCLUDES)
LIBS := $(QT_LIBRARY) $(LIBS)

${OUTPUTDIR}/framework.make: ${FWKSIDL} Core/Babel/timestamp
	if ! test -d $(dir $@); then mkdir -p $(dir $@); fi
	cp -u $(dir $<)*Impl.* $(dir $@)
	$(BABEL) --server=C++ --output-directory=$(dir $@) --repository-path=${BABEL_REPOSITORY} --suppress-timestamp $<
	mv  $(dir $@)babel.make $@

${OUTPUTDIR}/cca.make: ${CCASIDL} 
	$(BABEL) --client=C++ --output-directory=${dir $@} --suppress-timestamp $<
	mv $(dir $@)babel.make $@

SRCS := $(SRCS) $(SRCDIR)/BabelComponentModel.cc \
	$(SRCDIR)/BabelComponentInstance.cc \
	$(SRCDIR)/BabelComponentDescription.cc \
	$(SRCDIR)/BabelPortInstance.cc 
IORSRCS :=
STUBSRCS :=
IMPLSRCS :=
SKELSRCS :=
include ${OUTPUTDIR}/framework.make
SRCS := $(SRCS) $(patsubst %,$(SRCDIR)/%,$(IORSRCS) $(STUBSRCS) $(IMPLSRCS) $(SKELSRCS))

IORSRCS :=
STUBSRCS :=
IMPLSRCS :=
SKELSRCS :=
include ${OUTPUTDIR}/cca.make
SRCS := $(SRCS) $(patsubst %,$(SRCDIR)/%,$(IORSRCS) $(STUBSRCS) $(IMPLSRCS) $(SKELSRCS))
