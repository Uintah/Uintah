
SRCDIR   := SCIRun/Babel
FWKSIDL := ${SRCDIR}/framework.sidl

OUTPUTDIR :=${SRCTOP_ABS}/$(SRCDIR)

${OUTPUTDIR}/framework.make: ${FWKSIDL} Core/Babel/timestamp
	$(BABEL) -sC++ -o$(dir $@) -R${BABEL_REPOSITORY} $<	
	mv  $(dir $@)babel.make $@

${OUTPUTDIR}/cca.make: ${CCASIDL} 
	$(BABEL) -cC++ -o/${dir $@} $<
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
