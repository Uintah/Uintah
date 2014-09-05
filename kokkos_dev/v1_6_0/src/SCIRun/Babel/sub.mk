SRCDIR   := SCIRun/Babel
FWKSIDL := ${SRCDIR}/framework.sidl

include $(SCIRUN_SCRIPTS)/babel_defs.mk

OUTPUTDIR :=${SRCTOP_ABS}/$(SRCDIR)

${OUTPUTDIR}/framework.make: ${FWKSIDL} Core/Babel/timestamp
	$(BABEL) -sC++ -o$(dir $@) -R${BABEL_REPOSITORY} $<	
	mv  $(dir $@)babel.make $@

${OUTPUTDIR}/cca.make: ${CCASIDL} 
	$(BABEL) -cC++ -o/${dir $@} $<
	mv $(dir $@)babel.make $@

${OUTPUTDIR}/srcfile.make: ${OUTPUTDIR}/cca.make ${OUTPUTDIR}/framework.make
	cd $(dir $@) && $(MERGESRC) $(notdir $^)

include ${OUTPUTDIR}/srcfile.make

INCLUDES+=-I${BABELDIR}/include

SRCS     +=  $(CSRC:%=$(SRCDIR)/%) $(CCSRC:%=$(SRCDIR)/%) \
	$(SRCDIR)/BabelComponentModel.cc \
	$(SRCDIR)/BabelComponentInstance.cc \
	$(SRCDIR)/BabelComponentDescription.cc \
	$(SRCDIR)/BabelPortInstance.cc 







