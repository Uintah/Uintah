include $(SCIRUN_SCRIPTS)/smallso_prologue.mk

#define BABELDIR BABEL BABEL_REPOSITORY CCASIDL MERGESRC here
include $(SCIRUN_SCRIPTS)/babel_defs.mk

COMPONENT:=$(notdir $(SRCDIR))

OUTDIR := ${SRCTOP_ABS}/${SRCDIR}

${OUTDIR}/${COMPONENT}.make: ${OUTDIR}/${COMPONENT}.sidl Core/Babel/timestamp
	$(BABEL) -sF77 -o$(dir $@) -R${BABEL_REPOSITORY} $<	
	mv  $(dir $@)babel.make $@

${OUTDIR}/${COMPONENT}_stub.make: ${OUTDIR}/${COMPONENT}.sidl Core/Babel/timestamp
	$(BABEL) -cC++ -o$(dir $@) -R${BABEL_REPOSITORY} $<	
	mv  $(dir $@)babel.make $@

${OUTDIR}/cca.make: ${CCASIDL} 
	$(BABEL) -cF77 -o/${dir $@} $<
	mv $(dir $@)babel.make $@

${OUTDIR}/cca_stub.make: ${CCASIDL} 
	$(BABEL) -cC++ -o/${dir $@} $<
	mv $(dir $@)babel.make $@

${OUTDIR}/srcfile.make: ${OUTDIR}/cca.make ${OUTDIR}/${COMPONENT}.make ${OUTDIR}/cca_stub.make ${OUTDIR}/${COMPONENT}_stub.make
	cd $(dir $@) && $(MERGESRC) $(notdir $^)

include ${OUTDIR}/srcfile.make

INCLUDES+= -I${BABELDIR}/include 

SRCS    += $(XSRCS) $(CSRC:%=$(SRCDIR)/%) $(CCSRC:%=$(SRCDIR)/%) $(F77SRC:%=$(SRCDIR)/%) 

PSELIBS := Core/CCA/Component/CIA Core/CCA/Component/PIDL Core/CCA/Component/Comm\
	Core/CCA/spec Core/Thread Core/Containers Core/Exceptions

#QT_LIBDIR := /home/sparker/SCIRun/SCIRun_Thirdparty_32_linux/lib
#LIBS := $(QT_LIBRARY)

LIBS := -L${BABELDIR}/lib -lsidl

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk

