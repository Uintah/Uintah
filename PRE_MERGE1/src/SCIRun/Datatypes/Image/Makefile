THISDIR = Datatypes

SECTION = SCIRun

include $(TOP)/make.defaults

all : doit

clean :
	@ echo "Making ALL of Datatypes Clean in Architecture Specific directory [$(ARCH)]"
	@ cd $(ARCHDIR)/$(THISDIR); make clean

soclean :
	@ echo "Making  ALL of Datatypes .so Clean in Architecture Specific directory [$(ARCH)]"
	@ cd $(ARCHDIR)/$(THISDIR); make soclean

doit : Makefile
	@ echo "Making ALL of Datatypes in Architecture Specific directory [$(ARCH)]"
	@ cd $(ARCHDIR)/$(THISDIR); make

quick : 
	@ echo "Making ALL of Datatypes in Architecture Specific directory [$(ARCH)]"
	@ cd $(ARCHDIR)/$(THISDIR); smake
