THISDIR = Modules/Image

SECTION = SCIRun

include $(TOP)/make.defaults

all : doit

clean :
	@ echo "Making Clean in Architecture Specific directory [$(ARCH)]"
	@ cd $(ARCHDIR)/$(THISDIR); make clean

soclean :
	@ echo "Making .so Clean in Architecture Specific directory [$(ARCH)]"
	@ cd $(ARCHDIR)/$(THISDIR); make soclean

doit : Makefile
	@ echo "Making in Architecture Specific directory [$(ARCH)]"
	@ cd $(ARCHDIR)/$(THISDIR); make

quick : 
	@ echo "Making in Architecture Specific directory [$(ARCH)]"
	@ cd $(ARCHDIR)/$(THISDIR); smake
