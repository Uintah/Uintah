THISDIR = Thread

SECTION = SCICore

include $(TOP)/make.defaults

all : doit

clean :
	@ echo "Making Clean in Architecture Specific directory [$(ARCH)]"
	@ cd $(ARCHDIR)/$(THISDIR); \
	make clean -f $(TOP)/src/$(SECTION)/$(THISDIR)/Makefile.in

soclean :
	@ echo "Making .so Clean in Architecture Specific directory [$(ARCH)]"
	@ cd $(ARCHDIR)/$(THISDIR);\
	 make soclean -f $(TOP)/src/$(SECTION)/$(THISDIR)/Makefile.in

doit : Makefile
	@ echo "Making in Architecture Specific directory [$(ARCH)]"
	@ cd $(ARCHDIR)/$(THISDIR);\
          make -f $(TOP)/src/$(SECTION)/$(THISDIR)/Makefile.in

quick : 
	@ echo "Making in Architecture Specific directory [$(ARCH)]"
	@ cd $(ARCHDIR)/$(THISDIR); \
	  smake $(SMAKEJFLAG) -f $(TOP)/src/$(SECTION)/$(THISDIR)/Makefile.in

depend:
	@ echo "Making in Architecture Specific directory [$(ARCH)]"
	make depend -f Makefile.in


