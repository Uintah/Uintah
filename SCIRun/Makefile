THISDIR = SCIRun

SECTION = SCIRun

include $(TOP)/make.defaults

all : subdirs scirun

quick :
	@ make QUICK=quick

clean :
	rm *.o scirun
	rm -f  $(ARCHDIR)/Datatypes/*/*.o
	rm -fR $(ARCHDIR)/Datatypes/*/ii_files
	rm -f  $(ARCHDIR)/Modules/*/*.o
	rm -fR $(ARCHDIR)/Modules/*/ii_files
	rm -fR $(ARCHDIR)/lib/*.so
	rm -fR $(ARCHDIR)/lib/so_locations

soclean:
	rm -fR $(ARCHDIR)/lib/*.so
	rm -fR $(ARCHDIR)/lib/so_locations

scirun.o : scirun.cc
	$(CC) $(CFLAGS) -I../SCICore -I../PSECommon -c scirun.cc

scirun : scirun.o
	@ echo
	@ echo "Making SCIRun"
	@ echo
	$(CC) $(CFLAGS) scirun.o -o scirun \
	   -L../../$(ARCH)/SCIRun/lib \
	   -L../../$(ARCH)/PSECommon/lib \
	   -L../../$(ARCH)/SCICore/lib \
	   -L../../$(ARCH)/3rdParty/lib \
	   -lDatatypes \
	   -lCommonDatatypes -lCoreDatatypes \
	   -lConstraints \
	   -lWidgets -lDataflow -lComm \
	   -lTclInterface -lGeom -lPersistent -lDistributed \
	   -lGeometry -lContainers -lExceptions -lUtil \
	   -lMultitask -lMalloc \
	   -lGL -lXext -lX11 -ltcl -ltk -litk -lm

subdirs :
	@ echo
	@ echo "Making SCIRun Datatypes"
	@ cd Datatypes; make $(QUICK)
	@ echo
	@ echo
	@ echo "Making SCIRun Modules"
	@ cd Modules; make $(QUICK)
	@ echo
