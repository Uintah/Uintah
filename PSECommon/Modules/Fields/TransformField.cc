//static char *id="@(#) $Id$";

/*
 *  TransformField.cc:  Rotate and flip field to get it into "standard" view
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   December 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/CoreDatatypes/ScalarFieldRGdouble.h>
#include <SCICore/CoreDatatypes/ScalarFieldRGfloat.h>
#include <SCICore/CoreDatatypes/ScalarFieldRGint.h>
#include <SCICore/CoreDatatypes/ScalarFieldRGshort.h>
#include <SCICore/CoreDatatypes/ScalarFieldRGuchar.h>
#include <PSECore/CommonDatatypes/ScalarFieldPort.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/TclInterface/TCL.h>
#include <tcl.h>
#include <tk.h>
#include <iostream.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Math;
using namespace SCICore::Containers;

class TransformField : public Module {
    ScalarFieldIPort *iport;
    ScalarFieldOPort *oport;
    ScalarFieldHandle sfOH;		// output bitFld
    ScalarFieldHandle last_sfIH;	// last input fld

    TCLstring xmap;
    TCLstring ymap;
    TCLstring zmap;

    int xxmap;
    int yymap;
    int zzmap;

    int tcl_execute;
public:
    TransformField(const clString& id);
    TransformField(const TransformField&, int deep);
    virtual ~TransformField();
    virtual Module* clone(int deep);
    virtual void execute();
    void set_str_vars();
    void tcl_command( TCLArgs&, void * );
    clString makeAbsMapStr();
};

Module* make_TransformField(const clString& id) {
  return new TransformField(id);
}

TransformField::TransformField(const clString& id)
: Module("TransformField", id, Source), tcl_execute(0),
  xmap("xmap", id, this), ymap("ymap", id, this), zmap("zmap", id, this),
  xxmap(1), yymap(2), zzmap(3)
{
    // Create the input port
    iport = scinew ScalarFieldIPort(this, "SFRG", ScalarFieldIPort::Atomic);
    add_iport(iport);
    oport = scinew ScalarFieldOPort(this, "SFRG",ScalarFieldIPort::Atomic);
    add_oport(oport);
}

TransformField::TransformField(const TransformField& copy, int deep)
: Module(copy, deep), tcl_execute(0),
  xmap("xmap", id, this), ymap("ymap", id, this), zmap("zmap", id, this),
  xxmap(1), yymap(2), zzmap(3)
{
    NOT_FINISHED("TransformField::TransformField");
}

TransformField::~TransformField()
{
}

Module* TransformField::clone(int deep)
{
    return scinew TransformField(*this, deep);
}

void TransformField::execute()
{
    ScalarFieldHandle sfIH;
    iport->get(sfIH);
    if (!sfIH.get_rep()) return;
    if (!tcl_execute && (sfIH.get_rep() == last_sfIH.get_rep())) return;
    ScalarFieldRGBase *sfrgb;
    if ((sfrgb=sfIH->getRGBase()) == 0) return;

    ScalarFieldRGdouble *ifd, *ofd;
    ScalarFieldRGfloat *iff, *off;
    ScalarFieldRGint *ifi, *ofi;
    ScalarFieldRGshort *ifs, *ofs;
    ScalarFieldRGuchar *ifc, *ofc;

    ScalarFieldRGBase *ofb;

    ifd=sfrgb->getRGDouble();
    iff=sfrgb->getRGFloat();
    ifi=sfrgb->getRGInt();
    ifs=sfrgb->getRGShort();
    ifc=sfrgb->getRGUchar();

    ofd=0;
    off=0;
    ofs=0;
    ofi=0;
    ofc=0;

    if (sfIH.get_rep() != last_sfIH.get_rep()) {	// new field came in
	int nx=sfrgb->nx;
	int ny=sfrgb->ny;
	int nz=sfrgb->nz;
	Point min;
	Point max;
	sfrgb->get_bounds(min, max);
	clString map=makeAbsMapStr();

	int basex, basey, basez, incrx, incry, incrz;
	if (xxmap>0) {basex=0; incrx=1;} else {basex=nx-1; incrx=-1;}
	if (yymap>0) {basey=0; incry=1;} else {basey=ny-1; incry=-1;}
	if (zzmap>0) {basez=0; incrz=1;} else {basez=nz-1; incrz=-1;}

 	if (map=="123") {
	    if (ifd) {
		ofd=scinew ScalarFieldRGdouble(); 
		ofd->resize(nx,ny,nz);
		ofb=ofd;
	    } else if (iff) {
		off=scinew ScalarFieldRGfloat(); 
		off->resize(nx,ny,nz);
		ofb=off;
	    } else if (ifi) {
		ofi=scinew ScalarFieldRGint(); 
		ofi->resize(nx,ny,nz);
		ofb=ofi;
	    } else if (ifs) {
		ofs=scinew ScalarFieldRGshort(); 
		ofs->resize(nx,ny,nz);
		ofb=ofs;
	    } else if (ifc) {
		ofc=scinew ScalarFieldRGuchar(); 
		ofc->resize(nx,ny,nz);
		ofb=ofc;
	    }
	    ofb->set_bounds(Point(min.x(), min.y(), min.z()), 
			    Point(max.x(), max.y(), max.z()));
	    for (int i=0, ii=basex; i<nx; i++, ii+=incrx)
		for (int j=0, jj=basey; j<ny; j++, jj+=incry)
		    for (int k=0, kk=basez; k<nz; k++, kk+=incrz)
			if (ofd) ofd->grid(i,j,k)=ifd->grid(ii,jj,kk);
			else if (off) off->grid(i,j,k)=iff->grid(ii,jj,kk);
			else if (ofi) ofi->grid(i,j,k)=ifi->grid(ii,jj,kk);
			else if (ofs) ofs->grid(i,j,k)=ifs->grid(ii,jj,kk);
			else if (ofc) ofc->grid(i,j,k)=ifc->grid(ii,jj,kk);
	} else if (map=="132") {
	    if (ifd) {
		ofd=scinew ScalarFieldRGdouble(); 
		ofd->resize(nx,nz,ny);
		ofb=ofd;
	    } else if (iff) {
		off=scinew ScalarFieldRGfloat(); 
		off->resize(nx,nz,ny);
		ofb=off;
	    } else if (ifi) {
		ofi=scinew ScalarFieldRGint(); 
		ofi->resize(nx,nz,ny);
		ofb=ofi;
	    } else if (ifs) {
		ofs=scinew ScalarFieldRGshort(); 
		ofs->resize(nx,nz,ny);
		ofb=ofs;
	    } else if (ifc) {
		ofc=scinew ScalarFieldRGuchar(); 
		ofc->resize(nx,nz,ny);
		ofb=ofc;
	    }
	    ofb->set_bounds(Point(min.x(), min.y(), min.z()), 
			    Point(max.x(), max.y(), max.z()));
	    for (int i=0, ii=basex; i<nx; i++, ii+=incrx)
		for (int j=0, jj=basey; j<ny; j++, jj+=incry)
		    for (int k=0, kk=basez; k<nz; k++, kk+=incrz)
			if (ofd) ofd->grid(i,j,k)=ifd->grid(ii,kk,jj);
			else if (off) off->grid(i,j,k)=iff->grid(ii,kk,jj);
			else if (ofi) ofi->grid(i,j,k)=ifi->grid(ii,kk,jj);
			else if (ofs) ofs->grid(i,j,k)=ifs->grid(ii,kk,jj);
			else if (ofc) ofc->grid(i,j,k)=ifc->grid(ii,kk,jj);
	} else if (map=="213") {
	    if (ifd) {
		ofd=scinew ScalarFieldRGdouble(); 
		ofd->resize(ny,nx,nz);
		ofb=ofd;
	    } else if (iff) {
		off=scinew ScalarFieldRGfloat(); 
		off->resize(ny,nx,nz);
		ofb=off;
	    } else if (ifi) {
		ofi=scinew ScalarFieldRGint(); 
		ofi->resize(ny,nx,nz);
		ofb=ofi;
	    } else if (ifs) {
		ofs=scinew ScalarFieldRGshort(); 
		ofs->resize(ny,nx,nz);
		ofb=ofs;
	    } else if (ifc) {
		ofc=scinew ScalarFieldRGuchar(); 
		ofc->resize(ny,nx,nz);
		ofb=ofc;
	    }
	    ofb->set_bounds(Point(min.x(), min.y(), min.z()), 
			    Point(max.x(), max.y(), max.z()));
	    for (int i=0, ii=basex; i<nx; i++, ii+=incrx)
		for (int j=0, jj=basey; j<ny; j++, jj+=incry)
		    for (int k=0, kk=basez; k<nz; k++, kk+=incrz)
			if (ofd) ofd->grid(i,j,k)=ifd->grid(jj,ii,kk);
			else if (off) off->grid(i,j,k)=iff->grid(jj,ii,kk);
			else if (ofi) ofi->grid(i,j,k)=ifi->grid(jj,ii,kk);
			else if (ofs) ofs->grid(i,j,k)=ifs->grid(jj,ii,kk);
			else if (ofc) ofc->grid(i,j,k)=ifc->grid(jj,ii,kk);
	} else if (map=="231") {
	    if (ifd) {
		ofd=scinew ScalarFieldRGdouble(); 
		ofd->resize(ny,nz,nx);
		ofb=ofd;
	    } else if (iff) {
		off=scinew ScalarFieldRGfloat(); 
		off->resize(ny,nz,nx);
		ofb=off;
	    } else if (ifi) {
		ofi=scinew ScalarFieldRGint(); 
		ofi->resize(ny,nz,nx);
		ofb=ofi;
	    } else if (ifs) {
		ofs=scinew ScalarFieldRGshort(); 
		ofs->resize(ny,nz,nx);
		ofb=ofs;
	    } else if (ifc) {
		ofc=scinew ScalarFieldRGuchar(); 
		ofc->resize(ny,nz,nx);
		ofb=ofc;
	    }
	    ofb->set_bounds(Point(min.x(), min.y(), min.z()), 
			    Point(max.x(), max.y(), max.z()));
	    for (int i=0, ii=basex; i<nx; i++, ii+=incrx)
		for (int j=0, jj=basey; j<ny; j++, jj+=incry)
		    for (int k=0, kk=basez; k<nz; k++, kk+=incrz)
			if (ofd) ofd->grid(i,j,k)=ifd->grid(ii,jj,kk);
			else if (off) off->grid(i,j,k)=iff->grid(jj,kk,ii);
			else if (ofi) ofi->grid(i,j,k)=ifi->grid(jj,kk,ii);
			else if (ofs) ofs->grid(i,j,k)=ifs->grid(jj,kk,ii);
			else if (ofc) ofc->grid(i,j,k)=ifc->grid(jj,kk,ii);
	} else if (map=="312") {
	    if (ifd) {
		ofd=scinew ScalarFieldRGdouble(); 
		ofd->resize(nz,nx,ny);
		ofb=ofd;
	    } else if (iff) {
		off=scinew ScalarFieldRGfloat(); 
		off->resize(nz,nx,ny);
		ofb=off;
	    } else if (ifi) {
		ofi=scinew ScalarFieldRGint(); 
		ofi->resize(nz,nx,ny);
		ofb=ofi;
	    } else if (ifs) {
		ofs=scinew ScalarFieldRGshort(); 
		ofs->resize(nz,nx,ny);
		ofb=ofs;
	    } else if (ifc) {
		ofc=scinew ScalarFieldRGuchar(); 
		ofc->resize(nz,nx,ny);
		ofb=ofc;
	    }
	    ofb->set_bounds(Point(min.x(), min.y(), min.z()), 
			    Point(max.x(), max.y(), max.z()));
	    for (int i=0, ii=basex; i<nx; i++, ii+=incrx)
		for (int j=0, jj=basey; j<ny; j++, jj+=incry)
		    for (int k=0, kk=basez; k<nz; k++, kk+=incrz)
			if (ofd) ofd->grid(i,j,k)=ifd->grid(kk,ii,jj);
			else if (off) off->grid(i,j,k)=iff->grid(kk,ii,jj);
			else if (ofi) ofi->grid(i,j,k)=ifi->grid(kk,ii,jj);
			else if (ofs) ofs->grid(i,j,k)=ifs->grid(kk,ii,jj);
			else if (ofc) ofc->grid(i,j,k)=ifc->grid(kk,ii,jj);
	} else if (map=="321") {
	    if (ifd) {
		ofd=scinew ScalarFieldRGdouble(); 
		ofd->resize(nz,ny,nx);
		ofb=ofd;
	    } else if (iff) {
		off=scinew ScalarFieldRGfloat(); 
		off->resize(nz,ny,nx);
		ofb=off;
	    } else if (ifi) {
		ofi=scinew ScalarFieldRGint(); 
		ofi->resize(nz,ny,nx);
		ofb=ofi;
	    } else if (ifs) {
		ofs=scinew ScalarFieldRGshort(); 
		ofs->resize(nz,ny,nx);
		ofb=ofs;
	    } else if (ifc) {
		ofc=scinew ScalarFieldRGuchar(); 
		ofc->resize(nz,ny,nx);
		ofb=ofc;
	    }
	    ofb->set_bounds(Point(min.x(), min.y(), min.z()), 
			    Point(max.x(), max.y(), max.z()));
	    for (int i=0, ii=basex; i<nx; i++, ii+=incrx)
		for (int j=0, jj=basey; j<ny; j++, jj+=incry)
		    for (int k=0, kk=basez; k<nz; k++, kk+=incrz)
			if (ofd) ofd->grid(i,j,k)=ifd->grid(kk,jj,ii);
			else if (off) off->grid(i,j,k)=iff->grid(kk,jj,ii);
			else if (ofi) ofi->grid(i,j,k)=ifi->grid(kk,jj,ii);
			else if (ofs) ofs->grid(i,j,k)=ifs->grid(kk,jj,ii);
			else if (ofc) ofc->grid(i,j,k)=ifc->grid(kk,jj,ii);

//	    outFld->resize(nz,ny,nx);
//	    outFld->grid(i,j,k)=sfrg->grid(kk,jj,ii);

	} else {
	    cerr << "ERROR: TransformField::execute() doesn't recognize map code: "<<map<<"\n";
	}
	sfOH=ofb;
    }
    oport->send(sfOH);
    tcl_execute=0;
}
	    
void TransformField::set_str_vars() {
    if (xxmap==1) xmap.set("x <- x+");
    if (xxmap==-1) xmap.set("x <- x-");
    if (xxmap==2) xmap.set("x <- y+");
    if (xxmap==-2) xmap.set("x <- y-");
    if (xxmap==3) xmap.set("x <- z+");
    if (xxmap==-3) xmap.set("x <- z-");
    if (yymap==1) ymap.set("y <- x+");
    if (yymap==-1) ymap.set("y <- x-");
    if (yymap==2) ymap.set("y <- y+");
    if (yymap==-2) ymap.set("y <- y-");
    if (yymap==3) ymap.set("y <- z+");
    if (yymap==-3) ymap.set("y <- z-");
    if (zzmap==1) zmap.set("z <- x+");
    if (zzmap==-1) zmap.set("z <- x-");
    if (zzmap==2) zmap.set("z <- y+");
    if (zzmap==-2) zmap.set("z <- y-");
    if (zzmap==3) zmap.set("z <- z+");
    if (zzmap==-3) zmap.set("z <- z-");
}

clString TransformField::makeAbsMapStr() {
    return to_string(Abs(xxmap))+to_string(Abs(yymap))+to_string(Abs(zzmap));
}

void TransformField::tcl_command(TCLArgs& args, void* userdata) {
    if (args[1] == "send") {
	tcl_execute=1;
	want_to_execute();
    } else if (args[1] == "flipx") {
	reset_vars();
	xxmap*=-1;
	set_str_vars();
	last_sfIH=0;
	want_to_execute();
    } else if (args[1] == "flipy") {
	reset_vars();
	yymap*=-1;
	set_str_vars();
	last_sfIH=0;
	want_to_execute();
    } else if (args[1] == "flipz") {
	reset_vars();
	zzmap*=-1;
	set_str_vars();
	last_sfIH=0;
	want_to_execute();
    } else if (args[1] == "cyclePos") {
	reset_vars();
	int tmp=xxmap;
	xxmap=yymap;
	yymap=zzmap;
	zzmap=tmp;
	set_str_vars();
	last_sfIH=0;
	want_to_execute();
    } else if (args[1] == "cycleNeg") {
	reset_vars();
	int tmp=zzmap;
	zzmap=yymap;
	yymap=xxmap;
	xxmap=tmp;
	set_str_vars();
	last_sfIH=0;
	want_to_execute();
    } else if (args[1] == "reset") {
	reset_vars();
	xxmap=1;
	yymap=2;
	zzmap=3;
	set_str_vars();
	last_sfIH=0;
	want_to_execute();
    } else if (args[1] == "swapXY") {
	reset_vars();
	int tmp=xxmap;
	xxmap=yymap;
	yymap=tmp;
	set_str_vars();
	last_sfIH=0;
	want_to_execute();
    } else if (args[1] == "swapYZ") {
	reset_vars();
	int tmp=yymap;
	yymap=zzmap;
	zzmap=tmp;
	set_str_vars();
	last_sfIH=0;
	want_to_execute();
    } else if (args[1] == "swapXZ") {
	reset_vars();
	int tmp=xxmap;
	xxmap=zzmap;
	zzmap=tmp;
	set_str_vars();
	last_sfIH=0;
	want_to_execute();
    } else {
        Module::tcl_command(args, userdata);
    }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.2  1999/08/17 06:37:30  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:44  mcq
// Initial commit
//
// Revision 1.2  1999/04/28 20:51:13  dav
// deleted some files that are dependent on DaveW files
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
