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

#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Malloc/Allocator.h>
#include <Math/MiscMath.h>
#include <TCL/TCLTask.h>
#include <TCL/TCLvar.h>
#include <TCL/TCL.h>
#include <tcl/tcl/tcl.h>
#include <tcl/tk/tk.h>
#include <iostream.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

class TransformField : public Module {
    ScalarFieldIPort *iport;
    ScalarFieldOPort *oport;
    ScalarFieldHandle sfOH;		// output bitFld
    ScalarFieldHandle last_sfIH;	// last input fld
    ScalarFieldRG* last_sfrg;		// just a convenience

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

extern "C" {
Module* make_TransformField(const clString& id)
{
    return scinew TransformField(id);
}
};

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
    ScalarFieldRG *sfrg;
    if ((sfrg=sfIH->getRG()) == 0) return;
    ScalarFieldRG *outFld;
    if (sfIH.get_rep() != last_sfIH.get_rep()) {	// new field came in
	int nx=sfrg->nx;
	int ny=sfrg->ny;
	int nz=sfrg->nz;
	Point min;
	Point max;
	sfrg->get_bounds(min, max);
	clString map=makeAbsMapStr();

	int basex, basey, basez, incrx, incry, incrz;
	if (xxmap>0) {basex=0; incrx=1;} else {basex=nx-1; incrx=-1;}
	if (yymap>0) {basey=0; incry=1;} else {basey=ny-1; incry=-1;}
	if (zzmap>0) {basez=0; incrz=1;} else {basez=nz-1; incrz=-1;}

 	if (map=="123") {
	    outFld = scinew ScalarFieldRG();
	    outFld->resize(nx,ny,nz);
	    outFld->set_bounds(Point(min.x(), min.y(), min.z()), 
			       Point(max.x(), max.y(), max.z()));
	    for (int i=0, ii=basex; i<nx; i++, ii+=incrx)
		for (int j=0, jj=basey; j<ny; j++, jj+=incry)
		    for (int k=0, kk=basez; k<nz; k++, kk+=incrz)
			outFld->grid(i,j,k)=sfrg->grid(ii,jj,kk);
	} else if (map=="132") {
	    outFld = scinew ScalarFieldRG();
	    outFld->resize(nx,nz,ny);
	    outFld->set_bounds(Point(min.x(), min.z(), min.y()), 
			       Point(max.x(), max.z(), max.y()));
	    for (int i=0, ii=basex; i<nx; i++, ii+=incrx)
		for (int j=0, jj=basey; j<ny; j++, jj+=incry)
		    for (int k=0, kk=basez; k<nz; k++, kk+=incrz)
			outFld->grid(i,j,k)=sfrg->grid(ii,kk,jj);
	} else if (map=="213") {
	    outFld = scinew ScalarFieldRG();
	    outFld->resize(ny,nx,nz);
	    outFld->set_bounds(Point(min.y(), min.x(), min.z()), 
			       Point(max.y(), max.x(), max.z()));
	    for (int i=0, ii=basex; i<nx; i++, ii+=incrx)
		for (int j=0, jj=basey; j<ny; j++, jj+=incry)
		    for (int k=0, kk=basez; k<nz; k++, kk+=incrz)
			outFld->grid(i,j,k)=sfrg->grid(jj,ii,kk);
	} else if (map=="231") {
	    outFld = scinew ScalarFieldRG();
	    outFld->resize(ny,nz,nx);
	    outFld->set_bounds(Point(min.y(), min.z(), min.x()), 
			       Point(max.y(), max.z(), max.x()));
	    for (int i=0, ii=basex; i<nx; i++, ii+=incrx)
		for (int j=0, jj=basey; j<ny; j++, jj+=incry)
		    for (int k=0, kk=basez; k<nz; k++, kk+=incrz)
			outFld->grid(i,j,k)=sfrg->grid(jj,kk,ii);
	} else if (map=="312") {
	    outFld = scinew ScalarFieldRG();
	    outFld->resize(nz,nx,ny);
	    outFld->set_bounds(Point(min.z(), min.x(), min.y()), 
			       Point(max.z(), max.x(), max.y()));
	    for (int i=0, ii=basex; i<nx; i++, ii+=incrx)
		for (int j=0, jj=basey; j<ny; j++, jj+=incry)
		    for (int k=0, kk=basez; k<nz; k++, kk+=incrz)
			outFld->grid(i,j,k)=sfrg->grid(kk,ii,jj);
	} else if (map=="321") {
	    outFld = scinew ScalarFieldRG();
	    outFld->resize(nz,ny,nx);
	    outFld->set_bounds(Point(min.z(), min.y(), min.x()), 
			       Point(max.z(), max.y(), max.x()));
	    for (int i=0, ii=basex; i<nx; i++, ii+=incrx)
		for (int j=0, jj=basey; j<ny; j++, jj+=incry)
		    for (int k=0, kk=basez; k<nz; k++, kk+=incrz)
			outFld->grid(i,j,k)=sfrg->grid(kk,jj,ii);
	} else {
	    cerr << "ERROR: TransformField::execute() doesn't recognize map code: "<<map<<"\n";
	}
	sfOH=outFld;
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
