
/*
 *  VectorSeg.cc:  Segment field based on multiply valued elements
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <config.h>
#undef SCI_ASSERTION_LEVEL_3
#define SCI_ASSERTION_LEVEL_2
#include <Classlib/Array1.h>
#include <Classlib/Array2.h>
#include <Classlib/Array3.h>
#include <Classlib/Assert.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>
#include <stdio.h>

#define NUM_MATERIALS 5

class VectorSeg : public Module {
    Array1<ScalarFieldIPort*> ifields;
    ScalarFieldOPort* ofield;
    ScalarFieldHandle fieldHndl;
    ScalarFieldRG* fieldRG;
    Array1<int>	last_fld_sel;
    Array2<int> last_min;
    Array2<int> last_max;
    Array1<TCLint* > fld_sel;
    Array2<TCLint* > min;
    Array2<TCLint* > max;
    TCLint numFields;
    TCLint sameInput;
    int nx;
    int ny;
    int nz;
    clString myid;
    int have_ever_executed;
    void vector_seg_rg(const Array1<ScalarFieldHandle> &ifields, 
		       ScalarFieldRG* ofield);
public:
    VectorSeg(const clString&);
    VectorSeg(const VectorSeg&, int deep);
    virtual ~VectorSeg();
    virtual Module* clone(int deep);
    virtual void connection(Module::ConnectionMode, int, int);
    virtual void execute();
};

extern "C" {
Module* make_VectorSeg(const clString& id)
{
    return scinew VectorSeg(id);
}
};

VectorSeg::VectorSeg(const clString& id)
: Module("VectorSeg", id, Filter), numFields("numFields", id, this),
  have_ever_executed(0), sameInput("sameInput", id, this)
{
    myid=id;
    // Create the input port
    ifields.add(scinew ScalarFieldIPort(this, "ContourSet", 
				       ScalarFieldIPort::Atomic));

    add_iport(ifields[0]);
    ofield = scinew ScalarFieldOPort(this, "Surface", ScalarFieldIPort::Atomic); 
    add_oport(ofield);
    fieldRG=0;
}

void VectorSeg::connection(ConnectionMode mode, int which_port,
			       int output)
{
    if (output) return;
    if (have_ever_executed) {
	NOT_FINISHED("Can't add new ports after having executed... sorry!\n");
	return;
    }
    if (mode==Disconnected) {
	numFields.set(numFields.get()-1);
	remove_iport(which_port);
	delete ifields[which_port];
	ifields.remove(which_port);
    } else {
	numFields.set(numFields.get()+1);
	ScalarFieldIPort* ci=scinew ScalarFieldIPort(this, "ScalarField", 
						ScalarFieldIPort::Atomic);
	add_iport(ci);
	ifields.add(ci);
    }
}
	
VectorSeg::VectorSeg(const VectorSeg&copy, int deep)
: Module(copy, deep), numFields("numFields", id, this), 
  have_ever_executed(0), sameInput("sameInput", id, this)
{
    myid=id;
    NOT_FINISHED("VectorSeg::VectorSeg");
}

VectorSeg::~VectorSeg()
{
}

Module* VectorSeg::clone(int deep)
{
    return scinew VectorSeg(*this, deep);
}

void VectorSeg::execute()
{
    Array1<ScalarFieldHandle> fields(ifields.size()-1);
    for (int flag=0, i=0; i<ifields.size()-1; i++) {
	if (!ifields[i]->get(fields[i]))
	    flag=1;
    }
    if (ifields.size()==1) return;	// NOTHING ATTACHED
    if (flag) return;			// one of the fields isn't ready
    ScalarFieldRG* rg;
    if (!(rg = fields[0]->getRG())) {
	NOT_FINISHED("Can't segment unstructured fields yet!");
	return;
    }
    nx=rg->nx;
    ny=rg->ny;
    nz=rg->nz;
    for (flag=0, i=1; i<ifields.size()-1; i++) {
	ScalarFieldRG* rg_curr;
	if (!(rg_curr = fields[i]->getRG())) {
	    NOT_FINISHED("Can't segment unstructured fields yet!");
	    return;
	}
	if (nx != rg_curr->nx || ny != rg_curr->ny || nz != rg_curr->nz) {
	    cerr << "Can't segment from fields with different dimensions!\n";
	    return;
	}
    }
    if (!have_ever_executed || !sameInput.get()) {
	fieldHndl=fieldRG=0;
	fieldHndl=fieldRG=scinew ScalarFieldRG;
	last_min.newsize(numFields.get(), NUM_MATERIALS);
	last_max.newsize(numFields.get(), NUM_MATERIALS);
	last_fld_sel.resize(numFields.get());
	min.newsize(numFields.get(),NUM_MATERIALS);
	max.newsize(numFields.get(),NUM_MATERIALS);
	fld_sel.resize(numFields.get());
	fieldRG->resize(nx, ny, nz);
	Point old_min, old_max;
	rg->get_bounds(old_min, old_max);
	fieldRG->set_bounds(old_min, old_max);
	fieldRG->grid.initialize((1<<NUM_MATERIALS)-1);
	for (int fld=1; fld<=numFields.get(); fld++) {
	    clString fldName;
	    fldName = "f" + to_string(fld);
	    fld_sel[fld-1]=scinew TCLint(fldName, myid, this);
	    last_fld_sel[fld-1]=0;
	    for (int mat=1; mat<=NUM_MATERIALS; mat++) {
		clString minName;
		minName = "f" + to_string(fld)+ "m" + to_string(mat) + "min";
		min(fld-1,mat-1)=scinew TCLint(minName, myid, this);
		clString maxName;
		maxName = "f" + to_string(fld)+ "m" + to_string(mat) + "max";
		max(fld-1,mat-1)=scinew TCLint(maxName, myid, this);
	    }
	}
	have_ever_executed=1;
    }
    vector_seg_rg(fields, fieldRG);
    ofield->send(fieldRG);
}


void VectorSeg::vector_seg_rg(const Array1<ScalarFieldHandle> &ifields, 
			      ScalarFieldRG* ofieldRG) {
    ScalarFieldRG* rg=ifields[0]->getRG();
//    for (i=0; i<numFields.get(); i++) {
//	cout << "\nFIELD " << i << "\n";
//	printf("\nFIELD %d\n", i);
//	for (int j=0; j<NUM_MATERIALS; j++) {
//	    cout << "  Material " << j << ": min " << min(i,j)->get() << 
//		" max " << max(i,j)->get() << "\n";
//	    printf("  Material %d: min %d max %d\n", j, min(i,j)->get(),
//		   max(i,j)->get());
//	}
//   }
    int isoChanged=0;
    int fld_changed;
    int min_changed;
    int max_changed;
    for (int f=0, have_any=0; f<numFields.get(); f++) {
	int fs=fld_sel[f]->get();
	fld_changed = (fs!=last_fld_sel[f]);
	last_fld_sel[f]=fs;
	if (fs) have_any=1;
	for (int m=0; m<NUM_MATERIALS; m++) {
	    int mn=min(f,m)->get();
	    int mx=max(f,m)->get();
	    min_changed = (mn!=last_min(f,m));
	    max_changed = (mx!=last_max(f,m));
	    last_min(f,m)=mn;
	    last_max(f,m)=mx;
	    if (fld_changed || (fs && (min_changed || max_changed))){
		int bit=1<<m;
		isoChanged |= bit;
	    }
	}
    }
    if (!have_any) {
	ofieldRG->grid.initialize(0);
    } else if (isoChanged) {
	int first_active_field=1;
	for (f=0; f<numFields.get(); f++) {
	    if (fld_sel[f]->get()) {
		ScalarFieldRG* rg_in=ifields[f]->getRG();
		for (int x=0; x<nx; x++) {
		    for (int y=0; y<ny; y++) {
			for (int z=0; z<nz; z++) {
			    int inVal=rg_in->grid(x,y,z);
			    int outVal=ofieldRG->grid(x,y,z);
			    if (first_active_field) {
				outVal |= isoChanged;
			    }
			    for (int m=0; m<NUM_MATERIALS; m++) {
				int bit=1<<m;
				if ((bit & isoChanged) &&
				    ((outVal & bit) && 
				     (inVal < last_min(f,m) || 
				      inVal > last_max(f,m)))) {
				    outVal ^= bit;
				}
			    }
			    ofieldRG->grid(x,y,z)=outVal;
			}
		    }
		}
		first_active_field=0;
	    }
	}
    }
    ofieldRG->grid(0,0,0)=isoChanged;
}
