//static char *id="@(#) $Id$";

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

#include <sci_config.h>
#undef SCI_ASSERTION_LEVEL_3
#define SCI_ASSERTION_LEVEL_2
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array2.h>
#include <SCICore/Containers/Array3.h>
#include <SCICore/Util/Assert.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/Thread.h>
#include <stdio.h>

// just so I can see the proccess id...

#include <sys/types.h>
#include <unistd.h>

#define NUM_MATERIALS 5

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Containers;
using namespace SCICore::Thread;

class VectorSeg : public Module {
    Array1<ScalarFieldIPort*> ifields;
    Array1<ScalarFieldHandle> fields;
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
    int nx;
    int ny;
    int nz;
    clString myid;
    Array1<int> field_id;
    int have_ever_executed;
    void vector_seg_rg(const Array1<ScalarFieldHandle> &ifields, 
		       ScalarFieldRG* ofield);

    int np;
    void vec_seg();
    int isoChanged;
public:
    void parallel_vec_seg(int proc);
    VectorSeg(const clString&);
    virtual ~VectorSeg();
    virtual void connection(Module::ConnectionMode, int, int);
    virtual void execute();
};

Module* make_VectorSeg(const clString& id) {
  return new VectorSeg(id);
}

VectorSeg::VectorSeg(const clString& id)
: Module("VectorSeg", id, Filter), numFields("numFields", id, this),
  have_ever_executed(0)
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
	field_id.remove(field_id.size()-1);
	remove_iport(which_port);
	delete ifields[which_port];
	ifields.remove(which_port);
    } else {
	numFields.set(numFields.get()+1);
	field_id.add(0);
	ScalarFieldIPort* ci=scinew ScalarFieldIPort(this, "ScalarField", 
						ScalarFieldIPort::Atomic);
	add_iport(ci);
	ifields.add(ci);
    }
}
	
VectorSeg::~VectorSeg()
{
}

void VectorSeg::execute()
{
    fields.resize(ifields.size()-1);
    int flag;
    int i;
    for (flag=0, i=0; i<ifields.size()-1; i++) {
	if (!ifields[i]->get(fields[i]))
	    flag=1;
    }
    if (ifields.size()==1) return;	// NOTHING ATTACHED
    if (flag) return;			// one of the fields isn't ready
    ScalarFieldRG* rg;
    if (!fields[0].get_rep()) return;
    if (!(rg = fields[0]->getRG())) {
	NOT_FINISHED("Can't segment unstructured fields yet!");
	return;
    }
    nx=rg->nx;
    ny=rg->ny;
    nz=rg->nz;
    int old_gen=field_id[0];
    int new_field=((field_id[0]=rg->generation) != old_gen);
    for (flag=0, i=1; i<ifields.size()-1; i++) {
	ScalarFieldRG* rg_curr;
	if (!fields[i].get_rep()) return;
	if (!(rg_curr = fields[i]->getRG())) {
	    NOT_FINISHED("Can't segment unstructured fields yet!");
	    return;
	}
	old_gen=field_id[i];
	new_field &= ((field_id[i]=rg_curr->generation) != old_gen);
	if (nx != rg_curr->nx || ny != rg_curr->ny || nz != rg_curr->nz) {
	    cerr << "Can't segment from fields with different dimensions!\n";
	    return;
	}
    }
    if (!have_ever_executed || new_field) {
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

void VectorSeg::parallel_vec_seg(int proc) {
    int first_active_field=1;
    int sx=proc*(nx-1)/np;
    int ex=(proc+1)*(nx-1)/np;
    for (int f=0; f<numFields.get(); f++) {
	if (fld_sel[f]->get()) {
	    ScalarFieldRG* rg_in=fields[f]->getRG();
	    for (int x=sx; x<ex; x++) {
		for (int y=0; y<ny; y++) {
		    for (int z=0; z<nz; z++) {
			int inVal=(int)rg_in->grid(x,y,z);
			int outVal=(int)fieldRG->grid(x,y,z);
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
			fieldRG->grid(x,y,z)=outVal;
		    }
		}
	    }
	    first_active_field=0;
	}
    }
}

void VectorSeg::vec_seg()
{
    np=Thread::numProcessors();
    Thread::parallel(Parallel<VectorSeg>(this, &VectorSeg::parallel_vec_seg),
		     np, true);
}

void VectorSeg::vector_seg_rg(const Array1<ScalarFieldHandle> &, 
			      ScalarFieldRG* ofieldRG)
{
    //ScalarFieldRG* rg=ifields[0]->getRG();
    isoChanged=0;
    int fld_changed;
    int min_changed;
    int max_changed;
    int have_any=0;
    for (int f=0; f<numFields.get(); f++) {
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
	vec_seg();
    }
    ofieldRG->grid(0,0,0)=isoChanged;
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.6  1999/08/29 00:46:48  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.5  1999/08/25 03:48:11  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 05:30:54  sparker
// Configuration updates:
//  - renamed config.h to sci_config.h
//  - also uses sci_defs.h, since I couldn't get it to substitute vars in
//    sci_config.h
//  - Added flags for --enable-scirun, --enable-uintah, and
//    --enable-davew, to build the specific package set.  More than one
//    can be specified, and at least one must be present.
//  - Added a --enable-parallel, to build the new parallel version.
//    Doesn't do much yet.
//  - Made construction of config.h a little bit more general
//
// Revision 1.3  1999/08/18 20:20:12  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:54  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:17  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:58:02  dav
// updates in Modules for Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:34  dav
// Import sources
//
//
