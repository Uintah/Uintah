//static char *id="@(#) $Id$";

/*
 *  DipoleInSphere: User gives us discretization of a surface and a dipole,
 *          we compute potentials
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <DaveW/ThirdParty/NumRec/plgndr.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Datatypes/SurfTree.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Persistent/Pstreams.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <iostream>
using std::cerr;
#include <stdio.h>
#include <math.h>

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using namespace SCICore::Geometry;

class DipoleInSphere : public Module {
    SurfaceIPort* isurf;
    MatrixIPort* imat;
    SurfaceOPort* osurf;
    TCLstring methodTCL;
public:
    DipoleInSphere(const clString& id);
    void compute_three_sphere_potentials(const Array1<double>&, TriSurface *);
    void compute_one_sphere_potentials(const Array1<double>&, TriSurface *ts);
    void compute_infinite_medium_potentials(const Array1<double>& , 
					    TriSurface *ts);
    virtual ~DipoleInSphere();
    virtual void execute();
};

Module* make_DipoleInSphere(const clString& id)
{
    return new DipoleInSphere(id);
}

DipoleInSphere::DipoleInSphere(const clString& id)
: Module("DipoleInSphere", id, Filter), methodTCL("methodTCL", id, this)
{
    isurf=new SurfaceIPort(this, "SurfIn", SurfaceIPort::Atomic);
    add_iport(isurf);
    imat=new MatrixIPort(this, "Dipoles", MatrixIPort::Atomic);
    add_iport(imat);

    // Create the output port
    osurf=new SurfaceOPort(this, "SurfOut", SurfaceIPort::Atomic);
    add_oport(osurf);
}

DipoleInSphere::~DipoleInSphere()
{
}

void DipoleInSphere::compute_one_sphere_potentials(const Array1<double>& dip, 
						   TriSurface *ts) {
    double gamma=1;
    double R=1;
    double E[3];
    for (int i=0;i<ts->points.size();i++) {
	double V = 0.0;
	E[0] = ts->points[i].x();
	E[1] = ts->points[i].y();
	E[2] = ts->points[i].z();

	double rho = sqrt( pow((E[0] - dip[0]),2) + pow((E[1] - dip[1]),2) + pow((E[2] - dip[2]),2));
	double S = E[0]*dip[0] + E[1]*dip[1] + E[2]*dip[2];

	for(int k=0;k<3;k++) {
	    double F[3];
	    F[k] = (1/(4*M_PI*gamma*rho)) * 
		(2*(E[k]-dip[k])/pow(rho,2) +
		 (1/pow(R,2)) * (E[k] + (E[k]*S/R - R*dip[k])/(rho+R-S/R)));
	    V += F[k]*dip[k+3];
	}
	ts->bcVal[i]=V;
//	cerr << "Point: "<< ts->points[i]<<"  val: "<<V<<"\n";
    }
}

void DipoleInSphere::compute_infinite_medium_potentials(const Array1<double>& 
							dip, TriSurface *ts) {
}

void DipoleInSphere::compute_three_sphere_potentials(const Array1<double>& 
						      dip, TriSurface *ts) {
}

void DipoleInSphere::execute() {
    update_state(NeedData);

    SurfaceHandle sh;
    if (!isurf->get(sh))
	return;
    if (!sh.get_rep()) {
	cerr << "Error: empty surftree\n";
	return;
    }
    TriSurface *ts=dynamic_cast<TriSurface *> (sh.get_rep());
    if (!ts) {
	cerr << "Error: surface isn't a trisurface\n";
	return;
    }
    
    MatrixHandle mh;
    if (!imat->get(mh))
	return;
    if (!mh.get_rep()) {
	cerr << "Error: empty matrix\n";
	return;
    }
    if (mh->ncols() != 6) {
	cerr << "Error - dipoles must have 6 parameters.\n";
	return;
    }
    
    update_state(JustStarted);

    TriSurface *newTS = new TriSurface(*ts);

    newTS->bcVal.resize(newTS->points.size());
    newTS->bcIdx.resize(newTS->points.size());
    newTS->bcVal.initialize(0);
    int i;
    clString m=methodTCL.get();
    for (i=0; i<newTS->bcIdx.size(); i++) newTS->bcIdx[i]=i;
    for (i=0; i<mh->nrows(); i++) {
	Array1<double> dipole(6);
	for (int j=0; j<6; j++) dipole[j]=(*(mh.get_rep()))[i][j];
	// compute values here
	if (m=="OneSphere") {
	    compute_one_sphere_potentials(dipole, newTS);
	} else if (m=="InfiniteMedium") {
	    compute_infinite_medium_potentials(dipole, newTS);
	} else if (m=="ThreeSpheres") {
	    compute_three_sphere_potentials(dipole, newTS);
	} else {
	    cerr << "Error - I don't know method "<< m <<"\n";
	    return;
	}
    }
    SurfaceHandle sh2(newTS);
    osurf->send(sh2);
}    

} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.2  1999/10/07 02:06:31  sparker
// use standard iostreams and complex type
//
// Revision 1.1  1999/09/16 00:36:54  dmw
// added new Module that Chris Butson will work on (DipoleInSphere) and fixed SRCDIR references in DaveW makefiles
//
