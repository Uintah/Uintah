//static char *id="@(#) $Id$";

/*
 *  Thermal.cc:  Always increasing!
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/ScalarFieldRGBase.h>
#include <SCICore/Datatypes/ScalarFieldRGchar.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <stdio.h>

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using namespace SCICore::Geometry;

class Thermal : public Module {
    ScalarFieldIPort* ifield;
    ScalarFieldOPort* ofield;
public:
    TCLint iters;
    TCLdouble scale;
    TCLint stickFlag;
    TCLint maxFlag;
    TCLdouble stickVal;
    TCLdouble maxVal;
    TCLdouble cond0, cond1, cond2, cond3, cond4, cond5, cond6;
    int tcl_exec;
    ScalarFieldHandle isfh;
    ScalarFieldHandle osfh;
    ScalarFieldRG *osf;
    ScalarFieldRG *temp;
    int igen;
    int offset;
    int heat_up;
    Thermal(const clString& id);
    virtual ~Thermal();
    virtual void execute();
    virtual void tcl_command(TCLArgs&, void*);
};

Module* make_Thermal(const clString& id)
{
    return new Thermal(id);
}

Thermal::Thermal(const clString& id)
: Module("Thermal", id, Filter), 
  iters("iters", id, this), scale("scale", id, this), igen(-1), osf(0), 
  temp(0), tcl_exec(0), stickFlag("stickFlag", id, this), 
  maxFlag("maxFlag", id, this), stickVal("stickVal", id, this), 
  maxVal("maxVal", id, this), cond0("cond0", id, this), 
  cond1("cond1", id, this), cond2("cond2", id, this), cond3("cond3", id, this),
  cond4("cond4", id, this), cond5("cond5", id, this), cond6("cond6", id, this),
  heat_up(0)
{
    ifield=new ScalarFieldIPort(this, "SFin", ScalarFieldIPort::Atomic);
    add_iport(ifield);
    // Create the output port
    ofield=new ScalarFieldOPort(this, "SFout", ScalarFieldIPort::Atomic);
    add_oport(ofield);
}

Thermal::~Thermal()
{
}

void Thermal::execute()
{
    Array1<double> conds;
    conds.resize(7);

    double sc=scale.get();
    conds[0]=cond0.get()*sc;		// air
    conds[1]=cond1.get()*sc;		// skin
    conds[2]=cond2.get()*sc;		// skull
    conds[3]=cond3.get()*sc;		// CSF
    conds[4]=cond4.get()*sc;		// gray matter
    conds[5]=cond5.get()*sc;		// white matter
    conds[6]=cond6.get()*sc;		// slurpy

    Array1<double> temps;
    temps.resize(7);
    temps[0]=37;
    temps[1]=37;
    temps[2]=37;
    temps[3]=37;
    temps[4]=37;
    temps[5]=37;
    temps[6]=-30;

    if (!ifield->get(isfh) || !isfh.get_rep())
	return;
    ScalarFieldRGBase* sfrgb = isfh->getRGBase();
    if (!sfrgb) {
	cerr << "Bad input -- need an RGBase.\n";
	return;
    }
    ScalarFieldRGchar* isf = sfrgb->getRGChar();
    if (!isf) {
	cerr << "Bad input -- need an RGchar.\n";
	return;
    }

    int nx=isf->nx;
    int ny=isf->ny;
    int nz=isf->nz;
    Point min, max;
    isf->get_bounds(min,max);
    Vector d=max-min;
    d.x((nx-1)/d.x());
    d.y((ny-1)/d.y());
    d.z((nz-1)/d.z());
    d.normalize();

    // should use the d's to scale the weights!

    Array3<double> w;
    w.newsize(3,3,3);
    w(0,0,0)=w(0,0,2)=w(0,2,0)=w(0,2,2)=w(2,0,0)=w(2,0,2)=w(2,2,0)=w(2,2,2)=
	.01;	// 8 * 0.01 = 0.08
    w(0,0,1)=w(0,1,0)=w(0,2,1)=w(0,1,2)=w(1,0,0)=w(1,0,2)=w(1,2,0)=w(1,2,2)=
	w(2,0,1)=w(2,1,0)=w(2,2,1)=w(2,1,2)=.015;	// 12 * 0.015 = 0.18
    w(0,1,1)=w(1,0,1)=w(1,1,0)=w(2,1,1)=w(1,2,1)=w(1,1,2)=.1; // 6 * 0.05 = 0.3
    w(1,1,1)=0.44;

    if (igen == isf->generation && !tcl_exec) return;

    ScalarFieldRG* t=osf;
    osf = new ScalarFieldRG;
    osf->resize(nx, ny, nz);
    osf->set_bounds(min, max);
    if (igen!=isf->generation || t->nx!=nx || t->ny!=ny || t->nz!=nz) {
	t=0;
	tcl_exec=0;
    }

    temp = new ScalarFieldRG;
    temp->resize(nx, ny, nz);
    temp->set_bounds(min, max);

    igen = isf->generation;
    
    // go through and set all the default temps
    
    if (isf->grid(0,0,0) > 7) offset=(int) '0'; else offset=0;
    for (int i=0; i<nx; i++)
	for (int j=0; j<ny; j++)
	    for (int k=0; k<nz; k++)
		if (t)
		    osf->grid(i,j,k)=t->grid(i,j,k);
		else
		    osf->grid(i,j,k)=temps[isf->grid(i,j,k)-offset];

    if (tcl_exec) {
	int i,j,k;
	int niters=iters.get();
	double wtotal;
	while(niters) {
	    niters--;
	    if (heat_up)
		for (i=0; i<nx; i++)
		    for (j=0; j<ny; j++)
			for (k=0; k<nz; k++) 
			    if (isf->grid(i,j,k) == 6) 
				osf->grid(i,j,k)=(50+osf->grid(i,j,k))/2.;
	    
	    // just to set up the boundaries!
	    
//	    for (i=0; i<nx; i++)
//		for (j=0; j<ny; j++)
//		    for (k=0; k<nz; k++)
//			temp->grid(i,j,k)=osf->grid(i,j,k);

	    // for each iteration, each voxel in temp gets a weighted average
	    // of the local nighborhood of osf

	    int sF=stickFlag.get();
	    int nbF=maxFlag.get();
	    double sV=stickVal.get();
	    double nbV=maxVal.get();
	    
	    for (i=0; i<nx; i++)
		for (j=0; j<ny; j++)
		    for (k=0; k<nz; k++) {
			if (sF && osf->grid(i,j,k)<=sV) {
			    temp->grid(i,j,k)=osf->grid(i,j,k);
			} else {
			    temp->grid(i,j,k)=0;
			    wtotal=0;
			    for (int ii=i-1; ii<=i+1; ii++)
				for (int jj=j-1; jj<=j+1; jj++)
				    for (int kk=k-1; kk<=k+1; kk++) {
					if (ii<0 || ii>=nx || jj<0 || jj>=ny
					    || kk<0 || kk>=nz) continue;
//					wtotal+=w(ii-i+1,jj-j+1,kk-k+1)*
//					    conds[isf->grid(ii,jj,kk)-offset];
//					temp->grid(i,j,k)+=osf->grid(ii,jj,kk)*
//					    w(ii-i+1,jj-j+1,kk-k+1)*
//					    conds[isf->grid(ii,jj,kk)-offset];

					// temp is gonna be 1 - my conductivity
					// times me, plus my neighbors

					if (ii!=i || jj!=j || kk!=k) { 
					    double ctb=
						conds[isf->grid(ii,jj,kk)-
						     offset]*
						w(ii-i+1,jj-j+1,kk-k+1);
					    wtotal+=ctb;
					    temp->grid(i,j,k)+=ctb*
						osf->grid(ii,jj,kk);
//					} else {
//					    wtotal+=w(ii-i+1,jj-j+1,kk-k+1);
//					    temp->grid(i,j,k)+=
//						osf->grid(ii,jj,kk)*
//						w(ii-i+1,jj-j+1,kk-k+1);
					}
				    }
			    // normalize the sum of the neighbors contributions
			    temp->grid(i,j,k)/=wtotal;
			    // linear interpolation of our value and our
			    // nbrs' values, based on our conductivity
			    temp->grid(i,j,k)=temp->grid(i,j,k)*
				conds[isf->grid(i,j,k)]+
				osf->grid(i,j,k)*
				(1-conds[isf->grid(i,j,k)]);

			    if (nbF && osf->grid(i,j,k)<=nbV &&
				temp->grid(i,j,k)>nbV)
				temp->grid(i,j,k)=nbV;
			}
		    }
	    ScalarFieldRG *dummy;
	    dummy=osf;
	    osf=temp;
	    temp=dummy;
	}
    }
    cerr << "osf="<<osf<<"\n";
    osfh=0;
    osfh=osf;
    ofield->send(osfh);
}

void Thermal::tcl_command(TCLArgs& args, void* userdata)
{
    if(args.count() < 2)
        {
            args.error("Thermal needs a minor command");
            return;
        }
    else if(args[1] == "reset")
        {
	    igen = -1;
	    tcl_exec=1;
            want_to_execute();
        }
    else if (args[1] == "tcl_exec") 
	{
	    tcl_exec=1;
	    want_to_execute();
	}
    else if (args[1] == "heat_on")
	{
	    heat_up=1;
//	    tcl_exec=1;
//	    want_to_execute();
	}
    else if (args[1] == "heat_off")
	{
	    heat_up=0;
//	    tcl_exec=1;
//	    want_to_execute();
	}
    else
        {
            Module::tcl_command(args, userdata);
        }
}

} // End namespace Modules
} // End namespace DaveW


//
// $Log$
// Revision 1.3  1999/09/08 02:26:25  sparker
// Various #include cleanups
//
// Revision 1.2  1999/08/25 03:47:40  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.1  1999/08/24 06:23:03  dmw
// Added in everything for the DaveW branch
//
// Revision 1.2  1999/05/03 04:52:15  dmw
// Added and updated DaveW Datatypes/Modules
//
//
