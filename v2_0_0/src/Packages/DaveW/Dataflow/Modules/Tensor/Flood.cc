/*
 *  Flood.cc:  Generate Flood from a field...
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/General/SegFld.h>
#include <Packages/DaveW/Core/Datatypes/General/TensorFieldPort.h>
#include <Packages/DaveW/Core/Datatypes/General/TensorField.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Dataflow/Widgets/PointWidget.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomSwitch.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MusilRNG.h>
#include <Core/Math/Trig.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
using std::cerr;
#include <values.h>

namespace DaveW {
using namespace DaveW;
using namespace SCIRun;

#define CURVMIN 1.4

class Flood : public Module {
    TensorFieldIPort* itport;
    ScalarFieldOPort* ofport;
    GeometryOPort* ogport;

    int initialized;
    virtual void geom_release(GeomPick*, void*);

    GuiInt nsteps;
    int abort;
    GuiDouble stepsize;
    GuiDouble seed_x;
    GuiDouble seed_y;
    GuiDouble seed_z;
    int cx, cy, cz;
    Point seed;
    ScalarFieldHandle sfH;
    Array3<double> curvMax;
    Array3<int> cell_status;
    Array1<tripleInt> active_cell;
    clString msg;

    int seedMoved;
    PointWidget *pw;
    CrowdMonitor widget_lock;
public:
    Flood(const clString& id);
    virtual ~Flood();
    virtual void execute();
    void tcl_command( TCLArgs&, void * );
};

extern "C" Module* make_Flood(const clString& id)
{
    return scinew Flood(id);
}

static clString module_name("Flood");

// dC/dt = div D dC/dx
// C is an array of 27 double for the neighborhood about point x
// Gordon computed the 3x3x3 mask D_mask that's a function of D --
//    we just compute D_mask dot C to get dC/dt
// (ala the Watkin Reaction Diffusion SIGGRAPH texture generation paper)

double concentration_diff(Array1<double> &D, Array1<double> &C, 
			  Array1<double> &D_mask) {
    D_mask[1]  =  D[4]/2;	// a23/2
    D_mask[3]  =  D[2]/2; 	// a13/2
    D_mask[4]  =  D[5];		// a33
    D_mask[5]  =  -D[2]/2; 	// -a13/2
    D_mask[7]  =  -D[4]/2;	// -a23/2

    D_mask[9]  =  D[1]/2;	// a12/2
    D_mask[10] =  D[3];		// a22
    D_mask[11] =  -D[1]/2;	// -a12/2
    D_mask[12] =  D[0];		// a11
    D_mask[13] =  -2*(D[0]+D[3]+D[5]);	// -2a11 -2a22 -2a33
    D_mask[14] =  D[0];		// a11
    D_mask[15] =  -D[1]/2;	// -a12/2
    D_mask[16] =  D[3];		// a22
    D_mask[17] =  D[1];		// a12/2

    D_mask[19] =  -D[4]/2;	// -a23/2
    D_mask[21] =  -D[2]/2; 	// -a13/2
    D_mask[22] =  D[5];		// a33
    D_mask[23] =  D[2]/2; 	// a13/2
    D_mask[25] =  D[4]/2;	// a23/2

    double dC =  0;
    for (int i=0; i<27; i++)
	dC += D_mask[i]*C[i];
//    return dC;
    return Max(0.,dC);
}
double laplacian(Array1<double> &C) {
    return ((C[4]+C[10]+C[12]+C[16]+C[22]/(4*C[14]))-1);
}

Flood::Flood(const clString& id)
: Module(module_name, id, Filter), initialized(0),
  nsteps("nsteps", id, this), stepsize("stepsize", id, this),
  seed_x("seed_x", id, this), seed_y("seed_y", id, this), 
  seed_z("seed_z", id, this), widget_lock("flood widget_lock")
{
    // Create the input ports
    itport=scinew TensorFieldIPort(this, "Tensor Field",
				    TensorFieldIPort::Atomic);
    add_iport(itport);
    // Create the output port
    ogport=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogport);
    ofport=scinew ScalarFieldOPort(this, "Flood", ScalarFieldIPort::Atomic);
    add_oport(ofport);
}

Flood::~Flood()
{
}

void Flood::execute()
{
    TensorFieldHandle tfh;
    TensorFieldBase* tfield;
    abort = 0;

    if(!itport->get(tfh) || !(tfield=tfh.get_rep())) return;
    TensorField<double>* tfd = dynamic_cast<TensorField<double>*>(tfield);
    if (!tfd) {
	cerr << "Error - Flood can only deal with TensorFields of doubles.\n";
	return;
    }

    ScalarFieldRG *sf;
    int nx, ny, nz;
    nx = tfd->m_tensor_field[0].dim1();
    ny = tfd->m_tensor_field[0].dim2();
    nz = tfd->m_tensor_field[0].dim3();
    if (!initialized) {

#if 0
	for (int i=0; i<nx; i++)
	    for (int j=0; j<ny; j++)
		for (int k=0; k<nz; k++) {
		    tfd->m_tensor_field[0](i,j,k)=1;
		    tfd->m_tensor_field[3](i,j,k)=9;
		    tfd->m_tensor_field[5](i,j,k)=25;
		    tfd->m_tensor_field[1](i,j,k)=
		    tfd->m_tensor_field[2](i,j,k)=
		    tfd->m_tensor_field[4](i,j,k)=0;
		}
#endif

	Point fmin, fmax, fctr;
	Vector fdiag;
	tfield->get_bounds(fmin, fmax);
	fdiag=(fmax-fmin);
	fctr=fmin+fdiag*.5;
	double fscale=fdiag.length();
	pw=scinew PointWidget(this, &widget_lock, fscale/200.);
	double px=seed_x.get();
	double py=seed_y.get();
	double pz=seed_z.get();
	if (px!=0 && py!=0 && pz!=0 &&
	    px>=fmin.x() && px<fmax.x() &&
	    py>=fmin.y() && py<fmax.y() &&
	    pz>=fmin.z() && pz<fmax.z())
	    fctr=Point(px,py,pz);
	pw->SetPosition(fctr);
	GeomObj *w=pw->GetWidget();
	ogport->addObj(w, clString("Point Widget"), &widget_lock);
	pw->Connect(ogport);
	initialized=1;
	msg="reset";
    }
    if (msg == "reset") {
	Point min, max;
	tfield->get_bounds(min, max);
	sf = new ScalarFieldRG;
	sf->resize(nx, ny, nz);
	sf->grid.initialize(0);
	sf->set_bounds(min, max);
	sfH = sf;
	cell_status.resize(nx, ny, nz);
	cell_status.initialize(0);
	curvMax.resize(nx, ny, nz);
	curvMax.initialize(0);
	active_cell.resize(0);
	seedMoved=1;
    } else
	sf=dynamic_cast<ScalarFieldRG*>(sfH.get_rep());
    if (seedMoved) {
	seed = pw->GetPosition();
	Point min, max;
	sf->get_bounds(min, max);
	sf->locate(seed, cx, cy, cz);
	int ch=0;
	if (cx<0) {ch=1; cx=0; seed.x(min.x()); }
	if (cx>=nx) {ch=1; cx=nx-1; seed.x(max.x()-0.00001); }
	if (cy<0) {ch=1; cy=0; seed.y(min.y()); }
	if (cy>=ny) {ch=1; cy=ny-1; seed.y(max.y()-0.00001); }
	if (cz<0) {ch=1; cz=0; seed.z(min.z()); }
	if (cz>=nz) {ch=1; cz=nz-1; seed.z(max.z()-0.00001); }
	if (ch) pw->SetPosition(seed);
	sf->grid(cx,cy,cz)=255;
	if (cell_status(cx,cy,cz) != 1) {
	    cell_status(cx,cy,cz)=1;
	    active_cell.add(tripleInt(cx,cy,cz));
	}
    }
    msg="";

    int a, ii, jj, kk, idx, q;
    tripleInt c;
    Array1<double> C_mask, D_mask;
    C_mask.resize(27);
    D_mask.resize(27);
    D_mask.initialize(0);
    Array1<double> dC;
    double dt = stepsize.get();
    int niters = nsteps.get();
    Array1<double> D(6);
    Array1<int> remove_list;
    for (int iter=0; iter<niters && !abort; iter++) {
	// visit each active cell and compute its new concentration
//	if (iter && !(iter%10)) 
	cerr << "Flood iteration: "<<iter<<"\n";
//	cerr << "iter="<<iter<<" active_cell.size()="<<active_cell.size()<<"\n";
	    
	remove_list.resize(0);
	dC.resize(active_cell.size());
	for (q=0; q<dC.size(); q++) {
	    c=active_cell[q];
	    idx=0;
	    Array1<tripleInt> newTriples;
	    int oldACsize=active_cell.size();
	    for (ii=c.x-1; ii<=c.x+1; ii++)
		for (jj=c.y-1; jj<=c.y+1; jj++)
		    for (kk=c.z-1; kk<=c.z+1; kk++, idx++)
			if (ii>=0 && jj>=0 && kk>=0 && 
			    ii<nx && jj<ny && kk<nz &&
			    cell_status(ii,jj,kk) != -1) {
			    C_mask[idx] = sf->grid(ii,jj,kk);
			    if (cell_status(ii,jj,kk) == 0) {
				active_cell.add(tripleInt(ii,jj,kk));
				cell_status(ii,jj,kk) = 2;
				newTriples.add(tripleInt(ii,jj,kk));
			    }
			} else C_mask[idx] = sf->grid(c.x,c.y,c.z);
	    for (a=0; a<6; a++) 
		D[a] = tfd->m_tensor_field[a](c.x,c.y,c.z);
	    dC[q] = concentration_diff(D, C_mask, D_mask)*dt;
	    double curvature=fabs(laplacian(C_mask));
	    curvature=curvMax(c.x,c.y,c.z)=Max(curvature, 
					       curvMax(c.x,c.y,c.z));
	    if (iter>3 && !(iter%3) && (curvature<CURVMIN)) {
		active_cell.resize(oldACsize);
		for (ii=0; ii<newTriples.size(); ii++) {
		    tripleInt t=newTriples[ii];
		    cell_status(t.x,t.y,t.z)=0;
		}
		remove_list.add(q);
	    }
#if 0
	    if (iter < 0) {
		cerr << "iter="<<iter<<" ("<<i<<","<<j<<","<<k<<") - C="<<C(i,j,k)<<" dC="<<dC(i,j,k)<<" dt="<<dt;
		cerr << "\n    D=";
		int p;
		for (p=0; p<6; p++)
		    cerr << D[p]<<" ";
		cerr << "\n    C=";
		for (p=0; p<27; p++)
		    cerr << C[p]<<" ";
		cerr << "\n    D_mask=";
		for (p=0; p<27; p++)
		    cerr << D_mask[p]<<" ";
		cerr << "\n";
	    }
#endif
	}

	// cell_status: -1 -> deactivated; 0 -> dormant; 1 -> in; 2 -> border

	// update new C values (sf)
	Array1<tripleInt> newCells;
	for (q=0; q<dC.size(); q++) {
	    c=active_cell[q];
	    sf->grid(c.x,c.y,c.z) += dC[q];
	}
	for (; q<active_cell.size(); q++) {
	    c=active_cell[q];
	    cell_status(c.x,c.y,c.z) = 1;
	}

	cerr << "Inactivating "<<remove_list.size()<<" of "<<active_cell.size()<<" cells.\n";

	for (q=0; q<remove_list.size(); q++) {
	    a=remove_list[q];
	    c=active_cell[a];
	    cell_status(c.x,c.y,c.z) = -1;
	    sf->grid(c.x,c.y,c.z) = -0.1;
	}
	
	int full_list_idx=0;
	int new_list_idx=0;
	int skip_idx=0;
	int skip=0;
	int old_full_list_size=active_cell.size();
	remove_list.add(old_full_list_size);
	while (full_list_idx<old_full_list_size) {
	    skip_idx=remove_list[skip];
	    while(full_list_idx<skip_idx) {
		active_cell[new_list_idx]=active_cell[full_list_idx];
		new_list_idx++;
		full_list_idx++;
	    }
	    skip++;
	    full_list_idx++;
	}
	active_cell.resize(new_list_idx);

	sf->grid(cx,cy,cz)=255;

	if (abort || iter==(niters-1)) {
	    ofport->send(sfH);
	    break;
	} else {
	    ofport->send_intermediate(sfH);
	}
    }
}

void Flood::geom_release(GeomPick*, void*) {
    seedMoved=1;
}

void Flood::tcl_command(TCLArgs& args, void* userdata) {
    if (args[1] == "set_seed") {
	if (pw) {
	    seed = pw->GetPosition();
	    seed_x.set(seed.x());
	    seed_y.set(seed.y());
	    seed_z.set(seed.z());
	} else {
	    cerr << "Error - can't set points, since we don't have a widget yet!\n";
	}
    } else if (args[1] == "get_seed") {
	if (pw) {
	    seed.x(seed_x.get());
	    seed.y(seed_y.get());
	    seed.z(seed_z.get());
	    seedMoved=1;
	    want_to_execute();
	} else {
	    cerr << "Error - can't set points, since we don't have a widget yet!\n";	    
	}
    } else if (args[1] == "reset") {
	msg="reset";
	want_to_execute();
    } else if (args[1] == "break") {
	abort = 1;
    } else {
        Module::tcl_command(args, userdata);
    }
}
} // End namespace DaveW


