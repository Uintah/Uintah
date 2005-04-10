//static char *id="@(#) $Id$";

/*
 *  Gradient.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <SCICore/Datatypes/ScalarFieldRGdouble.h>
#include <SCICore/Datatypes/ScalarFieldRGfloat.h>
#include <SCICore/Datatypes/ScalarFieldRGint.h>
#include <SCICore/Datatypes/ScalarFieldRGshort.h>
#include <SCICore/Datatypes/ScalarFieldRGchar.h>
// #include <SCICore/Datatypes/ScalarFieldRGuchar.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <SCICore/Datatypes/VectorFieldUG.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class Gradient : public Module {
    ScalarFieldIPort* infield;
    VectorFieldOPort* outfield;
public:
    TCLint interpolate;
    Gradient(const clString& id);
    virtual ~Gradient();
    virtual void execute();
};

extern "C" Module* make_Gradient(const clString& id) {
  return new Gradient(id);
}

Gradient::Gradient(const clString& id)
: Module("Gradient", id, Filter), interpolate("interpolate", id, this)
{
    infield=new ScalarFieldIPort(this, "Geometry", ScalarFieldIPort::Atomic);
    add_iport(infield);
    // Create the output port
    outfield=new VectorFieldOPort(this, "Geometry", VectorFieldIPort::Atomic);
    add_oport(outfield);
}

Gradient::~Gradient()
{
}

void Gradient::execute()
{
    ScalarFieldHandle sf;
    VectorFieldHandle vf;
    if(!infield->get(sf))
	return;
    ScalarFieldUG* sfug=sf->getUG();
    ScalarFieldRGBase* sfb=sf->getRGBase();
    if (sfug) {
	VectorFieldUG* vfug;
	if (sfug->typ == ScalarFieldUG::NodalValues) {
	    if(interpolate.get()){
		vfug=new VectorFieldUG(VectorFieldUG::NodalValues);
		vfug->mesh=sfug->mesh;
		vfug->data.resize(sfug->data.size());
		Mesh* mesh=sfug->mesh.get_rep();
		int nnodes=mesh->nodes.size();
		Array1<Vector>& gradients=vfug->data;
		int i;
		for(i=0;i<nnodes;i++)
		    gradients[i]=Vector(0,0,0);
		int nelems=mesh->elems.size();
		for(i=0;i<nelems;i++){
		    if(i%1000 == 0)
			update_progress(i, nelems);
		    Element* e=mesh->elems[i];
		    Point pt;
		    Vector grad1, grad2, grad3, grad4;
		    /*double vol=*/mesh->get_grad(e, pt, grad1, grad2, grad3, grad4);
		    double v1=sfug->data[e->n[0]];
		    double v2=sfug->data[e->n[1]];
		    double v3=sfug->data[e->n[2]];
		    double v4=sfug->data[e->n[3]];
		    Vector gradient(grad1*v1+grad2*v2+grad3*v3+grad4*v4);
		    for(int j=0;j<4;j++){
			gradients[e->n[j]]+=gradient;
		    }
		}
		for(i=0;i<nnodes;i++){
		    if(i%1000 == 0)
			update_progress(i, nnodes);
		    NodeHandle& n=mesh->nodes[i];
		    gradients[i]*=1./(n->elems.size());
		}
	    } else {
		vfug=new VectorFieldUG(VectorFieldUG::ElementValues);
		vfug->mesh=sfug->mesh;
		Mesh* mesh=sfug->mesh.get_rep();
		int nelems=mesh->elems.size();
		vfug->data.resize(nelems);
		for(int i=0;i<nelems;i++){
		    //	    if(i%10000 == 0)
		    //		update_progress(i, nelems);
		    Element* e=mesh->elems[i];
		    Point pt;
		    Vector grad1, grad2, grad3, grad4;
		    /*double vol=*/mesh->get_grad(e, pt, grad1, grad2, grad3, grad4);
		    double v1=sfug->data[e->n[0]];
		    double v2=sfug->data[e->n[1]];
		    double v3=sfug->data[e->n[2]];
		    double v4=sfug->data[e->n[3]];
		    Vector gradient(grad1*v1+grad2*v2+grad3*v3+grad4*v4);
		    vfug->data[i]=gradient;
		}
	    }
	} else {
	    cerr << "Gradient: I don't know how to take element-value gradients.\n";
	    return;
	}
	vf=vfug;
    } else {
	int nx=sfb->nx;
	int ny=sfb->ny;
	int nz=sfb->nz;
	VectorFieldRG *vfrg=new VectorFieldRG();
	vfrg->resize(nx, ny, nz);
	Point min, max;
	sfb->get_bounds(min, max);
	vfrg->set_bounds(min, max);
	ScalarFieldRGdouble *sfrd=sfb->getRGDouble();
	ScalarFieldRGfloat *sfrf=sfb->getRGFloat();
	ScalarFieldRGint *sfri=sfb->getRGInt();
	ScalarFieldRGshort *sfrs=sfb->getRGShort();
	ScalarFieldRGchar *sfrc=sfb->getRGChar();
//	ScalarFieldRGuchar *sfru=sfb->getRGUchar();
	for(int k=0;k<nz;k++){
	    update_progress(k, nz);
	    for(int j=0;j<ny;j++){
		for(int i=0;i<nx;i++){
		    if (sfrd)
			vfrg->grid(i,j,k)=sfrd->gradient(i,j,k);
		    else if (sfrf)
			vfrg->grid(i,j,k)=sfrf->gradient(i,j,k);
		    else if (sfri)
			vfrg->grid(i,j,k)=sfri->gradient(i,j,k);
		    else if (sfrs)
			vfrg->grid(i,j,k)=sfrs->gradient(i,j,k);
		    else if (sfrc)
			vfrg->grid(i,j,k)=sfrc->gradient(i,j,k);
//		    else if (sfru)
//			vfrg->grid(i,j,k)=sfru->gradient(i,j,k);
		    else {
			cerr << "Unknown SFRG type in Gradient: "<<sfb->getType()<<".\n";
			return;
		    }
		}
	    }
	}
	vf=vfrg;
    }
    outfield->send(vf);
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.8  2000/10/29 04:34:51  dmw
// BuildFEMatrix -- ground an arbitrary node
// SolveMatrix -- when preconditioning, be careful with 0's on diagonal
// MeshReader -- build the grid when reading
// SurfToGeom -- support node normals
// IsoSurface -- fixed tet mesh bug
// MatrixWriter -- support split file (header + raw data)
//
// LookupSplitSurface -- split a surface across a place and lookup values
// LookupSurface -- find surface nodes in a sfug and copy values
// Current -- compute the current of a potential field (- grad sigma phi)
// LocalMinMax -- look find local min max points in a scalar field
//
// Revision 1.7  2000/03/17 09:26:58  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.6  1999/10/07 02:06:48  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/25 03:47:47  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:44  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:40  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:28  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:42  mcq
// Initial commit
//
// Revision 1.2  1999/04/28 20:51:11  dav
// deleted some files that are dependent on DaveW files
//
// Revision 1.1.1.1  1999/04/24 23:12:32  dav
// Import sources
//
//
