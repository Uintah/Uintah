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

#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <SCICore/Datatypes/ScalarFieldRGdouble.h>
#include <SCICore/Datatypes/ScalarFieldRGfloat.h>
#include <SCICore/Datatypes/ScalarFieldRGint.h>
#include <SCICore/Datatypes/ScalarFieldRGshort.h>
#include <SCICore/Datatypes/ScalarFieldRGuchar.h>
#include <SCICore/Datatypes/ScalarFieldRGchar.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Geometry/Transform.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MiscMath.h>
#include <iostream>
using std::cerr;
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::GeomSpace;
using namespace SCICore::Math;
using namespace SCICore::Containers;
using SCICore::Geometry::Transform;

class TransformField : public Module {
    ScalarFieldIPort *iport;
    MatrixIPort *imat;
    ScalarFieldOPort *oport;
    void MatToTransform(MatrixHandle mH, Transform& t);
public:
    TransformField(const clString& id);
    virtual ~TransformField();
    virtual void execute();
};

extern "C" Module* make_TransformField(const clString& id) {
  return new TransformField(id);
}

TransformField::TransformField(const clString& id)
: Module("TransformField", id, Source)
{
    // Create the input port
    iport = scinew ScalarFieldIPort(this, "SFRG", ScalarFieldIPort::Atomic);
    add_iport(iport);
    imat = scinew MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
    add_iport(imat);
    oport = scinew ScalarFieldOPort(this, "SFRG",ScalarFieldIPort::Atomic);
    add_oport(oport);
}

TransformField::~TransformField()
{
}

void TransformField::MatToTransform(MatrixHandle mH, Transform& t) {
    double a[16];
    double *p=&(a[0]);
    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            *p++=(*mH.get_rep())[i][j];
    t.set(a);
}

void TransformField::execute()
{
    ScalarFieldHandle sfIH;
    iport->get(sfIH);
    if (!sfIH.get_rep()) return;
    ScalarFieldRGBase *sfrgb;
    if ((sfrgb=sfIH->getRGBase()) == 0) return;

    MatrixHandle mIH;
    imat->get(mIH);
    if (!mIH.get_rep()) return;
    if ((mIH->nrows() != 4) || (mIH->ncols() != 4)) return;
    Transform t;
    MatToTransform(mIH, t);

    ScalarFieldRGdouble *ifd, *ofd;
    ScalarFieldRGfloat *iff, *off;
    ScalarFieldRGint *ifi, *ofi;
    ScalarFieldRGshort *ifs, *ofs;
    ScalarFieldRGuchar *ifu, *ofu;
    ScalarFieldRGchar *ifc, *ofc;
    
    ScalarFieldRGBase *ofb;

    ifd=sfrgb->getRGDouble();
    iff=sfrgb->getRGFloat();
    ifi=sfrgb->getRGInt();
    ifs=sfrgb->getRGShort();
    ifu=sfrgb->getRGUchar();
    ifc=sfrgb->getRGChar();
    
    ofd=0;
    off=0;
    ofs=0;
    ofi=0;
    ofc=0;

    int nx=sfrgb->nx;
    int ny=sfrgb->ny;
    int nz=sfrgb->nz;
    Point min;
    Point max;
    sfrgb->get_bounds(min, max);
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
    } else if (ifu) {
	ofu=scinew ScalarFieldRGuchar(); 
	ofu->resize(nx,ny,nz);
	ofb=ofu;
    } else if (ifc) {
	ofc=scinew ScalarFieldRGchar(); 
	ofc->resize(nx,ny,nz);
	ofb=ofc;
    }
    ofb->set_bounds(Point(min.x(), min.y(), min.z()), 
		    Point(max.x(), max.y(), max.z()));
    for (int i=0; i<nx; i++)
	for (int j=0; j<ny; j++)
	    for (int k=0; k<nz; k++) {
                Point oldp(sfrgb->get_point(i,j,k));
                Point newp(t.unproject(oldp));
                double val=0;
		if (ifd) { 
		    ifd->interpolate(newp, val); 
		    ofd->grid(i,j,k)=val;
		} else if (iff) {
		    iff->interpolate(newp, val);
		    off->grid(i,j,k)=(float)val;
		} else if (ifi) {
		    ifi->interpolate(newp, val);
		    ofi->grid(i,j,k)=(int)val;
		} else if (ifs) {
		    ifs->interpolate(newp, val);
		    ofs->grid(i,j,k)=(short)val;
		} else if (ifu) {
		    ifu->interpolate(newp, val);
		    ofu->grid(i,j,k)=(unsigned char)val;
		} else if (ifi) {
		    ifc->interpolate(newp, val);
		    ofc->grid(i,j,k)=(char)val;
		}
	    }
    ScalarFieldHandle sfOH(ofb);
    oport->send(sfOH);
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.9  2000/03/17 09:27:01  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.8  2000/03/13 05:33:21  dmw
// Transforms are done the same way for ScalarFields, Surfaces and Meshes now - build the transform with the BldTransform module, and then pipe the output matrix into a Transform{Field,Surface,Mesh} module
//
// Revision 1.7  2000/02/08 21:45:28  kuzimmer
// stuff for transforming and type changes of scalarfieldRGs
//
// Revision 1.6  1999/10/07 02:06:49  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/25 03:47:49  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:47  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:44  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
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
