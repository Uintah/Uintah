
/*
 *  BldTransform.cc:  Build a 4x4 transformation matrix
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <PSECore/Datatypes/MatrixPort.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Geometry/Transform.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/Math/MusilRNG.h>
#include <SCICore/Math/Trig.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>
using std::cerr;
#include <stdio.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Geometry;
using namespace SCICore::TclInterface;

class BldTransform : public Module {
    MatrixIPort* imatrix;
    MatrixOPort* omatrix;
    MatrixHandle omh;
    TCLdouble rx, ry, rz, th;
    TCLdouble tx, ty, tz;
    TCLdouble scale, scalex, scaley, scalez;
    TCLdouble td, shu, shv;
    TCLint xmapTCL;
    TCLint ymapTCL;
    TCLint zmapTCL;
    TCLint pre;
    TCLint whichxform;
public:
    BldTransform(const clString& id);
    virtual ~BldTransform();
    virtual void execute();
};

Module* make_BldTransform(const clString& id)
{
    return new BldTransform(id);
}

BldTransform::BldTransform(const clString& id)
: Module("BldTransform", id, Filter),
  rx("rx", id, this), ry("ry", id, this), rz("rz", id, this), 
  th("th", id, this),
  tx("tx", id, this), ty("ty", id, this), tz("tz", id, this),
  scalex("scalex", id, this), scaley("scaley", id, this), 
  scalez("scalez", id, this), 
  scale("scale", id, this), pre("pre", id, this),
  xmapTCL("xmapTCL", id, this), ymapTCL("ymapTCL", id, this), 
  zmapTCL("zmapTCL", id, this),
  td("td", id, this), shu("shu", id, this), shv("shv", id, this),
  whichxform("whichxform", id, this)
{
    imatrix=scinew MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
    add_iport(imatrix);
    // Create the output port
    omatrix=scinew MatrixOPort(this, "Matrix", MatrixIPort::Atomic);
    add_oport(omatrix);
}

BldTransform::~BldTransform()
{
}

void BldTransform::execute()
{
    int wh=whichxform.get();
    int i, j;

    // get the input matrix if there is one
    MatrixHandle imh;
    Matrix* im;
    Transform inT;
    if (imatrix->get(imh) && (im=imh.get_rep())) {
	double inV[16];
	double *p=&(inV[0]);
	for (i=0; i<4; i++)
	    for (j=0; j<4; j++)
		*p++=(*im)[i][j];
	inT.set(inV);
    }

    // set up the local transform this module is building
    Transform locT;
    
    // get the "fixed point"
    double txx=tx.get();
    double tyy=ty.get();
    double tzz=tz.get();
    Vector t(txx, tyy, tzz);
    // switch on the message and build the local matrix accordingly
    if (wh==0) {			       // TRANSLATE
	locT.post_translate(t);
    } else if (wh==1) {                        // SCALE
	double new_scale=scale.get();
	double s=pow(10.,new_scale);
	double new_scalex=scalex.get();
	double sx=pow(10.,new_scalex)*s;
	double new_scaley=scaley.get();
	double sy=pow(10.,new_scaley)*s;
	double new_scalez=scalez.get();
	double sz=pow(10.,new_scalez)*s;
	Vector sc(sx, sy, sz);
	locT.post_translate(t);	
	cerr << "sc="<<sc<<"\n";
	locT.post_scale(sc);
	locT.post_translate(-t);
    } else if (wh==2) {			       // ROTATE
	Vector axis(rx.get(),ry.get(),rz.get());
	if (!axis.length2()) axis.x(1);
	axis.normalize();
	locT.post_translate(t);
	locT.post_rotate(th.get()*M_PI/180., axis);
	locT.post_translate(-t);
    } else if (wh==3) {      		       // SHEAR
	double shD=td.get();
	Vector shV(1,shu.get(),shv.get());
	locT.post_shear(t, shV, shD);
	printf("Here's the shear matrix:\n");
	locT.print();
    } else { // (wh==4)			       // PERMUTE
	if (pre.get())
	    locT.pre_permute(xmapTCL.get(), ymapTCL.get(), zmapTCL.get());
	else
	    locT.post_permute(xmapTCL.get(), ymapTCL.get(), zmapTCL.get());
    }

    DenseMatrix *dm=scinew DenseMatrix(4,4);
    omh=dm;
    
    // now either pre- or post-multiply the transforms and store in matrix
    double finalP[16];
    if (pre.get()) {
	locT.post_trans(inT);
	locT.get(finalP);
    } else {
	inT.post_trans(locT);
	inT.get(finalP);
    }

    double *p=&(finalP[0]);
    int cnt=0;
    for (i=0; i<4; i++) 
	for (j=0; j<4; j++, cnt++)
	    (*dm)[i][j]=*p++;

    dm->print();

    // send it and we're done
    omatrix->send(omh);
}

} // End namespace Modules
} // End namespace PSECommon


//
// $Log$
// Revision 1.3  2000/03/13 05:33:22  dmw
// Transforms are done the same way for ScalarFields, Surfaces and Meshes now - build the transform with the BldTransform module, and then pipe the output matrix into a Transform{Field,Surface,Mesh} module
//
// Revision 1.2  1999/10/07 02:06:51  sparker
// use standard iostreams and complex type
//
// Revision 1.1  1999/09/05 05:32:25  dmw
// updated and added Modules from old tree to new
//
