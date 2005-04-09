
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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/DenseMatrix.h>
#include <Datatypes/Matrix.h>
#include <Datatypes/MatrixPort.h>
#include <Geometry/BBox.h>
#include <Math/Expon.h>
#include <Math/MusilRNG.h>
#include <Math/Trig.h>
#include <TCL/TCLvar.h>
#include <stdio.h>
#include <Malloc/Allocator.h>

static void buildRotateMatrix(double rm[][3], double angle, const Vector& axis) {
    // From Foley and Van Dam, Pg 227
    // NOTE: Element 0,1 is wrong in the text!
    double sintheta=Sin(angle);
    double costheta=Cos(angle);
    double ux=axis.x();
    double uy=axis.y();
    double uz=axis.z();
    rm[0][0]=ux*ux+costheta*(1-ux*ux);
    rm[0][1]=ux*uy*(1-costheta)-uz*sintheta;
    rm[0][2]=uz*ux*(1-costheta)+uy*sintheta;
    rm[1][0]=ux*uy*(1-costheta)+uz*sintheta;
    rm[1][1]=uy*uy+costheta*(1-uy*uy);
    rm[1][2]=uy*uz*(1-costheta)-ux*sintheta;
    rm[2][0]=uz*ux*(1-costheta)-uy*sintheta;
    rm[2][1]=uy*uz*(1-costheta)+ux*sintheta;
    rm[2][2]=uz*uz+costheta*(1-uz*uz);
}

class BldTransform : public Module {
    MatrixOPort* omatrix;
    MatrixHandle omh;
    TCLdouble rx, ry, rz, th;
    TCLdouble tx, ty, tz;
    TCLdouble scale, scalex, scaley, scalez;
    TCLint flipx, flipy, flipz;
    double last_rx;
    double last_ry;
    double last_rz;
    double last_th;
    double last_tx;
    double last_ty;
    double last_tz;
    double last_scale;
    double last_scalex;
    double last_scaley;
    double last_scalez;
    int last_flipx;
    int last_flipy;
    int last_flipz;
public:
    BldTransform(const clString& id);
    BldTransform(const BldTransform&, int deep);
    virtual ~BldTransform();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_BldTransform(const clString& id)
{
    return new BldTransform(id);
}
}

static clString module_name("BldTransform");

BldTransform::BldTransform(const clString& id)
: Module("BldTransform", id, Filter),
  rx("rx", id, this), ry("ry", id, this), rz("rz", id, this), 
  th("th", id, this),
  tx("tx", id, this), ty("ty", id, this), tz("tz", id, this),
  scalex("scalex", id, this), scaley("scaley", id, this), 
  scalez("scalez", id, this), 
  scale("scale", id, this),
  flipx("flipx", id, this), flipy("flipy", id, this), 
  flipz("flipz", id, this),
  last_rx(0), last_ry(0), last_rz(0), last_th(0),
  last_tx(0), last_ty(0), last_tz(0), last_scale(0),	
  last_scalex(0), last_scaley(0), last_scalez(0),
  last_flipx(0), last_flipy(0), last_flipz(0)
{
    // Create the output port
    omatrix=scinew MatrixOPort(this, "Matrix", MatrixIPort::Atomic);
    add_oport(omatrix);
}

BldTransform::BldTransform(const BldTransform& copy, int deep)
: Module(copy, deep),
  rx("rx", id, this), ry("ry", id, this), rz("rz", id, this),
  th("th", id, this),
  tx("tx", id, this), ty("ty", id, this), tz("tz", id, this),
  scalex("scalex", id, this), scaley("scaley", id, this), 
  scalez("scalez", id, this), 
  flipx("flipx", id, this), flipy("flipy", id, this), 
  flipz("flipz", id, this),
  scale("scale", id, this),
  last_rx(0), last_ry(0), last_rz(0), last_th(0),
  last_tx(0), last_ty(0), last_tz(0), last_scale(0),
  last_scalex(0), last_scaley(0), last_scalez(0), 
  last_flipx(0), last_flipy(0), last_flipz(0)
{
    NOT_FINISHED("BldTransform::BldTransform");
}

BldTransform::~BldTransform()
{
}

Module* BldTransform::clone(int deep)
{
    return new BldTransform(*this, deep);
}

void BldTransform::execute()
{
    double rxx=rx.get();
    double ryy=ry.get();
    double rzz=rz.get();
    double rtt=th.get();
    double txx=tx.get();
    double tyy=ty.get();
    double tzz=tz.get();
    double new_scale=scale.get();
    double s=pow(10.,new_scale);
    double new_scalex=scalex.get();
    double sx=pow(10.,new_scalex)*s;
    double new_scaley=scaley.get();
    double sy=pow(10.,new_scaley)*s;
    double new_scalez=scalez.get();
    double sz=pow(10.,new_scalez)*s;
    int fx=flipx.get();
    int fy=flipy.get();
    int fz=flipz.get();

    if (rxx==last_rx && ryy==last_ry && rzz==last_rz && rtt==last_th &&
	txx==last_rx && tyy==last_ry && tzz==last_rz && 
	fx==last_flipx && fy==last_flipy && fz==last_flipz && 
	new_scale == last_scale) {
	omatrix->send(omh);
	return;
    }
    
    DenseMatrix *dm=scinew DenseMatrix(4,4);
    omh=dm;

    last_rx = rxx;
    last_ry = ryy;
    last_rz = rzz;
    last_th = rtt;
    last_tx = txx;
    last_ty = tyy;
    last_tz = tzz;
    last_scale = new_scale;
    last_scalex = new_scalex;
    last_scaley = new_scaley;
    last_scalez = new_scalez;
    last_flipx = fx;
    last_flipy = fy;
    last_flipz = fz;
    Vector axis(rxx,ryy,rzz);
    if (!axis.length2()) axis.x(1);
    axis.normalize();

    // rotate about the center!

    if (fx) sx*=-1;
    if (fy) sy*=-1;
    if (fz) sz*=-1;

    Vector tr(txx,tyy,tzz);

    double m[3][3];
    buildRotateMatrix(m, rtt, axis);

    omatrix->send(omh);
    return;
}
