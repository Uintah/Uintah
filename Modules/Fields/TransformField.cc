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
#include <Geometry/BBox.h>
#include <Math/Expon.h>
#include <Math/MusilRNG.h>
#include <Math/Trig.h>
#include <TCL/TCLvar.h>
#include <iostream.h>
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

static Vector rotateVector(const Vector& v_r, double rm[][3]) {
    return Vector(v_r.x()*rm[0][0]+v_r.y()*rm[0][1]+v_r.z()*rm[0][2],
		  v_r.x()*rm[1][0]+v_r.y()*rm[1][1]+v_r.z()*rm[1][2],
		  v_r.x()*rm[2][0]+v_r.y()*rm[2][1]+v_r.z()*rm[2][2]);
}

// transform a pt by pushing it though a rotation matrix (M) and adding a
// displacement Vector (v)
static Point transformPt(double m[3][3], const Point& p, const Vector& v, double sx,
		  double sy, double sz) {
    return Point((p.x()*m[0][0]+p.y()*m[0][1]+p.z()*m[0][2])*sx+v.x(),
		 (p.x()*m[1][0]+p.y()*m[1][1]+p.z()*m[1][2])*sy+v.y(),
		 (p.x()*m[2][0]+p.y()*m[2][1]+p.z()*m[2][2])*sz+v.z());
}

class TransformField : public Module {
    ScalarFieldIPort *iport;
    ScalarFieldOPort *oport;
    TCLdouble rx, ry, rz, th;
    TCLdouble tx, ty, tz;
    TCLdouble scale, scalex, scaley, scalez;
    TCLint flipx, flipy, flipz;
    TCLint origin;
    ScalarFieldHandle osfh;
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
    int last_origin;
    int generation;
public:
    TransformField(const clString& id);
    TransformField(const TransformField&, int deep);
    virtual ~TransformField();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_TransformField(const clString& id)
{
    return scinew TransformField(id);
}
}

static clString module_name("TransformField");

TransformField::TransformField(const clString& id)
: Module("TransformField", id, Source),
  rx("rx", id, this), ry("ry", id, this), rz("rz", id, this), 
  th("th", id, this),
  tx("tx", id, this), ty("ty", id, this), tz("tz", id, this),
  scalex("scalex", id, this), scaley("scaley", id, this), 
  scalez("scalez", id, this), 
  scale("scale", id, this),
  flipx("flipx", id, this), flipy("flipy", id, this), 
  flipz("flipz", id, this), origin("origin", id, this),
  generation(-1), last_rx(0), last_ry(0), last_rz(0), last_th(0),
  last_tx(0), last_ty(0), last_tz(0), last_scale(0),	
  last_scalex(0), last_scaley(0), last_scalez(0),
  last_flipx(0), last_flipy(0), last_flipz(0), last_origin(0)
{
    // Create the input port
    iport = scinew ScalarFieldIPort(this, "SFRG", ScalarFieldIPort::Atomic);
    add_iport(iport);
    oport = scinew ScalarFieldOPort(this, "SFRG",ScalarFieldIPort::Atomic);
    add_oport(oport);
}

TransformField::TransformField(const TransformField& copy, int deep)
: Module(copy, deep),
  rx("rx", id, this), ry("ry", id, this), rz("rz", id, this),
  th("th", id, this),
  tx("tx", id, this), ty("ty", id, this), tz("tz", id, this),
  scalex("scalex", id, this), scaley("scaley", id, this), 
  scalez("scalez", id, this), 
  flipx("flipx", id, this), flipy("flipy", id, this), 
  flipz("flipz", id, this), origin("origin", id, this),
  scale("scale", id, this),
  generation(-1), last_rx(0), last_ry(0), last_rz(0), last_th(0),
  last_tx(0), last_ty(0), last_tz(0), last_scale(0),
  last_scalex(0), last_scaley(0), last_scalez(0), 
  last_flipx(0), last_flipy(0), last_flipz(0), last_origin(0)
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
    ScalarFieldRG* isfrg=sfIH->getRG();
    if (!isfrg) return;

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
    int orig=origin.get();

    if (generation == isfrg->generation && 
	rxx==last_rx && ryy==last_ry && rzz==last_rz && rtt==last_th &&
	txx==last_rx && tyy==last_ry && tzz==last_rz && 
	fx==last_flipx && fy==last_flipy && fz==last_flipz && 
	new_scale == last_scale && orig==last_origin) {
	oport->send(osfh);
	return;
    }

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
    last_origin = orig;
    Vector axis(rxx,ryy,rzz);
    if (!axis.length2()) axis.x(1);
    axis.normalize();

    // rotate about the center!

    if (fx) sx*=-1;
    if (fy) sy*=-1;
    if (fz) sz*=-1;

    Point min, max;
    isfrg->get_bounds(min, max);
    Point c(Interpolate(min, max, 0.5));
    Vector v(c.vector());
    Vector tr(txx,tyy,tzz);

    double m[3][3];
    buildRotateMatrix(m, rtt, axis);

    // make the new ScalarField here!
    oport->send(osfh);
    return;
}
