//static char *id="@(#) $Id$";

/*
 *  Coregister.cc:  Coregister a set of points to a surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1996
 *
 *  Copyright (C) 1996 SCI Group
 *
 *  Notes:  possible improvements -- Newton method for guiding convergence!!!
 *				     simulated annealing for convergence (??)
 */

#include <DaveW/Datatypes/General/ManhattanDist.h>
#include <DaveW/Datatypes/General/ScalarTriSurface.h>
#include <DaveW/ThirdParty/Quaternions/BallAux.h>
#include <DaveW/ThirdParty/NumRec/dsvdcmp.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <PSECore/Widgets/ScaledBoxWidget.h>
#include <SCICore/Datatypes/BasicSurfaces.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/SurfTree.h>
#include <SCICore/Datatypes/TriSurface.h>
#include <SCICore/Geom/Pt.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/GeomSphere.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/Math/Mat.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Math/Trig.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Util/NotFinished.h>

#include <iostream.h>

namespace DaveW {
namespace Modules {

using DaveW::Datatypes::ManhattanDist;
using DaveW::Datatypes::ScalarTriSurface;

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace PSECore::Widgets;

using namespace SCICore::TclInterface;
using namespace SCICore::Containers;
using namespace SCICore::Geometry;
using namespace SCICore::GeomSpace;
using namespace SCICore::Math;

double FiducialFit(double ax[], double ay[], double az[], double px[], 
		   double py[], double pz[], double TT[4][4], int useLSF,
		   int useScale);

class Coregister : public Module {
    SurfaceIPort* iport_scalp;
    SurfaceIPort* iport_pts;
    SurfaceOPort* oport_pts;
    GeomPts* geomPts;
    GeometryOPort* owidget;
    GeomPick* gpk;
    int left_id;
    int right_id;
    int nasion_id;
    Point left, left_dig;
    Point nasion, nasion_dig;
    Point right, right_dig;
    Array1<Point> sclpPts;
    CrowdMonitor widget_lock;
    ScaledBoxWidget *widget;
    int initialized;
    int widgetMoved;
    int autoRegister;
    Vector w_v;
    Point w_c;
    double long_edge;
    SurfaceHandle old_pts;
    Array1<Point> orig_pts;
    Array1<Vector> orig_vec;
    Array1<Point> trans_pts;
    Array1<Vector> trans_vec;
    SurfaceHandle old_scalp;
    SurfaceHandle osh;
    ManhattanDist *mgd;
    TCLdouble rot_r_x, rot_r_y, rot_r_z;
    TCLdouble rot_d_x, rot_d_y, rot_d_z;
    TCLdouble rot_i_x, rot_i_y, rot_i_z;
    TCLdouble trans_x, trans_y, trans_z;
    TCLdouble reg_error;
    TCLdouble scale;
    TCLint abortButton;
    TCLstring curr_iter;
    TCLstring iters;
    TCLstring percent;
    TCLstring transform;
    TCLstring fiducial;
    TCLstring fiducialMethod;
    TCLint useFirstSurfPts;
    TCLint useScale;
    MaterialHandle matl;
    MaterialHandle navy;		// for nasion
    MaterialHandle lavender;		// for left maxillary
    MaterialHandle orange;		// for right maxillary
    int recenter;
    int pickPts_id;
    clString send_fid;
    int ptCloudFids_id;
    int ptCloudFidsFlag;
    int force_redraw;
    int fiducialFit;
public:
    Coregister(const clString& id);
    virtual ~Coregister();
    virtual void execute();
    virtual void widget_moved(int last);    
    virtual void geom_pick(GeomPick*, void*);
    virtual void tcl_command(TCLArgs&, void*);
    double computeError(int perc=100);
    void auto_register();
    void transform_pts(const Point& w_c, const Point& p,
		       const Vector& vr, const Vector& vd, 
		       const Vector& vi, int perc=100);
    void makeMGD(int full);
    void makeBaseTransSurf(SurfaceHandle);
    void setTransSurfPts();
    double findMin(double, double, double, double*);
    Point unproject_point(const Point &p, const Vector &, const Vector &,
			  const Vector &, const Point &);
};

Module* make_Coregister(const clString& id)
{
    return new Coregister(id);
}

static clString module_name("Coregister");

Coregister::Coregister(const clString& id)
: Module("Coregister", id, Filter), widget_lock("Coregister widget lock"),
    reg_error("reg_error", id, this),
  rot_r_x("rot_r_x",id,this),rot_r_y("rot_r_y",id,this),
  rot_r_z("rot_r_z",id,this),rot_d_x("rot_d_x",id,this),
  rot_d_y("rot_d_y",id,this),rot_d_z("rot_d_z",id,this),
  rot_i_x("rot_i_x",id,this),rot_i_y("rot_i_y",id,this),
  rot_i_z("rot_i_z",id,this),trans_x("trans_x",id,this),
  trans_y("trans_y",id,this),trans_z("trans_z",id,this),
  scale("scale", id, this), iters("iters",id,this),
  abortButton("abortButton",id,this), curr_iter("curr_iter",id,this), 
  percent("percent",id,this), recenter(1), transform("transform", id, this),
  fiducial("fiducial",id,this), gpk(0), left_id(0), nasion_id(0), right_id(0),
  fiducialMethod("fiducialMethod",id,this),
  useFirstSurfPts("useFirstSurfPts", id, this), useScale("useScale", id, this)
{
    // Create the input ports
    iport_scalp=new SurfaceIPort(this, "Scalp", SurfaceIPort::Atomic);
    add_iport(iport_scalp);
    iport_pts=new SurfaceIPort(this, "Points", SurfaceIPort::Atomic);
    add_iport(iport_pts);
    // Create the output port
    owidget=new GeometryOPort(this, "Widget", GeometryIPort::Atomic);
    add_oport(owidget);
    oport_pts=new SurfaceOPort(this, "OPoints", SurfaceIPort::Atomic);
    add_oport(oport_pts);
    widget=scinew ScaledBoxWidget(this, &widget_lock, 0.2);
    widget->SetCurrentMode(2);
    geomPts=0;
    mgd=0;
    initialized=0;
    widgetMoved=0;
    autoRegister=0;
    pickPts_id=0;
    matl = scinew Material(Color(.2,.2,.2), Color(0,0,.6), 
			   Color(.5,.5,.5), 20);
    navy = scinew Material(Color(.2,.2,.2), Color(0,0,.3),
			   Color(.5,.5,.5), 20);
    lavender = scinew Material(Color(.2,.2,.2), Color(.85,.4,.85),
			   Color(.5,.5,.5), 20);
    orange = scinew Material(Color(.2,.2,.2), Color(.8,.4,0),
			   Color(.5,.5,.5), 20);
    ptCloudFids_id = 0;
    ptCloudFidsFlag = 0;
    force_redraw = 0;
    fiducialFit = 0;
}

Coregister::~Coregister()
{
}

void buildRotateMatrix(double rm[][3], double angle, const Vector& axis) {
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

Vector rotateVector(const Vector& v_r, double rm[][3]) {
    return Vector(v_r.x()*rm[0][0]+v_r.y()*rm[0][1]+v_r.z()*rm[0][2],
		  v_r.x()*rm[1][0]+v_r.y()*rm[1][1]+v_r.z()*rm[1][2],
		  v_r.x()*rm[2][0]+v_r.y()*rm[2][1]+v_r.z()*rm[2][2]);
}

void Coregister::transform_pts(const Point& w_c, const Point& p,
			       const Vector& vr, const Vector& vd, 
			       const Vector& vi, int perc) {
    double a0, a1, a2;
    int incr=(int)(100./perc);
    double sc=scale.get();
    int useSc=useScale.get();
    if (!useSc) sc=1.0;

    int i;
    for (i=0; i<orig_pts.size(); i+=incr) {
	a0=(orig_pts[i].x()-w_c.x())*sc; 
	a1=(orig_pts[i].y()-w_c.y())*sc; 
	a2=(orig_pts[i].z()-w_c.z())*sc;
	trans_pts[i]=Point (p.x()+a0*vr.x()+a1*vd.x()+a2*vi.x(),
			    p.y()+a0*vr.y()+a1*vd.y()+a2*vi.y(),
			    p.z()+a0*vr.z()+a1*vd.z()+a2*vi.z());
    }	
    for (i=0; i<orig_vec.size(); i+=incr) {
	a0=orig_vec[i].x();
	a1=orig_vec[i].y();
	a2=orig_vec[i].z();
	trans_vec[i]=Vector(a0*vr.x()+a1*vd.x()+a2*vi.x(),
			    a0*vr.y()+a1*vd.y()+a2*vi.y(),
			    a0*vr.z()+a1*vd.z()+a2*vi.z());
    }
}

void Coregister::makeMGD(int full) {
    Array1<NodeHandle> nodes;

    old_scalp->get_surfnodes(nodes);

    BBox bb;
    Array1<Point> pts;
    for (int aa=0; aa<nodes.size(); aa++) {
	if (!nodes[aa].get_rep()) continue;
	bb.extend(nodes[aa]->p);
	pts.add(nodes[aa]->p);
    }
    Point min, max;
    min=bb.min()-(bb.diagonal()/4.);
    max=bb.max()+(bb.diagonal()/4.);
    long_edge = bb.longest_edge();
    mgd = new ManhattanDist(pts, 16, full, min.x(), min.y(), min.z(),
			    max.x(), max.y(), max.z());
}

void inverse3by3(double Pmat[3][3], double Pimat[3][3]) {
    double t1=1./(Pmat[0][0]*Pmat[1][1]*Pmat[2][2]-
		  Pmat[0][0]*Pmat[1][2]*Pmat[2][1]-
		  Pmat[1][0]*Pmat[0][1]*Pmat[2][2]+
		  Pmat[1][0]*Pmat[0][2]*Pmat[2][1]+
		  Pmat[2][0]*Pmat[0][1]*Pmat[1][2]-
		  Pmat[2][0]*Pmat[0][2]*Pmat[1][1]);

    Pimat[0][0]=(Pmat[1][1]*Pmat[2][2]-Pmat[1][2]*Pmat[2][1])*t1;
    Pimat[0][1]=-(Pmat[0][1]*Pmat[2][2]-Pmat[0][2]*Pmat[2][1])*t1;
    Pimat[0][2]=(Pmat[0][1]*Pmat[1][2]-Pmat[0][2]*Pmat[1][1])*t1;

    Pimat[1][0]=-(Pmat[1][0]*Pmat[2][2]-Pmat[1][2]*Pmat[2][0])*t1;
    Pimat[1][1]=(Pmat[0][0]*Pmat[2][2]-Pmat[0][2]*Pmat[2][0])*t1;
    Pimat[1][2]=-(Pmat[0][0]*Pmat[1][2]-Pmat[0][2]*Pmat[1][0])*t1;

    Pimat[2][0]=(Pmat[1][0]*Pmat[2][1]-Pmat[1][1]*Pmat[2][0])*t1;
    Pimat[2][1]=-(Pmat[0][0]*Pmat[2][1]-Pmat[0][1]*Pmat[2][0])*t1;
    Pimat[2][2]=(Pmat[0][0]*Pmat[1][1]-Pmat[0][1]*Pmat[1][0])*t1;
}

void transpose3by3(double A[3][3], double At[3][3]) {
    for (int i=0; i<3; i++) for (int j=0; j<3; j++) At[i][j]=A[j][i];
}

void mult3by3(double A[3][3], double B[3][3], double C[3][3]) {
    for (int i=0; i<3; i++) {
	for (int j=0; j<3; j++) {
	    C[i][j]=0.0;
	    for (int k=0; k<3; k++) {
		C[i][j]+=A[i][k]*B[k][j];
	    }
	}
    }
}

// transform a pt by pushing it though a rotation matrix (M) and adding a
// displacement Vector (v)
Point transformPt(double m[3][3], const Point& p, const Vector& v) {
    return Point(p.x()*m[0][0]+p.y()*m[0][1]+p.z()*m[0][2]+v.x(),
		 p.x()*m[1][0]+p.y()*m[1][1]+p.z()*m[1][2]+v.y(),
		 p.x()*m[2][0]+p.y()*m[2][1]+p.z()*m[2][2]+v.z());
}

// angular distance (radians) from U1->U2 around normal N
double angularDistance(const Vector &N, const Vector &U1, const Vector &U2) {
    Vector V(Cross(N, U2));
    double d=Sqrt(1-(Dot(U1, U2))*(Dot(U1,U2)));
    if (Dot(U1, V) > 0) d=-d;
    return Asin(d);
}

// want to minimize the distance between the a point and the p points
// they're already in the same plane and they have the same centroid
// want to find the rotation (theta) about the centroid that minimizes
// the least summed squared distance between the points
// and want to find the scale factor (sc) to get a->p
double leastSquares(const Point& a0new, const Point& a1new, const Point& a2new,
		  const Point& p0, const Point& p1, const Point& p2, 
		  const Point& pC, const Vector& N, double &sc) {
    Vector a0v(a0new-pC);
    Vector a1v(a1new-pC);
    Vector a2v(a2new-pC);
    Vector p0v(p0-pC);
    Vector p1v(p1-pC);
    Vector p2v(p2-pC);

    double ra0=a0v.length2();
    double ra1=a1v.length2();
    double ra2=a2v.length2();
    double rp0=p0v.length2();
    double rp1=p1v.length2();
    double rp2=p2v.length2();
    
    // these are the "perserved" vectors (in this case, cast to a point)
    // that we'll need for computing the scale factor...
    // as opposed to the originals which will be normalized in a second
    // for computing the rotation.
    Point a0vp=a0v.point();
    Point a1vp=a1v.point();
    Point a2vp=a2v.point();

    Vector p0vp(p0v);
    Vector p1vp(p1v);
    Vector p2vp(p2v);

    a0v.normalize();
    a1v.normalize();
    a2v.normalize();
    p0v.normalize();
    p1v.normalize();
    p2v.normalize();
    double th0=angularDistance(N, a0v, p0v);
    double th1=angularDistance(N, a1v, p1v);
    double th2=angularDistance(N, a2v, p2v);

    double theta=th0+atan((ra1*rp1*sin(th0-th1)+ra2*rp2*sin(th0-th2))/
			  (ra0*rp0+ra1*rp1*cos(th0-th1)+ra2*rp2*cos(th0-th2)));

    double M[3][3];
    buildRotateMatrix(M, theta, N);

    // now we'll rotate the vectors through this matrix
    Vector a0vnew=Point(transformPt(M, a0vp, Vector(0,0,0))).vector();
    Vector a1vnew=Point(transformPt(M, a1vp, Vector(0,0,0))).vector();
    Vector a2vnew=Point(transformPt(M, a2vp, Vector(0,0,0))).vector();

    sc = (Dot(a0vnew,p0vp)+Dot(a1vnew,p1vp)+Dot(a2vnew,p2vp))/
	(a0vnew.length2()+a1vnew.length2()+a2vnew.length2());

    cerr << "Scale = "<<sc<<"  Theta = "<<theta<<" radians.\n";

    return theta;
}

// Match up the planes that the triangles are in, the centroids of the 
// triangles, and the direction vector from the left to the
// right ear... or, do a least-squares-fit of the three points
double FiducialFit(double ax[], double ay[], double az[], double px[], 
		double py[], double pz[], double TT[4][4], int useLSF,
		   int useScale) {
    Point p0(px[0], py[0], pz[0]);
    Point p1(px[1], py[1], pz[1]);
    Point p2(px[2], py[2], pz[2]);
    Point a0(ax[0], ay[0], az[0]);
    Point a1(ax[1], ay[1], az[1]);
    Point a2(ax[2], ay[2], az[2]);

    Vector pv0(p0-p1);
    Vector pv1(p0-p2);
    Vector pN(Cross(pv0,pv1));
    pN.normalize();
    Vector pU(p2-p0);
    pU.normalize();
    Vector pV(Cross(pN, pU));
    Point pC(AffineCombination(p0, 1./3, p1, 1./3, p2, 1./3));

    Vector av0(a0-a1);
    Vector av1(a0-a2);
    Vector aN(Cross(av0,av1));
    aN.normalize();
    Vector aU(a2-a0);
    aU.normalize();
    Vector aV(Cross(aN, aU));
    Point aC(AffineCombination(a0, 1./3, a1, 1./3, a2, 1./3));


    // ok, now we have the coordinate frames and the centroids of both 
    // triangles.

    // need Amat (basis vectors of A, i.e. A -> R^3)
    // and Pmat (basis vector of P, i.e. P -> R^3)
    // and finally Pimat (R^3 -> P)
    // Pinv * A = TT (rotation)   [A -> R^3 -> P]

    double Amat[3][3];
    double Pmat[3][3];
    double Pimat[3][3];
    
    Amat[0][0]=aN.x(); Amat[1][0]=aU.x(); Amat[2][0]=aV.x();
    Amat[0][1]=aN.y(); Amat[1][1]=aU.y(); Amat[2][1]=aV.y();
    Amat[0][2]=aN.z(); Amat[1][2]=aU.z(); Amat[2][2]=aV.z();

    Pmat[0][0]=pN.x(); Pmat[1][0]=pU.x(); Pmat[2][0]=pV.x();
    Pmat[0][1]=pN.y(); Pmat[1][1]=pU.y(); Pmat[2][1]=pV.y();
    Pmat[0][2]=pN.z(); Pmat[1][2]=pU.z(); Pmat[2][2]=pV.z();

    // need the inverse, but since it's a 3x3 orthonormal matrix, the
    // inverse is just the transpose
    transpose3by3(Pmat, Pimat);

    double APi[3][3];
    mult3by3(Pimat, Amat, APi);

    // this is the point that aC would project to with just the rotation
    // we'll subtract it from the desired point (pC) to get the translation
    // vector for TT
    Point aCrot(transformPt(APi, aC, Vector(0,0,0)));
    Vector d(pC-aCrot);
    
    Point a0new(transformPt(APi, a0, d));
    Point a1new(transformPt(APi, a1, d));
    Point a2new(transformPt(APi, a2, d));
		
    double sc;
    double theta = leastSquares(a0new, a1new, a2new, p0, p1, p2, pC, pN, sc);

    if (!useScale) sc=1.0;
    if (!useLSF) theta=0;

    // now we have to build the rotation matrix that corresponds to the
    // theta rotation in the p-plane about the centroid (pC)
    // and multiply our current rotation, APi, by this rotation M, to
    // get our final rotation matrix (F = M * APi)
    double M[3][3], F[3][3];
    buildRotateMatrix(M, theta, pN);
    mult3by3(M, APi, F);

    int i,j;
    // work the scale into the rotation matrix!
    for (i=0; i<3; i++) for (j=0; j<3; j++) F[i][j]*=sc;

    // need to recompute the translation vector
    Point aCrot2(transformPt(F, aC, Vector(0,0,0)));
    Vector d2(pC-aCrot2);

    TT[0][0]=1.0; TT[0][1]=TT[0][2]=TT[0][3]=0.0;
    TT[1][0]=d2.x(); TT[2][0]=d2.y(); TT[3][0]=d2.z();
    for (i=0; i<3; i++)
	for (j=0; j<3; j++)
	    TT[i+1][j+1]=F[i][j];
    return sc;
}

void M4x4fromVecs(double M[4][4], const Vector &r, const Vector &d, 
		  const Vector &i) {
    M[0][0]=r.x();
    M[0][1]=d.x();
    M[0][2]=i.x();
    M[0][3]=0;
    M[1][0]=r.y();
    M[1][1]=d.y();
    M[1][2]=i.y();
    M[1][3]=0;
    M[2][0]=r.z();
    M[2][1]=d.z();
    M[2][2]=i.z();
    M[2][3]=0;
    M[3][0]=0;
    M[3][1]=0;
    M[3][2]=0;
    M[3][3]=1;
}

void VecsFromM4x4(double M[4][4], Vector &r, Vector &d, Vector &i) {
    r.x(M[0][0]);
    r.y(M[1][0]);
    r.z(M[2][0]);
    d.x(M[0][1]);
    d.y(M[1][1]);
    d.z(M[2][1]);
    i.x(M[0][2]);
    i.y(M[1][2]);
    i.z(M[2][2]);
}

void Coregister::execute()
{
    SurfaceHandle iScalpHdl;
    SurfaceHandle iPtsHdl;

// a silly test-case to see if this code works...
//    Point aa0(0,0,0);
//    Point aa1(0,4,0);
//    Point aa2(3,0,0);
//    Point ppC(1,4./3, 0);
//    Vector ppN(0,0,1);
//    Point pp0(-0.3110, 0.3043 ,0);
//    Point pp1(0.7243, 4.1680, 0);
//    Point pp2(2.5868, -0.4722, 0);
//    double theta = leastSquares(aa0, aa1, aa2, pp0, pp1, pp2, ppC, ppN);
//    cerr << "theta*12="<<theta*12<<"\n";

    if(!iport_scalp->get(iScalpHdl))
	return;
    if(!iport_pts->get(iPtsHdl))
	return;

    if (force_redraw) {
	owidget->flushViews();
	force_redraw=0;
    }
    if (ptCloudFidsFlag) {
	ptCloudFidsFlag=0;
	if (iPtsHdl.get_rep()) {
	    Array1<NodeHandle> nodes;
	    iPtsHdl->get_surfnodes(nodes);
	    if (nodes.size() < 3) return;
	    GeomGroup *gg = scinew GeomGroup;
	    GeomObj *go;
	    GeomMaterial *gm;

	    int offset=0;

	    offset=128;		// we put our fiducials after out electrodes!

	    nasion_dig=nodes[0+offset]->p;
	    go = scinew GeomSphere(nasion_dig, .03);
	    gm = scinew GeomMaterial(go, navy);
	    gg->add(gm);
	    left_dig=nodes[1+offset]->p;
	    go = scinew GeomSphere(left_dig, .03);
	    gm = scinew GeomMaterial(go, lavender);
	    gg->add(gm);
	    right_dig=nodes[2+offset]->p;
	    go = scinew GeomSphere(right_dig, .03);
	    gm = scinew GeomMaterial(go, orange);
	    gg->add(gm);
	    geomPts = scinew GeomPts(nodes.size());
	    ptCloudFids_id=owidget->addObj(gg, "Pt Cloud Fiducials");
	}
    }
    if (send_fid != "") {
	GeomMaterial *gm;
//	cerr << "left_id="<<left_id<<"  nasion_id="<<nasion_id<<"  right_id="<<right_id<<"\n";
	if (send_fid == "nasion") {
	    GeomObj *go = scinew GeomSphere(nasion, .05);
	    if (nasion_id) owidget->delObj(nasion_id);
	    gm = new GeomMaterial(go, navy);
	    nasion_id = owidget->addObj(gm, "Nasion");
	} else if (send_fid == "left") {
	    GeomObj *go = scinew GeomSphere(left, .05);
	    if (left_id) owidget->delObj(left_id);
	    gm = new GeomMaterial(go, lavender);
	    left_id = owidget->addObj(gm, "Left Max");
	} else if (send_fid == "right") {
	    GeomObj *go = scinew GeomSphere(right, .05);
	    if (right_id) owidget->delObj(right_id);
	    gm = new GeomMaterial(go, orange);
	    right_id = owidget->addObj(gm, "Right Max");
	} else {
	    cerr << "Unknown fiducial type: "<<send_fid<<"\n";
	    return;
	}
	cerr << "left_id="<<left_id<<"  nasion_id="<<nasion_id<<"  right_id="<<right_id<<"\n";
	send_fid="";
//	return;
    }


    if (iScalpHdl.get_rep() != old_scalp.get_rep()) {
	old_scalp = iScalpHdl;

	// need to delete old Pick points if they exist and send new ones!
	if (pickPts_id) owidget->delObj(pickPts_id);
	TriSurface *ts=old_scalp->getTriSurface();

	if (!ts) {
	    cerr << "Error -- coregistration need a trisurface!\n";
	    return;
	}
	if (ts->name != "scalp") {
	    cerr << "Warning -- surface isn't named scalp!!\n";
	}
	GeomPts* gpts=scinew GeomPts(ts->points.size());
	sclpPts.resize(0);
	for (int s=0; s<ts->points.size(); s++) {
	    sclpPts.add(ts->points[s]);
	    gpts->add(ts->points[s]);
	}
	gpk=scinew GeomPick(gpts, this);
	gpts->pickable=1;
	reset_vars();
	if (fiducial.get() == "none")
	    gpk->drawOnlyOnPick=1;
	else
	    gpk->drawOnlyOnPick=0;
	pickPts_id = owidget->addObj(gpk, "Pickable surface nodes");
	makeMGD(0);
    }
    
    if ((sclpPts.size()>2) && useFirstSurfPts.get()) {
	GeomObj *go;
	GeomMaterial *gm;
        useFirstSurfPts.set(0);
	nasion=sclpPts[0]; left=sclpPts[1]; right=sclpPts[2];
	go = scinew GeomSphere(nasion, .05);
	if (nasion_id) owidget->delObj(nasion_id);
	gm = new GeomMaterial(go, navy);
	nasion_id = owidget->addObj(gm, "Nasion");
	go = scinew GeomSphere(left, .05);
	if (left_id) owidget->delObj(left_id);
	gm = new GeomMaterial(go, lavender);
	left_id = owidget->addObj(gm, "Left Max");
	go = scinew GeomSphere(right, .05);
	if (right_id) owidget->delObj(right_id);
	gm = new GeomMaterial(go, orange);
	right_id = owidget->addObj(gm, "Right Max");
    }

    if (iPtsHdl.get_rep()==old_pts.get_rep() && !autoRegister && 
	!widgetMoved && !fiducialFit){
	oport_pts->send(osh);
	return;
    }

    widgetMoved=0;

    if (!initialized) {	     // first time through - have to connect widget
 	initialized=1;
	GeomObj *w=widget->GetWidget();
	owidget->addObj(w, clString("Coreg Frame"), &widget_lock);
	widget->Connect(owidget);
        widget->SetRatioR(0.2);
        widget->SetRatioD(0.2);
        widget->SetRatioI(0.2);
    }
    
    if (autoRegister) {	    // they want us to register this??  well, ok...
	autoRegister=0;
        auto_register();
//        return;
    }

    if (fiducialFit) {
	fiducialFit=0;
	Vector v_r(widget->GetRightAxis());
	Vector v_d(widget->GetDownAxis());
	Vector v_i(widget->GetInAxis());
	Point p(widget->ReferencePoint());
	double ax[3], ay[3], az[3], px[3], py[3], pz[3], TT[4][4];

	px[0]=left.x(); px[1]=nasion.x(); px[2]=right.x();
	py[0]=left.y(); py[1]=nasion.y(); py[2]=right.y();
	pz[0]=left.z(); pz[1]=nasion.z(); pz[2]=right.z();

	double a0, a1, a2;

	// ignore scale here -- we'll recompute it if wanted...
	a0=left_dig.x()-w_c.x();
	a1=left_dig.y()-w_c.y();
	a2=left_dig.z()-w_c.z();
	ax[0]=p.x()+a0*v_r.x()+a1*v_d.x()+a2*v_i.x();
	ay[0]=p.y()+a0*v_r.y()+a1*v_d.y()+a2*v_i.y();
	az[0]=p.z()+a0*v_r.z()+a1*v_d.z()+a2*v_i.z();

	a0=nasion_dig.x()-w_c.x();
	a1=nasion_dig.y()-w_c.y();
	a2=nasion_dig.z()-w_c.z();
	ax[1]=p.x()+a0*v_r.x()+a1*v_d.x()+a2*v_i.x();
	ay[1]=p.y()+a0*v_r.y()+a1*v_d.y()+a2*v_i.y();
	az[1]=p.z()+a0*v_r.z()+a1*v_d.z()+a2*v_i.z();

	a0=right_dig.x()-w_c.x();
	a1=right_dig.y()-w_c.y();
	a2=right_dig.z()-w_c.z();
	ax[2]=p.x()+a0*v_r.x()+a1*v_d.x()+a2*v_i.x();
	ay[2]=p.y()+a0*v_r.y()+a1*v_d.y()+a2*v_i.y();
	az[2]=p.z()+a0*v_r.z()+a1*v_d.z()+a2*v_i.z();

	int useLSF(fiducialMethod.get() == "LSF");
	double sc=FiducialFit(ax, ay, az, px, py, pz, TT, useLSF, useScale.get());
	scale.set(sc);

	// now we have TT -- need to put the old rotation vector through it
	// to get the new ones, and the old centerpoint through it to get
	// the new one (scale is already build into TT here)

	Vector v_r_new, v_d_new, v_i_new;
	Point pn;

	v_r_new.x(v_r.x()*TT[1][1] + v_r.y()*TT[1][2] + v_r.z()*TT[1][3]);
	v_r_new.y(v_r.x()*TT[2][1] + v_r.y()*TT[2][2] + v_r.z()*TT[2][3]);
	v_r_new.z(v_r.x()*TT[3][1] + v_r.y()*TT[3][2] + v_r.z()*TT[3][3]);

	v_d_new.x(v_d.x()*TT[1][1] + v_d.y()*TT[1][2] + v_d.z()*TT[1][3]);
	v_d_new.y(v_d.x()*TT[2][1] + v_d.y()*TT[2][2] + v_d.z()*TT[2][3]);
	v_d_new.z(v_d.x()*TT[3][1] + v_d.y()*TT[3][2] + v_d.z()*TT[3][3]);

	v_i_new.x(v_i.x()*TT[1][1] + v_i.y()*TT[1][2] + v_i.z()*TT[1][3]);
	v_i_new.y(v_i.x()*TT[2][1] + v_i.y()*TT[2][2] + v_i.z()*TT[2][3]);
	v_i_new.z(v_i.x()*TT[3][1] + v_i.y()*TT[3][2] + v_i.z()*TT[3][3]);

	pn.x(TT[1][0] + p.x()*TT[1][1] + p.y()*TT[1][2] + p.z()*TT[1][3]);
	pn.y(TT[2][0] + p.x()*TT[2][1] + p.y()*TT[2][2] + p.z()*TT[2][3]);
	pn.z(TT[3][0] + p.x()*TT[3][1] + p.y()*TT[3][2] + p.z()*TT[3][3]);


// 	this next part is just to check -- p{x,y,z}[0-2] are the new
// 	positions of the a's

//	a0=left_dig.x()-w_c.x();
//	a1=left_dig.y()-w_c.y();
//	a2=left_dig.z()-w_c.z();
//	px[0]=pn.x()+a0*v_r_new.x()+a1*v_d_new.x()+a2*v_i_new.x();
//	py[0]=pn.y()+a0*v_r_new.y()+a1*v_d_new.y()+a2*v_i_new.y();
//	pz[0]=pn.z()+a0*v_r_new.z()+a1*v_d_new.z()+a2*v_i_new.z();

//	a0=nasion_dig.x()-w_c.x();
//	a1=nasion_dig.y()-w_c.y();
//	a2=nasion_dig.z()-w_c.z();
//	px[1]=pn.x()+a0*v_r_new.x()+a1*v_d_new.x()+a2*v_i_new.x();
//	py[1]=pn.y()+a0*v_r_new.y()+a1*v_d_new.y()+a2*v_i_new.y();
//	pz[1]=pn.z()+a0*v_r_new.z()+a1*v_d_new.z()+a2*v_i_new.z();

//	a0=right_dig.x()-w_c.x();
//	a1=right_dig.y()-w_c.y();
//	a2=right_dig.z()-w_c.z();
//	px[2]=pn.x()+a0*v_r_new.x()+a1*v_d_new.x()+a2*v_i_new.x();
//	py[2]=pn.y()+a0*v_r_new.y()+a1*v_d_new.y()+a2*v_i_new.y();
//	pz[2]=pn.z()+a0*v_r_new.z()+a1*v_d_new.z()+a2*v_i_new.z();

//	cerr << "new pt 0: ("<<px[0]<<", "<<py[0]<<", "<<pz[0]<<")\n";
//	cerr << "new pt 1: ("<<px[1]<<", "<<py[1]<<", "<<pz[1]<<")\n";
//	cerr << "new pt 2: ("<<px[2]<<", "<<py[2]<<", "<<pz[2]<<")\n";

// divide out the scale so the rotation vectors are normalized

#if 0
	Array1<Quat> Q(9);
	Array1<double> scs(9);
	Array1<Point> ps(9);
	for (int ii=0; ii<9; ii++) {
	    scs[ii]=(ii*(sc-1.))/8.+1;
	    ps[ii]=AffineCombination(p, ii/8., pn, 1-(ii/8.));
	}
	Array1<double[4][4]> M(9);
	M4x4fromVecs(M[0], v_r, v_d, v_i);
	M4x4fromVecs(M[8], v_r_new, v_d_new, v_i_new);
	Q[0].FromMatrix(M[0]);
	Q[8].FromMatrix(M[8]);
	Q[4] = V3_Bisect(Q[0], Q[8]);
	Q[2] = V3_Bisect(Q[0], Q[4]);
	Q[1] = V3_Bisect(Q[0], Q[2]);
	Q[6] = V3_Bisect(Q[4], Q[8]);
	Q[3] = V3_Bisect(Q[2], Q[4]);
	Q[5] = V3_Bisect(Q[4], Q[6]);
	Q[7] = V3_Bisect(Q[6], Q[8]);
	for (ii=1; ii<8; ii++) {
	    Q[ii].ToMatrix(M[ii]);
	}	    

	Vector rt, dt, it;
	for (ii=0; ii<9; ii++) {
	    VecsFromM4x4(M[ii], rt, dt, it);
	    transform_pts(w_c, ps[ii], rt/scs[ii], dt/scs[ii], it/scs[ii]);
	    setTransSurfPts();
	    oport_pts->send(osh);
	}
	transform_pts(w_c, pn, v_r_new/sc, v_d_new/sc, v_i_new/sc);
	oport_pts->send(osh);
#endif

	widget->SetPosition(pn,pn+(v_r_new/sc),pn+(v_d_new/sc),pn+(v_i_new/sc));
    }


    // if these points are new, we have to build a new GeomPts object
    if (iPtsHdl.get_rep() != old_pts.get_rep()) {
	makeBaseTransSurf(iPtsHdl);
	Array1<NodeHandle> nodes;
	iPtsHdl->get_surfnodes(nodes);
	geomPts = scinew GeomPts(nodes.size());
	orig_pts.resize(nodes.size());
	trans_pts.resize(nodes.size());
	TriSurface *ts=iPtsHdl->getTriSurface();
	if (ts && ts->normals.size()) {
	    orig_vec=ts->normals;
	    trans_vec=ts->normals;
	} else {
	    orig_vec.resize(0);
	    trans_vec.resize(0);
	}
	for (int i=0; i<nodes.size(); i++) {
	    orig_pts[i]=nodes[i]->p;
	    geomPts->add(nodes[i]->p);
	}
    } else {   // have to copy them b/c we're gonna erase them below...
	geomPts = scinew GeomPts(*geomPts);
    }

    // our defaults values will just be the old values
    widget_lock.readLock();
    Point p(widget->ReferencePoint());	
    Vector v_r(widget->GetRightAxis());
    Vector v_d(widget->GetDownAxis());
    Vector v_i(widget->GetInAxis());
    widget_lock.readUnlock();

    // if the pts are new, reset widget
    if (iPtsHdl.get_rep() != old_pts.get_rep() || recenter) {
	if (recenter) {
	    BBox b;
	    geomPts->get_bounds(b);
	    w_v = b.diagonal();
	    w_c = p = b.center();
	    Vector v_c = Vector(p.x(), p.y(), p.z()); 
	    v_r = Vector(1,0,0);
	    v_d = Vector(0,1,0);
	    v_i = Vector(0,0,1);
	    widget->SetPosition(w_c, w_c+v_r, w_c+v_d, w_c+v_i);
	    widget->SetScale(w_v.length()/20.*scale.get());
//	    setTransSurfPts();
//	    oport_pts->send(osh);
	    recenter=0;
	}
	old_pts = iPtsHdl;
    }

    rot_r_x.set(v_r.x()); rot_r_y.set(v_r.y()); rot_r_z.set(v_r.z());
    rot_d_x.set(v_d.x()); rot_d_y.set(v_d.y()); rot_d_z.set(v_d.z());
    rot_i_x.set(v_i.x()); rot_i_y.set(v_i.y()); rot_i_z.set(v_i.z());
    trans_x.set(p.x()-w_c.x()); trans_y.set(p.y()-w_c.y()); 
    trans_z.set(p.z()-w_c.z()); 

    transform_pts(w_c, p, v_r, v_d, v_i);

    setTransSurfPts();
    oport_pts->send(osh);

    if (ptCloudFids_id) {
	owidget->delObj(ptCloudFids_id);
	ptCloudFids_id=0;
	if (trans_pts.size()>3) {
//	    cerr << "p0: "<<trans_pts[0]<<"\n";
//	    cerr << "p1: "<<trans_pts[1]<<"\n";
//	    cerr << "p2: "<<trans_pts[2]<<"\n";
	    GeomGroup *gg = scinew GeomGroup;
	    GeomObj *go;
	    GeomMaterial *gm;
//	    nasion_dig=trans_pts[0];
	    go = scinew GeomSphere(trans_pts[0], .03);
	    gm = scinew GeomMaterial(go, navy);
	    gg->add(gm);
//	    left_dig=trans_pts[1];
	    go = scinew GeomSphere(trans_pts[1], .03);
	    gm = scinew GeomMaterial(go, lavender);
	    gg->add(gm);
//	    right_dig=trans_pts[2];
	    go = scinew GeomSphere(trans_pts[2], .03);
	    gm = scinew GeomMaterial(go, orange);
	    gg->add(gm);
	    ptCloudFids_id=owidget->addObj(gg, "Pt Cloud Fiducials");
	}
    }
    reg_error.set(computeError());
    geomPts->pts.resize(0);
    for (int pp=0; pp<trans_pts.size(); pp++) {
	geomPts->add(trans_pts[pp]);
    }
}	

void Coregister::makeBaseTransSurf(SurfaceHandle s) {
    PointsSurface *ps=dynamic_cast<PointsSurface* >(s.get_rep());
    TriSurface *ts=dynamic_cast<TriSurface* >(s.get_rep());
    SurfTree *st=dynamic_cast<SurfTree* >(s.get_rep());
    ScalarTriSurface *ss=dynamic_cast<ScalarTriSurface* >(s.get_rep());
    if (!ps && !ts && !st && !ss) {
	cerr << "Coregister: unknown surface type.\n";
	return;
    }
    if (ps) {
	PointsSurface* nps=new PointsSurface(*ps);
	osh=nps;
    } else if (ts) {
	TriSurface* nts=new TriSurface(*ts);
	osh=nts;
    } else if (st) {
	SurfTree* nst=new SurfTree(*st);
	osh=nst;
    } else if (ss) {
	ScalarTriSurface* nss=new ScalarTriSurface(*ss);
	osh=nss;
    }
}

void Coregister::setTransSurfPts(void) {
    PointsSurface *ps=dynamic_cast<PointsSurface* >(osh.get_rep());
    TriSurface *ts=dynamic_cast<TriSurface* >(osh.get_rep());
    SurfTree *st=dynamic_cast<SurfTree* >(osh.get_rep());
    ScalarTriSurface *ss=dynamic_cast<ScalarTriSurface* >(osh.get_rep());
    if (!ps && !ts && !st && !ss) {
	cerr << "Coregister: unknown surface type.\n";
	return;
    }
    osh->monitor.writeLock();
    if (ps) {
	ps->pos=trans_pts;
    } else if (ts) {
	ts->points=trans_pts;
	ts->normals=trans_vec;
    } else if (st) {
	st->nodes=trans_pts;
    } else if (ss) {
	ss->points=trans_pts;
    }
    osh->monitor.writeUnlock();
}

double Coregister::computeError(int perc) {
    if (!mgd) {
	cerr << "Can't compute the Error without a ManhattanDist!\n";
	return -1;
    }
    if (!trans_pts.size()) return 0;
    double err=0;
    int incr=(int)(100./perc);
    int count=0;
    for (int i=0; i<trans_pts.size(); i+=incr, count++) {
	err+=mgd->dist2(trans_pts[i]);
    }
    err=Sqrt(err/count);
    return err;
}

double Coregister::findMin(double a, double b, double c, double *s) {
    if (a<b || c<b) {
       if (a<c) {
          *s=a-b;
          return b/(a-b);
       } else {
          *s=b-c;
          return b/(b-c);
       }
    }
    *s=0;
    return 10000;
}

void Coregister::auto_register() {
    int num_iters;
    int perc;
    double error_tol;

    reset_vars();
    clString ni(iters.get());
    clString pe(percent.get());
    clString trans(transform.get());

    cerr << "Using transform: "<<trans<<"\n";
    if (!ni.get_int(num_iters)) {
        cerr << "Invalid int for number of iterations: "<<ni<<"\n";
        iters.set("100");
        num_iters=100;
    }
    error_tol=0;
    if (!pe.get_int(perc)) {
        cerr << "Invalid percentage: "<<perc<<"\n";
        percent.set("10");
        perc=10;
    }

    int done=0;
    int currIter=1;
    curr_iter.set("1");
    widget_lock.readLock();
    Point p(widget->ReferencePoint());	
    Vector v_r(widget->GetRightAxis());
//    cerr << "Right axis="<<v_r<<"\n";
    Vector v_d(widget->GetDownAxis());
//    cerr << "Down axis="<<v_d<<"\n";
    Vector v_i(widget->GetInAxis());
//    cerr << "In axis="<<v_i<<"\n";
    if (v_r.length2()>2) {
	v_r=v_r/1;
	v_d=v_d/1;
	v_i=v_i/1;
    }
    widget_lock.readUnlock();

    transform_pts(w_c, p, v_r, v_d, v_i, perc);
    double currError=computeError(perc);
    
    Array1<Vector> a_r(13), a_d(13), a_i(13);
    Array1<Point> a_c(13);
    double t_scale=long_edge/20;
    double r_scale=.03*2*PI;
    double rm[3][3];
    int redraw=0;
    int in_a_row=0;

    // as long as we haven't exceded "iters" of iterations, and our
    // error isn't below "tolerance", and we're not "done" for some
    // other reason, we continue to iterate...

    while(currIter<=num_iters && currError>error_tol && !done) {
        // try some new transformations, and pick the one that's best.

        int idx;
        for (idx=0; idx<7; idx++) {
            a_r[idx]=v_r; a_d[idx]=v_d; a_i[idx]=v_i;
        }
        a_c[0]=p;
        for (idx=7; idx<13; idx++)
            a_c[idx]=p;
        a_c[1]=p+Vector(t_scale, 0, 0);
        a_c[2]=p+Vector(-t_scale, 0, 0);
        a_c[3]=p+Vector(0, t_scale, 0);
        a_c[4]=p+Vector(0, -t_scale, 0);
        a_c[5]=p+Vector(0, 0, t_scale);
        a_c[6]=p+Vector(0, 0, -t_scale);
        Array1<Vector> axis(6);
        axis[0]=Vector(1,0,0);
        axis[1]=Vector(-1,0,0);
        axis[2]=Vector(0,1,0);
        axis[3]=Vector(0,-1,0);
        axis[4]=Vector(0,0,1);
        axis[5]=Vector(0,0,-1);
        for (idx=0; idx<6; idx++) {
            buildRotateMatrix(rm, r_scale, axis[idx]);
            a_r[idx+7]=rotateVector(v_r, rm);
            a_d[idx+7]=rotateVector(v_d, rm);
            a_i[idx+7]=rotateVector(v_i, rm);
        }
        int bestIdx=0;
        double allErrors[13];
        allErrors[0]=currError;
        for (idx=1; idx<13; idx++) {
	    if (trans == "all" ||
		(idx<7 && trans=="translate") ||
		(idx>=7 && trans=="rotate")) {
		transform_pts(w_c, a_c[idx], a_r[idx], a_d[idx], a_i[idx], perc);
		allErrors[idx]=computeError(perc);
		if (allErrors[idx]<allErrors[bestIdx]) {
		    bestIdx=idx;
		}
            }
        }
        currError=allErrors[bestIdx];
	if (0) {
            Point new_c(p);
            Vector new_r(v_r), new_d(v_d), new_i(v_i);
            if (bestIdx<7) {
                double xScale, yScale, zScale;
                double aa,bb,cc;
                xScale = findMin(allErrors[2], allErrors[0], allErrors[1],&aa);
                yScale = findMin(allErrors[4], allErrors[0], allErrors[3],&bb);
                zScale = findMin(allErrors[6], allErrors[0], allErrors[5],&cc);
                Vector v(aa,bb,cc);
                double sc=Min(xScale,yScale,zScale);
                if (xScale==10000) xScale=0;
                if (yScale==10000) yScale=0;
                if (zScale==10000) zScale=0;
                if (sc!=10000) {
                    v.normalize();
                    v*=sc*t_scale;
                }
                new_c+=v;
            } else {
                int bestRotIdx = 7;
                //double bestRorErr = allErrors[7];
                for (idx=8; idx<13; idx++) {
                    if (allErrors[idx]<allErrors[bestRotIdx]);
                        bestRotIdx = idx;
                }
                bestRotIdx -= (bestRotIdx+1)%2;
                double aa;
                double rScale = findMin(allErrors[bestRotIdx+1], allErrors[0], 
                                        allErrors[bestRotIdx], &aa);
                buildRotateMatrix(rm, r_scale*rScale, axis[bestRotIdx-7]);
                new_r = rotateVector(v_r, rm);
                new_d = rotateVector(v_d, rm);
                new_i = rotateVector(v_i, rm);
            }
            transform_pts(w_c, new_c, new_r, new_d, new_i, perc);
            double ee=computeError(perc);
            if (ee<currError) {
                currError=ee;
                p=new_c;
                v_r=new_r; v_d=new_d; v_i=new_i;
            } else {
                p=a_c[bestIdx]; 
                v_r=a_r[bestIdx]; v_d=a_d[bestIdx]; v_i=a_i[bestIdx];
            }
        } else {
            p=a_c[bestIdx]; 
            v_r=a_r[bestIdx]; v_d=a_d[bestIdx]; v_i=a_i[bestIdx];
        }
        if (bestIdx==0) {
            r_scale*=.8;
            t_scale*=.8;
            in_a_row++;
        } else {
            in_a_row=0;
            redraw=1;
        }
        if (in_a_row>4) {
            r_scale*=.8;
            t_scale*=.8;
        }
        if (((currIter % 2) == 0) || (currIter==num_iters-1) || 
             (currError<=error_tol)) { // every 2 iters, check for abort
                                       // and output the current geometry
            reset_vars();
            if (abortButton.get()) {
                done=1;
                abortButton.set(0);
            }
            if (redraw) {
                redraw=0;
                transform_pts(w_c, p, v_r, v_d, v_i);
                rot_r_x.set(v_r.x()); rot_r_y.set(v_r.y()); rot_r_z.set(v_r.z());
                rot_d_x.set(v_d.x()); rot_d_y.set(v_d.y()); rot_d_z.set(v_d.z());
                rot_i_x.set(v_i.x()); rot_i_y.set(v_i.y()); rot_i_z.set(v_i.z());
                trans_x.set(p.x()-w_c.x()); trans_y.set(p.y()-w_c.y()); 
		trans_z.set(p.z()-w_c.z()); 
                widget->SetPosition(p, p+(v_r*1), p+(v_d*1), p+(v_i*1));

		setTransSurfPts();
		oport_pts->send_intermediate(osh);

		if (ptCloudFids_id) {
		    owidget->delObj(ptCloudFids_id);
		    ptCloudFids_id=0;
		    if (trans_pts.size()>3) {
			GeomGroup *gg = scinew GeomGroup;
			GeomObj *go;
			GeomMaterial *gm;
//			nasion_dig=trans_pts[0];
			go = scinew GeomSphere(trans_pts[0], .03);
			gm = scinew GeomMaterial(go, navy);
			gg->add(gm);
//			left_dig=trans_pts[1];
			go = scinew GeomSphere(trans_pts[1], .03);
			gm = scinew GeomMaterial(go, lavender);
			gg->add(gm);
//			right_dig=trans_pts[2];
			go = scinew GeomSphere(trans_pts[2], .03);
			gm = scinew GeomMaterial(go, orange);
			gg->add(gm);
			ptCloudFids_id=owidget->addObj(gg, "Pt Cloud Fiducials");
		    }
		}
   	        geomPts = scinew GeomPts(*geomPts);
 		geomPts->pts.resize(0);
                for (int pp=0; pp<trans_pts.size(); pp++) {
	            geomPts->add(trans_pts[pp]);
                }
            }
        }
        currIter++;
        reset_vars();
        if (currIter<=num_iters)
            curr_iter.set(to_string(currIter));
        reg_error.set(currError);
        reset_vars();
    }
}

void Coregister::geom_pick(GeomPick* pick, void* cb)
{
    int *val((int *)cb);
    if (cb && (*val != -1234)) {	
	int picknode = *val;
	Point p(sclpPts[picknode]);
	reset_vars();
	cerr <<"NODE "<<picknode<<"was picked for the "<<fiducial.get()<<".\n";
	if (fiducial.get() == "none") return;
	if (fiducial.get() == "nasion") {
	    nasion=p;
	    send_fid="nasion";
	} else if (fiducial.get() == "left") {
	    left=p;
	    send_fid="left";
	} else if (fiducial.get() == "right") {
	    right=p;
	    send_fid="right";
	}
	want_to_execute();
    }
}


void Coregister::widget_moved(int)
{
    if(!abort_flag)
        {
            abort_flag=1;
	    widgetMoved=1;
            want_to_execute();
        }
}


void Coregister::tcl_command(TCLArgs& args, void* userdata)
{
    if(args.count() < 2)
        {
            args.error("Coregister needs a minor command");
            return;
        }
    if(args[1] == "auto")
        {
	    autoRegister = 1;
            want_to_execute();
        }
    else if(args[1] == "go")
        {
	    widgetMoved = 1;
            want_to_execute();
        }
    else if(args[1] == "fiducialFit")
        {
	    if (ptCloudFids_id && left_id && right_id && nasion_id) {
		fiducialFit = 1;	
		want_to_execute();
	    } else {
		cerr << "Can't run fiducialFit -- need all the fiducials!\n";
	    }
        }
    else if(args[1] == "scale")
        {
	    widgetMoved = 1;
            want_to_execute();
        }
    else if(args[1] == "recenter")
        {
	    widgetMoved = 1;
	    recenter = 1;
            want_to_execute();
        }
    else if (args[1] == "none")
	{
	    if (gpk) {
		if (gpk->drawOnlyOnPick != 1) {
		    gpk->drawOnlyOnPick=1;
		    force_redraw=1;
		    want_to_execute();
		}
	    }
	}
    else if (args[1] == "left" || args[1] == "nasion" || args[1] == "right" ||
	     args[1] == "firstthree")
	{
	    int exec=0;
	    if (gpk) {
		if (gpk->drawOnlyOnPick != 0) {
		    gpk->drawOnlyOnPick=0;
		    force_redraw=1;
		    exec=1;
		}
	    }
	    if (!ptCloudFids_id) {
		ptCloudFidsFlag=1;
		exec=1;
	    }
	    if (exec) want_to_execute();
	}
    else if (args[1] == "print")
	{
	    cerr << "From surface:\n";
	    if (left_id) {
		cerr << "Left pt = "<<left<<"\n";
	    }
	    if (right_id) {
		cerr << "Right pt = "<<right<<"\n";
	    }
	    if (nasion_id) {
		cerr << "Nasion pt = "<<nasion<<"\n";
	    }
	    cerr << "From cloud:\n";
	    if (ptCloudFids_id) {
		cerr << "Left pt = "<<left_dig<<"\n";
		cerr << "Nasion pt = "<<nasion_dig<<"\n";
		cerr << "Right pt = "<<right_dig<<"\n";
	    }
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
// Revision 1.3  1999/08/29 00:46:36  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.2  1999/08/25 03:47:38  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.1  1999/08/24 06:23:01  dmw
// Added in everything for the DaveW branch
//
// Revision 1.2  1999/05/03 04:52:12  dmw
// Added and updated DaveW Datatypes/Modules
//
//
