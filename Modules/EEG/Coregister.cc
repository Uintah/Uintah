
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

#include <config.h>
#undef SCI_ASSERTION_LEVEL_3
#define SCI_ASSERTION_LEVEL_2
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/BasicSurfaces.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ManhattanDist.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/TriSurface.h>
#include <Geom/Pt.h>
#include <Geom/Material.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Math/Expon.h>
#include <Math/MinMax.h>
#include <Math/Trig.h>
#include <TCL/TCLvar.h>
#include <Widgets/ScaledBoxWidget.h>
#include <iostream.h>
#include <Malloc/Allocator.h>

class Coregister : public Module {
    SurfaceIPort* iport_scalp;
    SurfaceIPort* iport_pts;
    SurfaceOPort* oport_pts;
    GeomPts* geomPts;
    GeometryOPort* ogeom;
    GeometryOPort* owidget;
    CrowdMonitor widget_lock;
    ScaledBoxWidget *widget;
    int build_full_mgd;
    int print_mgd;
    int send_pts;
    int geom_id;
    int initialized;
    int widgetMoved;
    int autoRegister;
    Vector w_v;
    Point w_c;
    double long_edge;
    SurfaceHandle old_pts;
    Array1<Point> orig_pts;
    Array1<Point> trans_pts;
    SurfaceHandle old_scalp;
    ManhattanDist *mgd;
    TCLdouble rot_r_x, rot_r_y, rot_r_z;
    TCLdouble rot_d_x, rot_d_y, rot_d_z;
    TCLdouble rot_i_x, rot_i_y, rot_i_z;
    TCLdouble trans_x, trans_y, trans_z;
    TCLdouble reg_error;
    TCLdouble scale;
    TCLint abortButton;
    TCLint newtonB;
    TCLstring curr_iter;
    TCLstring error_metric;
    TCLstring iters;
    TCLstring tolerance;
    TCLstring percent;
    MaterialHandle matl;
    int recenter;
public:
    Coregister(const clString& id);
    Coregister(const Coregister&, int deep);
    virtual ~Coregister();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void widget_moved(int last);    
    virtual void tcl_command(TCLArgs&, void*);
    double computeError(int perc=100);
    void auto_register();
    void transform_pts(const Point& w_c, const Point& p,
		       const Vector& vr, const Vector& vd, 
		       const Vector& vi, int perc=100);
    void makeMGD(int full);
    void printMGD();
    void sendPts();
    double findMin(double, double, double, double*);
};

extern "C" {
Module* make_Coregister(const clString& id)
{
    return new Coregister(id);
}
}

static clString module_name("Coregister");

Coregister::Coregister(const clString& id)
: Module("Coregister", id, Filter), reg_error("reg_error", id, this),
  rot_r_x("rot_r_x",id,this),rot_r_y("rot_r_y",id,this),rot_r_z("rot_r_z",id,this),
  rot_d_x("rot_d_x",id,this),rot_d_y("rot_d_y",id,this),rot_d_z("rot_d_z",id,this),
  rot_i_x("rot_i_x",id,this),rot_i_y("rot_i_y",id,this),rot_i_z("rot_i_z",id,this),
  trans_x("trans_x",id,this),trans_y("trans_y",id,this),trans_z("trans_z",id,this),
  scale("scale", id, this),
  error_metric("error_metric",id,this), iters("iters",id,this),
  tolerance("tolerance",id,this), abortButton("abortButton",id,this),
  curr_iter("curr_iter",id,this), percent("percent",id,this),
  newtonB("newtonB",id,this), build_full_mgd(0), print_mgd(0), recenter(1),
  send_pts(0)
{
    // Create the input ports
    iport_scalp=new SurfaceIPort(this, "Scalp", SurfaceIPort::Atomic);
    add_iport(iport_scalp);
    iport_pts=new SurfaceIPort(this, "Points", SurfaceIPort::Atomic);
    add_iport(iport_pts);
    // Create the output port
    owidget=new GeometryOPort(this, "Widget", GeometryIPort::Atomic);
    add_oport(owidget);
    ogeom=new GeometryOPort(this, "GeomOut", GeometryIPort::Atomic);
    add_oport(ogeom);
    oport_pts=new SurfaceOPort(this, "OPoints", SurfaceIPort::Atomic);
    add_oport(oport_pts);
    widget=scinew ScaledBoxWidget(this, &widget_lock, 0.2);
    widget->SetCurrentMode(2);
    geom_id=0;
    geomPts=0;
    mgd=0;
    initialized=0;
    widgetMoved=0;
    autoRegister=0;
    matl = scinew Material(Color(.2,.2,.2), Color(0,0,.6), Color(0,0,.5), 20);
}

Coregister::Coregister(const Coregister& copy, int deep)
: Module(copy, deep), reg_error("reg_error", id, this),
  rot_r_x("rot_r_x",id,this),rot_r_y("rot_r_y",id,this),rot_r_z("rot_r_z",id,this),
  rot_d_x("rot_d_x",id,this),rot_d_y("rot_d_y",id,this),rot_d_z("rot_d_z",id,this),
  rot_i_x("rot_i_x",id,this),rot_i_y("rot_i_y",id,this),rot_i_z("rot_i_z",id,this),
  trans_x("trans_x",id,this),trans_y("trans_y",id,this),trans_z("trans_z",id,this),
  scale("scale", id, this),
  error_metric("error_metric",id,this), iters("iters",id,this),
  tolerance("tolerance",id,this), abortButton("abortButton",id,this),
  curr_iter("curr_iter",id,this), percent("percent",id,this),
  newtonB("newtonB",id,this), build_full_mgd(0), print_mgd(0), recenter(1),
  send_pts(0)
{
    NOT_FINISHED("Coregister::Coregister");
}

Coregister::~Coregister()
{
}

Module* Coregister::clone(int deep)
{
    return new Coregister(*this, deep);
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
    for (int i=0; i<orig_pts.size(); i+=incr) {
	a0=(orig_pts[i].x()-w_c.x())*sc; 
	a1=(orig_pts[i].y()-w_c.y())*sc; 
	a2=(orig_pts[i].z()-w_c.z())*sc;
	trans_pts[i]=Point (p.x()+a0*vr.x()+a1*vd.x()+a2*vi.x(),
			    p.y()+a0*vr.y()+a1*vd.y()+a2*vi.y(),
			    p.z()+a0*vr.z()+a1*vd.z()+a2*vi.z());
    }	
}

void Coregister::makeMGD(int full) {
//    if (mgd) delete [] mgd;
    Array1<NodeHandle> nodes;
    old_scalp->get_surfnodes(nodes);
    BBox bb;
    Array1<Point> pts;
    for (int aa=0; aa<nodes.size(); aa++) {
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

void Coregister::printMGD() {
	for (int a=0; a<mgd->grid.dim1(); a++) {
	    cerr << "MGD Slice: "<<a<<"\n";
	    for (int b=0; b<mgd->grid.dim2(); b++) {
		for (int c=0; c<mgd->grid.dim3(); c++) {
		    cerr << mgd->grid(a,b,c) << " ";
		}
		cerr << "\n";
	    }
	    cerr << "\n\n";
	}
}
    
void Coregister::execute()
{
    SurfaceHandle iScalpHdl;
    SurfaceHandle iPtsHdl;
    if(!iport_scalp->get(iScalpHdl))
	return;
    if(!iport_pts->get(iPtsHdl))
	return;
    if (iScalpHdl.get_rep()!=old_scalp.get_rep()) {
	old_scalp = iScalpHdl;
	makeMGD(0);
//        reg_error.set(computeError());
    }	

    if (build_full_mgd) {
	build_full_mgd=0;
	makeMGD(1);
	return;
    }

    if (print_mgd) {
	print_mgd=0;
	printMGD();
	return;
    }

    if (send_pts) {
	send_pts=0;
	sendPts();
	return;
    }

    if (!autoRegister && !widgetMoved && iPtsHdl.get_rep()==old_pts.get_rep())
	return;

    widgetMoved=0;

    if (!initialized) {	     // first time through - have to connect widget
 	initialized=1;
	GeomObj *w=widget->GetWidget();
	owidget->addObj(w, module_name, &widget_lock);
	widget->Connect(owidget);
        widget->SetRatioR(0.2);
        widget->SetRatioD(0.2);
        widget->SetRatioI(0.2);

#if 0
// let's print the info, just to make sure...
        cerr << "Initializing...\n";
        cerr << "Are we aligned? "<<widget->IsAxisAligned()<<"\n";
        widget->print(cerr);
#endif

    }
    
    if (autoRegister) {	    // they want us to register this??  well, ok...
        auto_register();
	autoRegister=0;
        return;
    }

    // if these points are new, we have to build a new GeomPts object
    if (iPtsHdl.get_rep() != old_pts.get_rep()) {
	Array1<NodeHandle> nodes;
	iPtsHdl->get_surfnodes(nodes);
	geomPts = scinew GeomPts(nodes.size());
	orig_pts.resize(nodes.size());
	trans_pts.resize(nodes.size());
	for (int i=0; i<nodes.size(); i++) {
	    orig_pts[i]=nodes[i]->p;
	    geomPts->add(nodes[i]->p);
	}
    } else {   // have to copy them b/c we're gonna erase them below...
	geomPts = scinew GeomPts(*geomPts);
    }

    if (geom_id) {	     // erase old points
	ogeom->delObj(geom_id);
	geom_id = 0;
    }

    // our defaults values will just be the old values
    widget_lock.read_lock();
    Point p(widget->ReferencePoint());	
    Vector v_r(widget->GetRightAxis());
    Vector v_d(widget->GetDownAxis());
    Vector v_i(widget->GetInAxis());
    widget_lock.read_unlock();
    if (v_r.length2()>2) {
	v_r=v_r/1;
	v_d=v_d/1;
	v_i=v_i/1;
    }
    // if the pts are new, reset widget
    if (iPtsHdl.get_rep() != old_pts.get_rep() || recenter) {
	if (recenter) {
	    BBox b;
	    geomPts->get_bounds(b);
	    w_v = b.diagonal();
	    w_c = p = b.center();
	    w_c = b.center();
	    //Vector v_c = Vector(p.x(), p.y(), p.z()); 
	    v_r = Vector(1,0,0);
	    v_d = Vector(0,1,0);
	    v_i = Vector(0,0,1);
//	cerr << "Moving the widget to starting location!\n";
	    widget->SetPosition(w_c, w_c+(v_r*1), w_c+(v_d*1), w_c+(v_i*1));
	    widget->SetScale
(w_v.length()/20.);
	    recenter=0;
	}
	old_pts = iPtsHdl;
    }

#if 0
// let's print the info, just to make sure...
        cerr << "In execute...\n";
        cerr << "Are we aligned? "<<widget->IsAxisAligned()<<"\n";
        widget->print(cerr);
#endif

    rot_r_x.set(v_r.x()); rot_r_y.set(v_r.y()); rot_r_z.set(v_r.z());
    rot_d_x.set(v_d.x()); rot_d_y.set(v_d.y()); rot_d_z.set(v_d.z());
    rot_i_x.set(v_i.x()); rot_i_y.set(v_i.y()); rot_i_z.set(v_i.z());
    trans_x.set(p.x()); trans_y.set(p.y()); trans_z.set(p.z()); 

    transform_pts(w_c, p, v_r, v_d, v_i);
    reg_error.set(computeError());
    geomPts->pts.resize(0);
    for (int pp=0; pp<trans_pts.size(); pp++) {
	geomPts->add(trans_pts[pp]);
    }
    GeomObj* topobj=scinew GeomMaterial(geomPts, matl);
    geom_id = ogeom->addObj(topobj, module_name);
    ogeom->flushViews();
}	

void Coregister::sendPts(void) {
    TriSurface *os=old_pts->getTriSurface();
    if (!os) {
	error("Can't send non-trisurface!");
	return;
    }
    cerr << "os.points.size:"<<os->points.size()<<"  os.elements.size:"<<os->elements.size()<<"\n";
    TriSurface *ts=new TriSurface();
    ts->points.resize(os->points.size());
    ts->elements.resize(os->elements.size());
    for (int i=0; i<os->elements.size(); i++)
	ts->elements[i]=new TSElement(*(os->elements[i]));
    if (ts->points.size() != trans_pts.size()) {
	cerr << "Points.size: "<<ts->points.size()<<"  TransPts.size:"<<trans_pts.size()<<"\n";
	error("Wrong number of points!");
	return;
    }
    for (i=0; i<ts->points.size(); i++) {
	ts->points[i]=trans_pts[i];
    }
    oport_pts->send(SurfaceHandle(ts));
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
    if (error_metric.get()=="blocks") {
        for (int i=0; i<trans_pts.size(); i+=incr, count++) {
	    err+=mgd->distFast(trans_pts[i]);
        }
        err/=count;
    } else if (error_metric.get()=="dist") {
        for (int i=0; i<trans_pts.size(); i+=incr, count++) {
	    err+=mgd->dist(trans_pts[i]);
        }
        err/=count;
    } else { // (error_metric.get()=="dist2")
        for (int i=0; i<trans_pts.size(); i+=incr, count++) {
	    err+=mgd->dist2(trans_pts[i]);
        }
        err=Sqrt(err/count);
    }
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
    double error_tol;
    int perc;

    reset_vars();
    clString ni(iters.get());
    clString et(tolerance.get());
    clString pe(percent.get());

    if (!ni.get_int(num_iters)) {
        cerr << "Invalid int for number of iterations: "<<ni<<"\n";
        iters.set("100");
        num_iters=100;
    }
    if (!et.get_double(error_tol)) {
        cerr << "Invalid double for error tolerance: "<<et<<"\n";
        tolerance.set("0.0001");
        error_tol=0.0001;
    }
    if (!pe.get_int(perc)) {
        cerr << "Invalid percentage: "<<perc<<"\n";
        percent.set("10");
        perc=10;
    }

    int done=0;
    int currIter=1;
    curr_iter.set("1");
    widget_lock.read_lock();
    Point p(widget->ReferencePoint());	
    Vector v_r(widget->GetRightAxis());
    Vector v_d(widget->GetDownAxis());
    Vector v_i(widget->GetInAxis());
    if (v_r.length2()>2) {
	v_r=v_r/1;
	v_d=v_d/1;
	v_i=v_i/1;
    }
    widget_lock.read_unlock();

    transform_pts(w_c, p, v_r, v_d, v_i, perc);
    double currError=computeError(perc);
    
    Array1<Vector> a_r(13), a_d(13), a_i(13);
    Array1<Point> a_c(13);
    double t_scale=long_edge/5;
    double r_scale=.2*2*PI;
    double rm[3][3];
    int redraw=0;
    int in_a_row=0;

    // as long as we haven't exceded "iters" of iterations, and our
    // error isn't below "tolerance", and we're not "done" for some
    // other reason, we continue to iterate...

    while(currIter<=num_iters && currError>error_tol && !done) {
        // try some new transformations, and pick the one that's best.

        for (int idx=0; idx<7; idx++) {
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
//        cerr << "Errors (iter "<<currIter<<", opt 0): "<<currError<<"\n";
        double allErrors[13];
        allErrors[0]=currError;
        for (idx=1; idx<13; idx++) {
            transform_pts(w_c, a_c[idx], a_r[idx], a_d[idx], a_i[idx], perc);
            allErrors[idx]=computeError(perc);
//            cerr << "Errors (iter "<<currIter<<", opt "<<idx<<"): "<<allErrors[idx]<<"\n";
            if (allErrors[idx]<allErrors[bestIdx]) {
                bestIdx=idx;
            }
        }
        currError=allErrors[bestIdx];
        if (newtonB.get() && error_metric.get() != "blocks" && bestIdx!=0) {
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
//                cerr << "xScale="<<xScale<<"; yScale="<<yScale<<"; zScale="<<zScale<<"\n";
//                cerr << "v: "<<v<<"\n";
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
//                cerr << "rScale="<<rScale<<" ("<<bestRotIdx<<")\n";
                buildRotateMatrix(rm, r_scale*rScale, axis[bestRotIdx-7]);
                new_r = rotateVector(v_r, rm);
                new_d = rotateVector(v_d, rm);
                new_i = rotateVector(v_i, rm);
            }
            transform_pts(w_c, new_c, new_r, new_d, new_i, perc);
            double ee=computeError(perc);
//            cerr << "New error: "<<ee<<"\n";
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
                trans_x.set(p.x()); trans_y.set(p.y()); trans_z.set(p.z()); 
                widget->SetPosition(p, p+(v_r*1), p+(v_d*1), p+(v_i*1));
   	        geomPts = scinew GeomPts(*geomPts);
                if (geom_id) {	     // erase old points
             	    ogeom->delObj(geom_id);
         	    geom_id = 0;
                }
 		geomPts->pts.resize(0);
                for (int pp=0; pp<trans_pts.size(); pp++) {
	            geomPts->add(trans_pts[pp]);
                }
		GeomObj* topobj=scinew GeomMaterial(geomPts, matl);
		geom_id = ogeom->addObj(topobj, module_name);
                ogeom->flushViews();
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
    else if(args[1] == "build_full_mgd")
        {
	    build_full_mgd=1;
            want_to_execute();
        }
    else if(args[1] == "print_mgd")
        {
	    print_mgd=1;
	    want_to_execute();
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
    else if(args[1] == "send_pts")
        {
	    send_pts=1;
            want_to_execute();
        }
    else
        {
            Module::tcl_command(args, userdata);
        }
}

