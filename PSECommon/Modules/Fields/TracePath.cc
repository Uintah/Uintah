//static char *id="@(#) $Id$";

/*
 * TracePath.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1998
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/FLPQueue.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Datatypes/ScalarFieldRGuchar.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/GeomPick.h>
#include <SCICore/Geom/Pt.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/CrowdMonitor.h>
#include <PSECore/Widgets/PointWidget.h>
#include <string.h>

#define PCGVCCOMINDEX(i,x,I,L) ((I) - (i)                                  \
				? ((x) > (I) - ((I)-(i))/(2.0*(L))         \
				   ? (L) - 1                               \
				   : (int)((L)*((x)-(i))/((I)-(i))))       \
				: 0)

#define SWAPMACROXZ(p) if(SWAPXZ) { double dmy=p.z(); p.z(p.x()); p.x(dmy); }

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace PSECore::Widgets;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Containers;
using namespace SCICore::Math;
using namespace SCICore::Geometry;

class TracePath : public Module {
    ScalarFieldIPort *inscalarfield;
    GeometryOPort* ogeom;
    ScalarFieldOPort *outscalarfield;
    ScalarFieldOPort *ofield;
    int init;
    int SWAPXZ;
    Array3<char> parray;
    Array3<int> darray;
    ScalarFieldHandle ufieldHandle;
    ScalarFieldRGuchar* ufield;
    ScalarFieldHandle dfieldHandle;
    ScalarFieldRG* dfield;
    ScalarFieldHandle ssh;
    ScalarFieldRGuchar* sss;
    CrowdMonitor widget_lock;
    PointWidget *swidget;
    Array1<PointWidget *> ewidgets;
    Array1<int> ew_id;
    Array1<Point> visitedpts;
    GeomPts *geompts;
    int pts_idx;
    TCLint updatelines;
    TCLint partial;
    TCLint swapXZ;
    TCLdouble thresh;
    TCLdouble sx, sy, sz;
    TCLdouble eax, eay, eaz;
    TCLdouble ebx, eby, ebz;
    TCLdouble ecx, ecy, ecz;
    TCLdouble tclAlpha, tclBeta;
    Point Rmin, Rmax;
    Point Gmin, Gmax;
    clString tclMsg;
    int in_addlines;
    MaterialHandle greenMatl;	
    MaterialHandle yellowMatl;
    MaterialHandle darkGreenMatl;
    int abort;
    GeomGroup *lines;
    int lines_idx;
    int have_paths;
public:
    TracePath( const clString& id);
    void genPaths(ScalarFieldRG* dfield, Array3<char> &parray,
		  ScalarFieldRGuchar* rgchar, const Point &p, 
		  ScalarFieldRGuchar* sss, Array3<int> &darray, 
		  int beamsize);
    int addLines();
    GeomLines *findLines(Array3<char> &parray, Array3<int> &darray, 
			 ScalarFieldRG *dfield,
			 int si, int sj, int sk, 
			 int ei, int ej, int ek);
    virtual ~TracePath();
    virtual void widget_moved(int last);
    virtual void execute();
    void tcl_command( TCLArgs&, void * );
};



Module* make_TracePath( const clString& id) {
  return new TracePath(id);
}


static clString module_name("TracePath");
static clString widget_name("VolVisLocatorWidget");

TracePath::TracePath(const clString& id) 
: Module("TracePath", id, Filter), swapXZ("swapXZ", id, this),
  thresh("thresh", id, this), partial("partial", id, this),
  updatelines("updatelines", id, this), 
  sx("sx", id, this), sy("sy", id, this), sz("sz", id, this),
  eax("eax", id, this), eay("eay", id, this), eaz("eaz", id, this),
  ebx("ebx", id, this), eby("eby", id, this), ebz("ebz", id, this),
  ecx("ecx", id, this), ecy("ecy", id, this), ecz("ecz", id, this),
  tclAlpha("tclAlpha", id, this), tclBeta("tclBeta", id, this),
    widget_lock("TracePath widget lock")
{
    // Create the input ports
    inscalarfield = scinew ScalarFieldIPort( this, "Scalar Field",
					     ScalarFieldIPort::Atomic);
    
    add_iport(inscalarfield);
    
    // Create the output port
    ogeom = scinew GeometryOPort(this, "Geometry", 
				 GeometryIPort::Atomic);
    add_oport(ogeom);
    outscalarfield = scinew ScalarFieldOPort( this, "Scalar Field",
					     ScalarFieldIPort::Atomic);
    
    add_oport(outscalarfield);
    ofield = scinew ScalarFieldOPort( this, "Scalar Field",
					     ScalarFieldIPort::Atomic);
    
    add_oport(ofield);
    init=0;
    greenMatl = scinew Material(Color(0,0,0),Color(0.1, 1.0, 0.1),
				Color(.5,.5,.5),20);
    yellowMatl = scinew Material(Color(0,0,0),Color(1.0, 1.0, 0.1),
			      Color(.5,.5,.5),20);
    darkGreenMatl = scinew Material(Color(0,0,0),Color(0.0, 0.4, 0.1),
				    Color(.1,.1,.1),20);
    dfield=0;
    ufield=0;
    lines_idx=0;
    pts_idx=0;
    sss=0;
    have_paths=0;
    in_addlines=0;
}

TracePath::~TracePath()
{
}

#define POS_X 1
#define NEG_X 2
#define POS_Y 3
#define NEG_Y 4
#define POS_Z 5
#define NEG_Z 6
#define UNVISITED 0
#define STARTPT 7

class VoxelPath {
public:
    int i;
    int j;
    int k;
    char dir;
    double dist;
    inline int isequal(const VoxelPath& t) const {return (i==t.i && j==t.j && k== t.k);}
};

// Make a fix-lengthed queue, starting at the point p -
//   insert face neighbors, and set 
void TracePath::genPaths(ScalarFieldRG* dfield, Array3<char> &parray,
			 ScalarFieldRGuchar* rgchar, const Point &p, 	
			 ScalarFieldRGuchar *sss, Array3<int> &darray, 
			 int beamsize) {

    double alpha = tclAlpha.get();
    //double beta = tclBeta.get();
    FLPQueue<VoxelPath> flpq(beamsize);
    visitedpts.resize(0);
    abort=0;
    int count=0;
    VoxelPath start;
    start.dir = STARTPT;
    start.dist = 0;
    int nx=dfield->nx;
    int ny=dfield->ny;
    int nz=dfield->nz;
    dfield->locate(p, start.i, start.j, start.k);
    cerr << "Located start pt in cell: ("<<start.i<<","<<start.j<<","<<start.k<<")\n";
    if (start.i>=nx || start.j>=ny || start.k>=nz ||
	start.i<0 || start.j<0 || start.k<0) {
	cerr << "Point not in field!  start.i="<<start.i<<" start.j="<<start.j<<" start.k="<<start.k<<" p="<<p<<"\n";
	return;
    }
    double seedval = rgchar->grid(start.i, start.j, start.k);
    // indicates that we've queued it, but haven't pulled it off yet.
    dfield->grid(start.i, start.j, start.k) = -1;
    int cb;
    VoxelPath b;
    ASSERT(flpq.insert(start,0.1,cb,b));
    Point tmp1, tmp2;
    dfield->get_bounds(tmp1, tmp2);
    Vector v(tmp2-tmp1);
    double longest=rgchar->longest_dimension();
    double maxNum=Max(Max(rgchar->nx, rgchar->ny), rgchar->nz);
    double dx=rgchar->nx/maxNum*longest/v.x();
    double dy=rgchar->ny/maxNum*longest/v.y();
    double dz=rgchar->nz/maxNum*longest/v.z();
//    dx=dy=dz=0;
    cerr << "dx="<<dx <<"  dy="<<dy<<"  dz="<<dz<<"\n";
    int nbrd_i[6];
    int nbrd_j[6];
    int nbrd_k[6];
    char nbr_dir[6];
    double nbr_dd[6];
    nbr_dir[0]=POS_X;
    nbr_dir[1]=NEG_X;
    nbr_dir[2]=POS_Y;
    nbr_dir[3]=NEG_Y;
    nbr_dir[4]=POS_Z;
    nbr_dir[5]=NEG_Z;
    nbr_dd[0]=nbr_dd[1]=dx;
    nbr_dd[2]=nbr_dd[3]=dy;
    nbr_dd[4]=nbr_dd[5]=dz;
//    double threshold = thresh.get();
//    threshold*=threshold*threshold;
    //double maxw=255;
    //double MAXDIST=nx*ny;
    while(!flpq.is_empty()) {
	if ((count % 100) == 0) {
	    reset_vars();
//	    threshold = thresh.get();
//	    threshold *= threshold*threshold;
	    if (abort) {
		abort=0; 
		break;
	    }	

	    if (partial.get()) outscalarfield->send_intermediate(ssh);

	    if (pts_idx) { ogeom->delObj(pts_idx); pts_idx=0; geompts=0; }

	    geompts = scinew GeomPts(visitedpts.size());
	    if (SWAPXZ) {
		for (int i=0; i<visitedpts.size(); i++) {
		    geompts->add(Point(visitedpts[i].z(), visitedpts[i].y(),
				       visitedpts[i].x()));
		}
	    } else {
		for (int i=0; i<visitedpts.size(); i++) {
		    geompts->add(visitedpts[i]);
		}
	    }
	    GeomMaterial *gm = scinew GeomMaterial(geompts, darkGreenMatl);
	    pts_idx = ogeom->addObj(gm, "AV Pts");

	    ogeom->flushViews();

	    // if there's a newpathline flag set, call add_lines
	    if (updatelines.get()) if (addLines()) {
		outscalarfield->send(ssh);
		return;
	    }

#if 0
	    if (!geompts) {
		geompts = scinew GeomPts(0);
		GeomMaterial *gm = scinew GeomMaterial(geompts, darkGreenMatl);
		ogeom->addObj(gm, "AV Pts");
	    }
	    int oldsize=geompts->pts.size()/3;
	    geompts->pts.resize(visitedpts.size()*3);
	    cerr << "Resizing points to "<<visitedpts.size()*3<<"...\n";
	    if (swapXZ.get()) {
		for (int i=oldsize; i<visitedpts.size(); i++) {
		    geompts->pts[i*3]=visitedpts[i].z();
		    geompts->pts[i*3+1]=visitedpts[i].y();
		    geompts->pts[i*3+2]=visitedpts[i].x();
		}
	    } else {
		for (int i=oldsize; i<visitedpts.size(); i++) {
		    geompts->pts[i*3]=visitedpts[i].x();
		    geompts->pts[i*3+1]=visitedpts[i].y();
		    geompts->pts[i*3+2]=visitedpts[i].z();
		}
	    }
	    cerr << "Flushing ogeom.\n";
	    ogeom->flushViews();
#endif
	}
	// pop off the top one (and place it back on the free list)
	double w;
	VoxelPath curr=flpq.pop(w);
	visitedpts.add(sss->get_point(curr.i, curr.j, curr.k));
#if 0
	if (w > threshold) {
	    cerr << "Threshold = "<<threshold<<"  w = "<<w<<"\n";
	    break;
	}
//	cerr << "w="<<w<<"\n";
	if (w>maxw) {
	    cerr << "RAISING MAXW!\n";
	    maxw*=2;
	    for (int i=0; i<sss->nx; i++)
		for (int j=0; j<sss->ny; j++)
		    for (int k=0; k<sss->nz; k++)
			if (sss->grid(i,j,k)) 
			    sss->grid(i,j,k)=127.5+sss->grid(i,j,k)/2.;
	}
#endif

	count++;
	parray(curr.i, curr.j, curr.k) = curr.dir;
	darray(curr.i, curr.j, curr.k) = curr.dist;
//	double newvvv=255*(1-(w+1)/maxw);
//	sss->grid(curr.i, curr.j, curr.k) = newvvv;
	sss->grid(curr.i, curr.j, curr.k) = 255-(curr.dist*0.01);
	if ((count % 200) == 0) cerr << "   w="<<w<<"  curr.dist="<<curr.dist<<"\n";
	dfield->grid(curr.i, curr.j, curr.k)=w;
	// now look at the neighbors
	// enqueue them if they haven't been visited and we have space left...
	// ...check their weight if they're in the queue and update if closer
	int idx;
	for (idx=0; idx<6; idx++) {
	    nbrd_i[idx]=curr.i;
	    nbrd_j[idx]=curr.j;
	    nbrd_k[idx]=curr.k;
	}
	nbrd_i[0]++; nbrd_i[1]--;
	nbrd_j[2]++; nbrd_j[3]--;
	nbrd_k[4]++; nbrd_k[5]--;
	
	int currVal = rgchar->grid(curr.i, curr.j, curr.k);
	for (idx=0; idx<6; idx++) {
	    VoxelPath nbr;
	    nbr.i=nbrd_i[idx];
	    nbr.j=nbrd_j[idx];
	    nbr.k=nbrd_k[idx];
	    nbr.dir=nbr_dir[idx];
	    nbr.dist=curr.dist+1;
	    if (nbr.i>=0 && nbr.i<dfield->nx &&
		nbr.j>=0 && nbr.j<dfield->ny &&
		nbr.k>=0 && nbr.k<dfield->nz) {

		// here's our cost function -- the "edge length" to our nbrs
		// "w" was the weight to have come this far; 
		// "locDist" will be the "edge length" to each neighbor
		double eucDist;
		double gradDist;
		double valDist;
		double totalDist;
		double dirDist=0;

		// penalty for straying from seedpoint intensity
		uchar nbrVal = rgchar->grid(nbr.i, nbr.j, nbr.k);
		valDist = Abs(seedval - nbrVal);
#if 0
		valDist = seedval - nbrVal + 50;
		double valDistPos = fabs(valDist);
		valDist = (beta*valDist) + (1-beta)*valDistPos;
		if (valDist < 0) valDist=0;
#endif
		// penalty for traveling any distances (enforces pos. cost)
		eucDist = nbr_dd[idx];

		// add in a penalty for high grad-mag -- GK
//		gradDist =  (currVal-nbrVal)*(currVal-nbrVal)/
//		    (nbr_dd[idx]*nbr_dd[idx]);
		gradDist =  (currVal-nbrVal)*(currVal-nbrVal)*(currVal-nbrVal)/
		    (nbr_dd[idx]*nbr_dd[idx]*nbr_dd[idx]);

		if (gradDist < 0) gradDist/=3;
		gradDist = Abs(gradDist);
//		gradDist-=30;
		if (gradDist < 0) gradDist = 0;

		// add a penalty for going in the wrong direction!
		if (nbr.j < start.j) dirDist=1000;
		// add a penalty for low curvature -- GK

		// total distance is sum of local penalties and previous dist
//		totalDist = valDist + gradDist + eucDist + w;
		totalDist = valDist*alpha + gradDist*(1-alpha) + 
		    eucDist/10000. + w + dirDist;

//		locDist = currVal-rgchar->grid(nbr.i, nbr.j, nbr.k);
//		locDist = locDist*locDist + nbr_dd[idx] + w;
//		if (locDist < 0) locDist=0;
//		locDist += w + 1/sqrt(1+curr.dist);
//		locDist += w + (MAXDIST-curr.dist)*(MAXDIST-curr.dist);
//		locDist += w + 100/(5.+curr.dist);
//		locDist += nbr_dd[idx];

		if (dfield->grid(nbr.i, nbr.j, nbr.k)==0) {
		    // hasn't been queued or visited yet -- try to put on the
		    // priority queue
		    VoxelPath bumped;
		    int caused_bump;
		    if (flpq.insert(nbr, totalDist, caused_bump, bumped)) {
			if (caused_bump) {
			    dfield->grid(bumped.i, bumped.j, bumped.k)=0;
			}
			dfield->grid(nbr.i, nbr.j, nbr.k)=-1;
		    }
		} else if (dfield->grid(nbr.i, nbr.j, nbr.k)==-1) {
		    // is in the queue already - see if new distance is closer,
		    // if so, reorder position in queue
		    flpq.update_weight(nbr, totalDist);
		}
	    }
	}
    }
    outscalarfield->send(ssh);
}

int TracePath::addLines() {
    if (!have_paths) return 0;
    in_addlines=1;
    if (lines_idx) {
	ogeom->delObj(lines_idx);
	lines_idx=0;
    }
    lines = scinew GeomGroup();
    int num=ewidgets.size();
    Point start(swidget->GetPosition());
    SWAPMACROXZ(start)
    int si, sj, sk;
    dfield->locate(start, si, sj, sk);
    for (int i=0; i<num; i++) {
	Point end(ewidgets[i]->GetPosition());
	SWAPMACROXZ(end)
	int ei, ej, ek;
	dfield->locate(end, ei, ej, ek);
	GeomLines *gl = findLines(parray, darray, dfield, ei, ej, ek, 
				  si, sj, sk);
	if (gl) {	
	    lines->add(gl);
	}
    }
    int retval=0;
    if (lines->size()) {
	lines_idx = ogeom->addObj(lines, "AV Paths");
	ogeom->flushViews();
	retval=1;
    } else {
	delete lines;
    }
    in_addlines=0;
    return retval;
}	

GeomLines *TracePath::findLines(Array3<char> &parray, Array3<int>&/*darray*/, 
				ScalarFieldRG *dfield,
				int si, int sj, int sk, 
				int ei, int ej, int ek) {
    cerr << "In findLines.  Finding lines from: ("<<si<<","<<sj<<","<<sk<<") to ( "<<ei<<","<<ej<<","<<ek<<")\n";
    GeomLines *l = scinew GeomLines;
    Point p1=dfield->get_point(si,sj,sk);
    SWAPMACROXZ(p1)
    Point p2;
    while (si != ei || sj != ej || sk != ek) {
	char dir = parray(si,sj,sk);
//	cerr << "Current line pt: ("<<si<<","<<sj<<","<<sk<<") moving in ";
	if (dir == POS_X) { 
//	    cerr << "Pos X... ";
	    si--;
	} else if (dir == NEG_X) {
//	    cerr << "Neg X... ";
	    si++;
	} else if (dir == POS_Y) {
//	    cerr << "Pos Y... ";
	    sj--;
	} else if (dir == NEG_Y) {
//	    cerr << "Neg Y... ";
	    sj++;
	} else if (dir == POS_Z) {
//	    cerr << "Pos Z... ";
	    sk--;
	} else if (dir == NEG_Z) {
//	    cerr << "Neg Z... ";
	    sk++;
	} else if (dir == UNVISITED) {
	    cerr << "Error -- node wasn't ever visited!\n";
	    return (GeomLines*)0;
	} else if (dir == STARTPT) {
	    cerr << "Error -- node is start pt!\n";
	    return (GeomLines*)0;
	}
//	cerr << "\n";
	p2=dfield->get_point(si,sj,sk);
	SWAPMACROXZ(p2)
	l->add(p1,p2);
	p1=p2;
    }
    if (!l->pts.size()) { delete l; l=0; }
    return l;
}

void TracePath::execute(void) {
    ScalarFieldHandle sfield;
    reset_vars();
    SWAPXZ = swapXZ.get();

    if (!inscalarfield->get(sfield)||!sfield.get_rep()||!sfield->getRGBase())
	return;

    ScalarFieldRGuchar *rgchar = sfield->getRGBase()->getRGUchar();
    
    if (!rgchar) {
	cerr << "Not a char field!\n";
	return;
    }
    
    if (!init) {
	init=1;
	ufieldHandle = sfield;
	ufield = rgchar;
	swidget=scinew PointWidget(this, &widget_lock, 0.2);
	// make the start point green
	swidget->SetMaterial(0, greenMatl);
	GeomObj *w=swidget->GetWidget();
	ogeom->addObj(w, "StartPt", &widget_lock);
	swidget->Connect(ogeom);
	
	// DAVE: HACK!
	//    sfield->get_bounds(Rmin, Rmax);
	
	Point tmp1, tmp2;
	sfield->get_bounds(Rmin, Rmax);
	Gmin=Rmin;
	Gmax=Rmax;
	SWAPMACROXZ(Gmin)
	SWAPMACROXZ(Gmax)
	cerr << "Original field: Rmin="<<Rmin<<"  Rmax="<<Rmax<<"\n";
	swidget->SetPosition(Interpolate(Gmin,Gmax,0.6));
	swidget->SetScale(sfield->longest_dimension()/150.0);
    }

    if (tclMsg == "") return;

    if (tclMsg == "getstart") {
	Point p;
	p = swidget->GetPosition();
	SWAPMACROXZ(p);
	sx.set(p.x());
	sy.set(p.y());
	sz.set(p.z());
    } else if (tclMsg == "setstart") {
	Point p;
	p.x(sx.get());
	p.y(sy.get());
	p.z(sz.get());
	SWAPMACROXZ(p);
	swidget->SetPosition(p);
    } else if (tclMsg == "getendpta") {
	if (ewidgets.size() >= 1) {
	    Point p;
	    p = ewidgets[0]->GetPosition();
	    SWAPMACROXZ(p);
	    eax.set(p.x());
	    eay.set(p.y());
	    eaz.set(p.z());
	} else {
	    cerr << "Error -- don't have EndPt 1 yet...\n";
	}
    } else if (tclMsg == "setendpta") {
	if (ewidgets.size() >= 1) {
	    Point p;
	    p.x(eax.get());
	    p.y(eay.get());
	    p.z(eaz.get());
	    SWAPMACROXZ(p);
	    ewidgets[0]->SetPosition(p);
	} else {
	    cerr << "Error -- don't have EndPt 1 yet...\n";
	}
    } else if (tclMsg == "getendptb") {
	if (ewidgets.size() >= 2) {
	    Point p;
	    p = ewidgets[1]->GetPosition();
	    SWAPMACROXZ(p);
	    ebx.set(p.x());
	    eby.set(p.y());
	    ebz.set(p.z());
	} else {
	    cerr << "Error -- don't have EndPt 2 yet...\n";
	}
    } else if (tclMsg == "setendptb") {
	if (ewidgets.size() >= 2) {
	    Point p;
	    p.x(ebx.get());
	    p.y(eby.get());
	    p.z(ebz.get());
	    SWAPMACROXZ(p);
	    ewidgets[1]->SetPosition(p);
	} else {
	    cerr << "Error -- don't have EndPt 2 yet...\n";
	}
    } else if (tclMsg == "getendptc") {
	if (ewidgets.size() >= 3) {
	    Point p;
	    p = ewidgets[2]->GetPosition();
	    SWAPMACROXZ(p);
	    ecx.set(p.x());
	    ecy.set(p.y());
	    ecz.set(p.z());
	} else {
	    cerr << "Error -- don't have EndPt 3 yet...\n";
	}
    } else if (tclMsg == "setendptc") {
	if (ewidgets.size() >= 3) {
	    Point p;
	    p.x(ecx.get());
	    p.y(ecy.get());
	    p.z(ecz.get());
	    SWAPMACROXZ(p);
	    ewidgets[2]->SetPosition(p);
	} else {
	    cerr << "Error -- don't have EndPt 3 yet...\n";
	}
    } else if (tclMsg == "add_endpt") {
	int num=ewidgets.size();
	ewidgets.add(scinew PointWidget(this, &widget_lock, 0.2));
	// make the start point green
	ewidgets[num]->SetMaterial(0, yellowMatl);
	GeomObj *w=ewidgets[num]->GetWidget();
	ew_id.add(ogeom->addObj(w, clString(clString("EndPt")+to_string(num)), 
				&widget_lock));
	ewidgets[num]->Connect(ogeom);
	Point p=Interpolate(Gmin, Gmax, 0.65);
	ewidgets[num]->SetPosition(p);
	ewidgets[num]->SetScale(ufield->longest_dimension()/150.0);
    } else if (tclMsg == "new_field") {
	ufieldHandle = sfield;
	ufield = rgchar;
	Point Rmin, Rmax;
	sfield->get_bounds(Rmin, Rmax);
	Gmin=Rmin;
	Gmax=Rmax;
	SWAPMACROXZ(Gmin)
	SWAPMACROXZ(Gmax)
	cerr << "New field: Rmin="<<Rmin<<"  Rmax="<<Rmax<<"\n";	
    } else if (tclMsg == "clear_endpts") {
	for (int i=0; i<ewidgets.size(); i++) {
	    ogeom->delObj(ew_id[i]);
	    delete(ewidgets[i]);
	}
	ew_id.resize(0);
	ewidgets.resize(0);
    } else if (tclMsg == "add_lines") {
	addLines();
    } else if (tclMsg == "clear_lines") {
	if (lines_idx) {
	    ogeom->delObj(lines_idx);
	    lines_idx=0;
	    ogeom->flushViews();
	}
    } else if (tclMsg == "gen_paths") {
	cerr << "Generating paths.\n";

	long int nnn = ufield->nx*ufield->ny*ufield->nz;

	if (!dfield) {
	    dfieldHandle = dfield = scinew ScalarFieldRG;
	    cerr << "Allocating dfield...\n";
	    dfield->resize(ufield->nx, ufield->ny, ufield->nz);
	    dfield->set_bounds(Rmin, Rmax);
	    parray.newsize(ufield->nx, ufield->ny, ufield->nz);
	    darray.newsize(ufield->nx, ufield->ny, ufield->nz);
	}
	cerr << "Initializing parray...\n";
	have_paths=1;
	memset(&(parray(0,0,0)), 0, nnn*sizeof(char));
//	parray.initialize(UNVISITED);

	cerr << "Initializing darray...\n";

	memset(&(darray(0,0,0)), 0, nnn*sizeof(int));
//	darray.initialize(0);

	cerr << "Initializing dfield...\n";

	memset(&(dfield->grid(0,0,0)), 0, nnn*sizeof(double));
//	dfield->grid.initialize(-1);

	if (!sss) {
	    ssh = sss = scinew ScalarFieldRGuchar;
	    sss->resize(ufield->nx, ufield->ny, ufield->nz);
	    sss->set_bounds(Rmin,Rmax);
	}
	cerr << "Initializing sss...\n";

	memset(&(sss->grid(0,0,0)), 0, nnn*sizeof(uchar));
//	sss->grid.initialize(0);

	Point p(swidget->GetPosition());
	cerr << "P="<<p<<"\n";
	SWAPMACROXZ(p)
	cerr << "P="<<p<<"\n";
	int bmsz = pow(10., thresh.get());
	cerr << "Beamsize = "<<bmsz<<"\n";
	genPaths(dfield, parray, rgchar, p, sss, darray, bmsz);
	cerr << "Done generating all paths!\n";
	ofield->send(dfieldHandle);
	// go through all the endpoint widget positions, and draw the lines
	// from start to end for each.
    } else if (tclMsg == "clear_paths") {
	cerr << "Clearing paths.\n";
	have_paths=0;
	if (sss) {
	    long int nnn = sss->nx*sss->ny*sss->nz;
	    memset(&(sss->grid(0,0,0)), 0, nnn*sizeof(uchar));
	    outscalarfield->send(ssh);
	}
	visitedpts.resize(0);
	if (pts_idx) {
	    ogeom->delObj(pts_idx);
	    geompts=0;
	    pts_idx=0;
	}
    } else {
	cerr << "Unknown tclMsg: "<<tclMsg<<"\n";
    }
    tclMsg = "";
}

void TracePath::widget_moved(int last) {
    // see if we've computed a path field yet
    // if so, see if it's one of the endpoints that's moved
    // if so, call want_to_execute with add_lines
	    // if there's a newpathline flag set, call add_lines
    cerr << "updatelines.get() == "<<updatelines.get()<<"\n";
    reset_vars();
    if (updatelines.get()) {
	if (last || !in_addlines) {
	    tclMsg = "add_lines";
	    want_to_execute();
	}
    }
}

void TracePath::tcl_command(TCLArgs& args, void* userdata) {
    if (args[1] == "gen_paths") {
	tclMsg="gen_paths";
	want_to_execute();
    } else if (args[1] == "clear_paths") {
	tclMsg="clear_paths";
	want_to_execute();
    } else if (args[1] == "add_endpt") {
	tclMsg="add_endpt";
	want_to_execute();
    } else if (args[1] == "clear_endpts") {
	tclMsg="clear_endpts";
	want_to_execute();
    } else if (args[1] == "add_lines") {
	tclMsg="add_lines";
	want_to_execute();
    } else if (args[1] == "clear_lines") {
	tclMsg="clear_lines";
	want_to_execute();
    } else if (args[1] == "new_field") {
	tclMsg="new_field";
	want_to_execute();
    } else if (args[1] == "abort") {
	abort=1;
    } else if (args[1] == "getstart") {
	tclMsg="getstart";
	want_to_execute();
    } else if (args[1] == "setstart") {
	tclMsg="setstart";
	want_to_execute();
    } else if (args[1] == "getendpta") {
	tclMsg="getendpta";
	want_to_execute();
    } else if (args[1] == "setendpta") {
	tclMsg="setendpta";
	want_to_execute();
    } else if (args[1] == "getendptb") {
	tclMsg="getendptb";
	want_to_execute();
    } else if (args[1] == "setendptb") {
	tclMsg="setendptb";
	want_to_execute();
    } else if (args[1] == "getendptc") {
	tclMsg="getendptc";
	want_to_execute();
    } else if (args[1] == "setendptc") {
	tclMsg="setendptc";
	want_to_execute();
    } else {
        Module::tcl_command(args, userdata);
    }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.8  1999/09/08 02:26:34  sparker
// Various #include cleanups
//
// Revision 1.7  1999/08/29 00:46:40  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.6  1999/08/25 03:47:48  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.5  1999/08/23 06:30:31  sparker
// Linux port
// Added X11 configuration options
// Removed many warnings
//
// Revision 1.4  1999/08/19 23:17:46  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:42  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:29  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:43  mcq
// Initial commit
//
// Revision 1.2  1999/04/28 20:51:12  dav
// deleted some files that are dependent on DaveW files
//
// Revision 1.1.1.1  1999/04/24 23:12:32  dav
// Import sources
//
//
