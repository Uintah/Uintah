/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


/*
 *  PlanarContoursToSegVol.cc:  Scanline algorithm for building a segmented vol from contours
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   August 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <DaveW/Datatypes/General/ContourSet.h>
#include <DaveW/Datatypes/General/ContourSetPort.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/Array2.h>
#include <SCICore/Datatypes/ScalarFieldRGchar.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <iostream>

namespace DaveW {
namespace Modules {

using namespace DaveW::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Containers;
using namespace SCICore::Geometry;
using namespace SCICore::Math;
using namespace SCICore::Thread;
using std::cerr;

class Scanline {
    typedef struct tEdge {
	int yUpper;
	float xIntersect, dxPerScan;
	struct tEdge *next;
    } SLEdge;
    int nrows;
    int ncols;
    void insertEdge(SLEdge *list, SLEdge *SLEdge);
    void makeEdgeRec(Point lower, Point upper, SLEdge *Edge, SLEdge *Edges[]);
    void buildEdgeList(Array1<Point> pts, SLEdge *Edges[]);
    void buildActiveList(int scan, SLEdge *active, SLEdge *Edges[]);
    void fillScan(int scan, SLEdge *active, char val, Array2<char> &im);
    void deleteAfter(SLEdge *q);
    void updateActiveList(int scan, SLEdge *active);
    void resortActiveList(SLEdge *active);
    void fillPgon(Array1<Point> p, char val, Array2<char> &im);
public:
    Scanline(Array2<char> &pixels, Array1<Array1<Point> > polys);
};


void Scanline::insertEdge(SLEdge *list, SLEdge *edge) {
    SLEdge *p, *q = list;
    p = q->next;
    while(p != 0) {
	if(edge->xIntersect < p->xIntersect)
	    p = 0;
	else {
	    q = p;
	    p = p->next;
	}
    }
    edge->next = q->next;
    q->next = edge;
}

void Scanline::makeEdgeRec(Point lower, Point upper, SLEdge *edge, 
			   SLEdge *edges[]) {
    
    /* clip to the bottom and top edges of the image */
    if(lower.y() >= nrows)
	return;
    if(upper.y() <= 0)
	return;
    
    if(upper.x() == lower.x())
	edge->dxPerScan = 0.0;
    else
	edge->dxPerScan = (float)(upper.x() - lower.x()) / (float)(upper.y() - lower.y());
    
    /* Change here to deal with lower boundary clipping (upper boundary clipping not necessary) */
    if(lower.y() < 0) {
	lower.x(lower.x() + (int)((-lower.y()) * edge->dxPerScan + 0.5));
	lower.y(0);
    }
    
    /* Change here to reduce the right edge of the polygon by 1 for all cases */
    //  edge->xIntersect = lower.x() + edge->dxPerScan * 0.5;
    edge->xIntersect = lower.x();
    
    /* Change here to reduce the height of the filled polygon by 1 for all cases */
    
    /* H & B original
       if (upper.y() < yComp)
       edge->yUpper = upper.y() - 1;
       else
       edge->yUpper = upper.y();
       */
    
    /* my fix */
    edge->yUpper = upper.y() - 1;
    
    insertEdge(edges[(int)(lower.y())], edge);
    
}

void Scanline::buildEdgeList(Array1<Point> pts, SLEdge *edges[]) {
    SLEdge *edge;
    Point v1, v2;
    int cnt=pts.size();
    int i;

    v1=pts[cnt-1];
    
    for(i=0;i<cnt;i++) {
	v2=pts[i];
	if(((int)v1.y()) != ((int)v2.y())) { /* not a horizontal line */
	    edge = (SLEdge *)malloc(sizeof(SLEdge));
	    if(v1.y() < v2.y())
		makeEdgeRec(v1, v2, edge, edges);
	    else
		makeEdgeRec(v2, v1, edge, edges);
	}
	v1 = v2;
    }
}

void Scanline::buildActiveList(int scan, SLEdge *active, SLEdge *edges[]) {
    SLEdge *p, *q;
    
    p = edges[scan]->next;
    while(p) {
	q = p->next;
	insertEdge(active, p);
	p = q;
    }
}

void Scanline::fillScan(int scan, SLEdge *active, char val, Array2<char> &im) {
    SLEdge *p1, *p2;
    int i, f;
    
    p1 = active->next;
    while(p1) {
	p2 = p1->next;
	
	f = i = 0;
	
	/* check the left side of the span */
	if(p1->xIntersect >= ncols) {
	    p1 = p2->next;
	    continue; /* don't need to draw this span */
	}
	else if(p1->xIntersect >= 0)
	    i += (int)p1->xIntersect;  /* truncate */
	
	/* check the right side of the span */
	if(p2->xIntersect >= ncols)
	    f += ncols;
	else if(p2->xIntersect > 0)
	    f += (int)p2->xIntersect;  /* truncate */
	else {
	    p1 = p2->next;
	    continue; /* don't need to draw this span */
	}
	
	for(;i<f;i++)
	    im(i,scan)=val;
	
	p1 = p2->next;
    }
    
}

void Scanline::deleteAfter(SLEdge *q) {
    SLEdge *p = q->next;
    
    q->next = p->next;
    free(p);
}

void Scanline::updateActiveList(int scan, SLEdge *active) {
    SLEdge *q = active, *p = active->next;
    
    while(p) {
	if(scan >= p->yUpper) {
	    p = p->next;
	    deleteAfter(q);
	}
	else {
	    p->xIntersect = p->xIntersect + p->dxPerScan;
	    q = p;
	    p = p->next;
	}
    }
}


void Scanline::resortActiveList(SLEdge *active) {
    SLEdge *q, *p = active->next;
    
    active->next = 0;
    while(p) {
	q = p->next;
	insertEdge(active, p);
	p = q;
    }
}

void Scanline::fillPgon(Array1<Point> p, char val, Array2<char> &im) {
    SLEdge **edges, *active;
    int i, scan;
    edges = (SLEdge **)malloc(sizeof(SLEdge *) * nrows);
    if(edges == 0) {
	printf("Allocation error\n");
	exit(0);
    }
    
    for(i=0;i<nrows;i++) {
	edges[i] = (SLEdge *)malloc(sizeof(SLEdge));
	edges[i]->next = 0;
    }
    
    buildEdgeList(p, edges);
    active = (SLEdge *)malloc(sizeof(SLEdge));
    active->next = 0;
    
    for(scan = 0; scan < nrows; scan++) {
	buildActiveList(scan, active, edges);
	if(active->next) {
	    fillScan(scan, active, val, im);
	    updateActiveList(scan, active);
	    resortActiveList(active);
	}
    }
    
    /* free malloced edges! */
    for(i=0;i<nrows;i++)
	free(edges[i]);
    free(edges);
}

Scanline::Scanline(Array2<char> &pixels, Array1<Array1<Point> > polys) {
    ncols=pixels.dim1();
    nrows=pixels.dim2();
    for (int i=0; i<polys.size(); i++)
	if (polys[i].size()>=3) {
//	    cerr << "Filling poly "<<i+1<<"...\n";
//	    for (int j=0; j<polys[i].size(); j++)
//		cerr << "   "<<polys[i][j]<<"\n";
//	    cerr <<"\n";
	    fillPgon(polys[i], i+1, pixels);
	}
#if 0
    cerr << "Here's this slice...\n";
    for (i=0; i<nrows; i++) {
	for (int j=0; j<ncols; j++) {
	    cerr << (char) (pixels(i,j)+'0') << " ";
	}
	cerr << "\n";
    }
    cerr << "\n\n";
#endif
}

class PlanarContoursToSegVol : public Module {
    ContourSetIPort* incontour;
    ScalarFieldOPort* ofield;
    TCLstring nxTCL;
    TCLstring nyTCL;
    clString execMsg;
    int gen;
    int nx;
    int ny;
    ScalarFieldHandle sfH;
public:
    PlanarContoursToSegVol(const clString&);
    virtual ~PlanarContoursToSegVol();
    virtual void execute();
    virtual void tcl_command(TCLArgs& args, void* userdata);
};

extern "C" Module* make_PlanarContoursToSegVol(const clString& get_id())
{
    return new PlanarContoursToSegVol(get_id());
}

PlanarContoursToSegVol::PlanarContoursToSegVol(const clString& get_id())
: Module("PlanarContoursToSegVol", get_id(), Filter), nxTCL("nxTCL", get_id(), this), 
  nyTCL("nyTCL", get_id(), this)

{
    // Create the input port
    incontour=new ContourSetIPort(this, "ContourSet", 
				  ContourSetIPort::Atomic);

    add_iport(incontour);
    ofield=new ScalarFieldOPort(this, "Field", ScalarFieldIPort::Atomic);
    add_oport(ofield);
    gen=-1;
}


PlanarContoursToSegVol::~PlanarContoursToSegVol()
{
}


void
PlanarContoursToSegVol::execute()
{
    ContourSetHandle contours;
    if (!incontour->get(contours)) return;
    int genTmp=contours->generation;
    int newNx=atoi(nxTCL.get()());
    int newNy=atoi(nyTCL.get()());
    if (execMsg=="" && genTmp==gen && (newNx==nx) && (newNy==ny)){
	ofield->send(sfH);
	return;
    }
    execMsg="";
    gen=genTmp;
    nx=newNx;
    ny=newNy;
    ScalarFieldRGchar *sf=new ScalarFieldRGchar;
    Point min(contours->bbox.min());
    Point max(contours->bbox.max());

    cerr << "min="<<min<<"  max="<<max;

    int nz=(int)((max.z()-min.z())/contours->space+3);
    cerr << "nz="<<nz<<"\n";
    Vector diag(max-min+Vector(1,1,1));
    
    cerr << " first diag="<<diag<<"\n";
    diag.x(((diag.x()*nx/(nx-2))-diag.x())/2);
    diag.y(((diag.y()*ny/(ny-2))-diag.y())/2);
    diag.z(contours->space*2);
    
    cerr << "  diag="<<diag<<"\n";
    sf->set_bounds(min-diag/2, max+diag/2);
    sf->resize(nx,ny,nz);
    sf->grid.initialize('0');

    // for each contour of each material, run a scanline algorithm to fill
    // the voxels of that slice with that material idx

    // first copy the contour points and rescale them so the BBox goes from
    // (1,1,1) to (nx-2,ny-2,nz-2)

    Array1<Array1<Array1<Point> > > pts(contours->levels);
    int i;
    for (i=0; i<pts.size(); i++)
	for (int j=0; j<pts[i].size(); j++)
	    for (int k=0; k<pts[i][j].size(); k++) {
		pts[i][j][k].x((pts[i][j][k].x()-min.x())/
			       (max.x()-min.x())*(nx-2)+1.4999);
		pts[i][j][k].y((pts[i][j][k].y()-min.y())/
			       (max.y()-min.y())*(ny-2)+1.4999);
	    }

    Array2<char> slice;
    slice.newsize(nx,ny);
    for (i=0; i<pts.size(); i++) {
	slice.initialize(0);
	Scanline s(slice, pts[i]);
	for (int j=0; j<nx; j++)
	    for (int k=0; k<ny; k++) 
		if (slice(j,k)) {
//		    cerr << "["<<i<<","<<j<<","<<k<<"] was "<<sf->grid(i,j,k);
		    sf->grid(j,k,i+1)+=contours->level_map[i][slice(j,k)-1];
//		    cerr << " now it's "<<sf->grid(i,j,k)<<"\n";
		}
    }
    sfH=sf;
    ofield->send(sfH);
}

void PlanarContoursToSegVol::tcl_command(TCLArgs& args, void* userdata) {
    if (args[1] == "execute") {
	execMsg="execute";
	want_to_execute();
    } else {
        Module::tcl_command(args, userdata);
    }
}

} // End namespace Modules
} // End namespace DaveW
