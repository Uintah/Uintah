
/*
 *  AugmentSurf.cc:  Refine/Decimate a surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/BasicSurfaces.h>
#include <Datatypes/ManhattanDist.h>
#include <Datatypes/Surface.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/TriSurface.h>
#include <Geometry/BBox.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

#include <Classlib/Array2.h>

class AugmentSurf : public Module {
    SurfaceIPort* isurface;
    SurfaceOPort* osurface;
    TriSurface* last_tsIn;
    SurfaceHandle last_osh;
    int last_gs;
    double spacing;
    TCLint tclGridSize;
public:
    AugmentSurf(const clString& id);
    AugmentSurf(const AugmentSurf&, int deep);
    virtual ~AugmentSurf();
    virtual Module* clone(int deep);
    virtual void execute();
    void voxelCoalesce(TriSurface* ts, int gs);
    void temp();
};

extern "C" {
Module* make_AugmentSurf(const clString& id)
{
    return scinew AugmentSurf(id);
}
};

AugmentSurf::AugmentSurf(const clString& id)
: Module("AugmentSurf", id, Filter), last_tsIn(0), last_osh(0), spacing(-1),
  last_gs(-1), tclGridSize("tclGridSize", id, this)
{
    // Create the input port
    isurface=scinew SurfaceIPort(this, "SurfaceIn", SurfaceIPort::Atomic);
    add_iport(isurface);

    // Create the output port
    osurface=scinew SurfaceOPort(this, "SurfaceOut", SurfaceIPort::Atomic);
    add_oport(osurface);
}

AugmentSurf::AugmentSurf(const AugmentSurf&copy, int deep)
: Module(copy, deep), last_tsIn(copy.last_tsIn), last_osh(copy.last_osh), 
  spacing(copy.spacing), last_gs(copy.last_gs), 
  tclGridSize("tclGridSize", id, this)
{
    NOT_FINISHED("AugmentSurf::AugmentSurf");
}

AugmentSurf::~AugmentSurf()
{
}

Module* AugmentSurf::clone(int deep)
{
    return scinew AugmentSurf(*this, deep);
}

void AugmentSurf::execute()
{
    SurfaceHandle surf;
    if (!isurface->get(surf)){
	return;
    }
    TriSurface* tsIn=surf->getTriSurface();
    if (!tsIn) {
	cerr << "Error: can only handle TriSurfaces in AugmentSurf::execute\n";
	return;
    }
    int gs = tclGridSize.get();
    if (gs==last_gs && tsIn==last_tsIn) return;
    last_gs=gs;
    last_tsIn=tsIn;

    TriSurface* tsOut=scinew TriSurface();
    tsOut->points.resize(tsIn->points.size());
    for (int i=0; i<tsIn->points.size(); i++)
	tsOut->points[i] = tsIn->points[i];
    tsOut->elements.resize(tsIn->elements.size());
    for (i=0; i<tsIn->elements.size(); i++)
	tsOut->elements[i] = new TSElement(*(tsIn->elements[i]));
    voxelCoalesce(tsOut, gs);

cerr << "Input surface: "<<tsIn->points.size()<<" points, "<<tsIn->elements.size()<<" elements.\n";
cerr << "Output surface: "<<tsOut->points.size()<<" points, "<<tsOut->elements.size()<<" elements.\n";

    last_osh=tsOut;
    osurface->send(last_osh);
}

void AugmentSurf::temp() {
    int i=3;
    if (i>2) {
	cerr << "trying an Array3<int>...\n";
	Array3<int > l(12,15,15);
	l(3,2,5)=3;
    }
    if (i>0) {
	cerr << "trying an array1 of array1 's.\n";
	Array1<Array1<int> > l[12];
	l[3].add(3);
    }

    if (i>0) {
	cerr << "trying an array2 of array1 's.\n";
	Array2<Array1<int> > l(8,12);
	l(3,4).add(3);
    }

    if (i>0) {
	cerr << "trying an array3 of array1 's.\n";
	Array3<Array1<int> > l(12,12,13);
	l(3,4,6).add(3);
    }

    if (i>1) {
	cerr << "trying dynamic allocation...\n";
	Array3<Array1<int> >* l = new Array3<Array1<int> > (12,15,15);
	((*l)(3,2,5)).add(3);
//	delete l;
    }
    cerr << "seems to have made it out alive!\n";
    return;
}

void AugmentSurf::voxelCoalesce(TriSurface* ts, int gs) {

    BBox bb;
    for (int aa=0; aa<ts->points.size(); aa++) {
        bb.extend(ts->points[aa]);
    }

#if 0
    cerr << "trying temp!\n";
    temp();
    cerr << "made it out!!!!\n";
    return;
#endif

    Array1<int> ptRemoved(ts->points.size());
    Array1<int> ptNewIdx(ts->points.size());
    Array1<int> ptMap(ts->points.size());
    for (int i=0; i<ptRemoved.size(); i++) {
	ptRemoved[i]=0;
	ptNewIdx[i]=-1;
    }

    ManhattanDist* md = new ManhattanDist(ts->points, gs, 0, 
			       bb.min().x(), bb.min().y(), bb.min().z(), 
			       bb.max().x(), bb.max().y(), bb.max().z());
    for (i=0; i<md->nx; i++) {
	for (int j=0; j<md->ny; j++) {
	    for (int k=0; k<md->nz; k++) {
		if (md->closestNodeIdx(i,j,k).size() > 1) {
		    Array1<int> idx(md->closestNodeIdx(i,j,k));
//cerr << idx.size() << " points at ("<<i<<","<<j<<","<<k<<") ["<<idx[0];
		    BBox b;
		    if (ptRemoved[idx[0]]) {
//			cerr << "\n\nHEY!!!!  Can't do that!!!\n\n";
		    }
		    b.extend(ts->points[idx[0]]);
		    for (int l=1; l<idx.size(); l++) {
			if (ptRemoved[idx[l]]) {
//			    cerr << "\n\nHEY!!!!  Can't do that!!!\n\n";
			}
//			cerr << " " << idx[l];
			b.extend(ts->points[idx[l]]);
			ptRemoved[idx[l]]=1;
			ptNewIdx[idx[l]]=idx[0];
		    }
//		    cerr << "]  point "<<idx[0]<<" moved from: "<<ts->points[idx[0]]<<" to "<<b.center()<<"  ";
		    ts->points[idx[0]] = b.center();
		}
	    }
	}
    }

    int newIdx, oldIdx;
    for (newIdx=oldIdx=0; oldIdx<ts->points.size(); oldIdx++) {
	if (!ptRemoved[oldIdx]) {
	    ptMap[oldIdx]=newIdx;
	    ts->points[newIdx]=ts->points[oldIdx];
	    newIdx++;
	} else {
	    ptMap[oldIdx]=-1;
	}
    }
    ts->points.resize(newIdx);

    for (newIdx=oldIdx=0; oldIdx<ts->elements.size(); oldIdx++) {
	int v1 = ts->elements[oldIdx]->i1;
	int v2 = ts->elements[oldIdx]->i2;
	int v3 = ts->elements[oldIdx]->i3;
	if (ptRemoved[v1]) {
	    ts->elements[oldIdx]->i1 = v1 = ptNewIdx[v1];
	}
	if (ptRemoved[v2]) {
	    ts->elements[oldIdx]->i2 = v2 = ptNewIdx[v2];
	}
	if (ptRemoved[v3]) {
	    ts->elements[oldIdx]->i3 = v3 = ptNewIdx[v3];
	}
	if (v1 == v2 || v2 == v3 || v1 == v3) {
//	    cerr << "element:"<<oldIdx<<" ("<<v1<<","<<v2<<","<<v3<<")  ";
	} else {
	    ts->elements[newIdx]->i1 = ptMap[v1];
	    ts->elements[newIdx]->i2 = ptMap[v2];
	    ts->elements[newIdx]->i3 = ptMap[v3];
	    if (ptMap[v1] == -1 || ptMap[v2] == -1 || ptMap[v3] == -1) {
//		cerr << "ERROR!!!!!  Screwed up somehow...\n";
	    }
	    newIdx++;
	}
    }
    ts->elements.resize(newIdx);

#if 0
    cerr << "checking all remaining triangles...";
    cerr << "# elements = "<<ts->elements.size()<<"   # pts = " << ts->points.size()<<"\n";
    for (i=0; i<ts->elements.size(); i++) {
	if (ts->elements[i]->i1 < 0 || (ts->elements[i]->i1 >= ts->points.size()))
	    cerr << "bad 1st pt. in element "<<i<<": "<<ts->elements[i]->i1<<"\n";
	if (ts->elements[i]->i2 < 0 || (ts->elements[i]->i2 >= ts->points.size()))
	    cerr << "bad 2nd pt. in element "<<i<<": "<<ts->elements[i]->i2<<"\n";
	if (ts->elements[i]->i3 < 0 || (ts->elements[i]->i3 >= ts->points.size()))
	    cerr << "bad 3rd pt. in element "<<i<<": "<<ts->elements[i]->i3<<"\n";

    }
#endif

}
