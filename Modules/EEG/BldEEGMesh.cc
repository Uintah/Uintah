
/*
 *  BldEEGMesh: Generate points, send points in head to be meshed,
 *		remove large gray-matter and all air elements --
 *		storing cortical surface nodes.  Identify nearest
 *		scalp nodes to electrodes.  Store Mesh and BC's
 *		(source node numbers and solution node numbers) to
 *		disk.

 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Classlib/Queue.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColumnMatrix.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/SparseRowMatrix.h>
#include <Datatypes/SegFld.h>
#include <Datatypes/SegFldPort.h>
#include <Datatypes/SurfTree.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/TriSurface.h>
#include <Geometry/BBox.h>
#include <Geometry/Point.h>
#include <Classlib/Array1.h>
#include <Classlib/Array2.h>
#include <Malloc/Allocator.h>
#include <Math/MusilRNG.h>
#include <TCL/TCLvar.h>

#include <iostream.h>
#include <stdio.h>

/*
 * The values from Peters and DeMunck's 1991 papers:
 *        skin:           1.000
 *        skull:          0.050
 *        csf:            4.620
 *        gray matter:    1.000
 *        white matter:   0.430
 */

#define AIR_CONDUCTIVITY 0.0
#define SKIN_CONDUCTIVITY 1.0
#define BONE_CONDUCTIVITY 0.05
#define CSF_CONDUCTIVITY 4.620
#define GREY_CONDUCTIVITY 1.0
#define WHITE_CONDUCTIVITY 0.43

class BldEEGMesh : public Module {
    SegFldIPort* iseg;
    SurfaceIPort* istree;
    SurfaceIPort* itrisurf;
    MeshOPort * omesh;
public:
    BldEEGMesh(const clString& id);
    BldEEGMesh(const BldEEGMesh&, int deep);
    virtual ~BldEEGMesh();
    virtual Module* clone(int deep);
    virtual void execute();

    void tess(const MeshHandle& mesh);
    void randomPointsInTetra(Array1<Point> &pts, const Point &v0, 
			     const Point &v1,const Point &v2,const Point &v3);
    void genPts(SegFldHandle sf, SurfTree *st, int num, MeshHandle mesh);
    void classifyElements(SegFldHandle sf, Mesh *m, SurfTree* st, int greyMatlIdx,
			  int whiteMatlIdx);
    void removeAirAndGreyMatlElems(Mesh *m, Array1<int>& bcCortex);
    void applyScalpBCs(Mesh *m, TriSurface *ts, Array1<int>& bcScalp,
		       const Array1<int>& bcCortex);
    int findLargestGreyMatterIdx(SegFldHandle sf);
    int findLargestWhiteMatterIdx(SegFldHandle sf);
    void reorderPts(Mesh *m, const Array1<int>& bcScalp, 
		    const Array1<int>& bcCortex);
    MusilRNG mr;
};

extern "C" {
Module* make_BldEEGMesh(const clString& id)
{
    return new BldEEGMesh(id);
}
};

BldEEGMesh::BldEEGMesh(const clString& id)
: Module("BldEEGMesh", id, Filter)
{
    iseg=new SegFldIPort(this, "SegIn", SegFldIPort::Atomic);
    add_iport(iseg);
    istree=new SurfaceIPort(this, "SurfTreeIn", SurfaceIPort::Atomic);
    add_iport(istree);
    itrisurf=new SurfaceIPort(this, "TriSurfIn", SurfaceIPort::Atomic);
    add_iport(itrisurf);
    // Create the output port
    omesh=new MeshOPort(this, "MeshOut", MeshIPort::Atomic);
    add_oport(omesh);
}

BldEEGMesh::BldEEGMesh(const BldEEGMesh& copy, int deep)
: Module(copy, deep)
{
}

BldEEGMesh::~BldEEGMesh()
{
}

Module* BldEEGMesh::clone(int deep)
{
    return new BldEEGMesh(*this, deep);
}

void BldEEGMesh::tess(const MeshHandle& mesh)
{
    BBox bbox;
    int nn=mesh->nodes.size();
    Array1<int> swap(nn);
    
    for(int ii=0;ii<nn;ii++){
	swap[ii]=ii;
	Point p(mesh->nodes[ii]->p);
	bbox.extend(p);
    }

    for (ii=0; ii<nn; ii++) {
	int swapIdx=mr()*nn;
	int tmp=swap[ii];
	swap[ii]=swap[swapIdx];
	swap[swapIdx]=tmp;
    }
	
    double epsilon=.1*bbox.longest_edge();

    // Extend by max-(eps, eps, eps) and min+(eps, eps, eps) to
    // avoid thin/degenerate bounds
    Point max(bbox.max()+Vector(epsilon, epsilon, epsilon));
    Point min(bbox.min()-Vector(epsilon, epsilon, epsilon));

    mesh->nodes.add(new Node(Point(min.x(), min.y(), min.z())));
    mesh->nodes.add(new Node(Point(max.x(), min.y(), min.z())));
    mesh->nodes.add(new Node(Point(max.x(), min.y(), max.z())));
    mesh->nodes.add(new Node(Point(min.x(), min.y(), max.z())));
    mesh->nodes.add(new Node(Point(min.x(), max.y(), min.z())));
    mesh->nodes.add(new Node(Point(max.x(), max.y(), min.z())));
    mesh->nodes.add(new Node(Point(max.x(), max.y(), max.z())));
    mesh->nodes.add(new Node(Point(min.x(), max.y(), max.z())));

    Element* el1=new Element(mesh.get_rep(), nn+0, nn+1, nn+4, nn+3);
    Element* el2=new Element(mesh.get_rep(), nn+2, nn+1, nn+3, nn+6);
    Element* el3=new Element(mesh.get_rep(), nn+7, nn+3, nn+6, nn+4);
    Element* el4=new Element(mesh.get_rep(), nn+5, nn+6, nn+4, nn+1);
    Element* el5=new Element(mesh.get_rep(), nn+1, nn+3, nn+4, nn+6);
    el1->faces[0]=4; el1->faces[1]=-1; el1->faces[2]=-1; el1->faces[3]=-1;
    el2->faces[0]=4; el2->faces[1]=-1; el2->faces[2]=-1; el2->faces[3]=-1;
    el3->faces[0]=4; el3->faces[1]=-1; el3->faces[2]=-1; el3->faces[3]=-1;
    el4->faces[0]=4; el4->faces[1]=-1; el4->faces[2]=-1; el4->faces[3]=-1;
    el5->faces[0]=2; el5->faces[1]=3; el5->faces[2]=1; el5->faces[3]=0;
    el1->orient();
    el2->orient();
    el3->orient();
    el4->orient();
    el5->orient();
    mesh->elems.add(el1);
    mesh->elems.add(el2);
    mesh->elems.add(el3);
    mesh->elems.add(el4);
    mesh->elems.add(el5);

    for(int node=0;node<nn;node++){
//	cerr << "Adding node " << node << " " << mesh->nodes[node]->p << endl;
	if(!mesh->insert_delaunay(swap[node], 0)){
	    cerr << "Mesher failed!\n";
	    exit(-1);
	}
	if((node+1)%500 == 0){
	    mesh->pack_elems();
	    cerr << node+1 << " nodes meshed (" << mesh->elems.size() << " elements)\r";
	}
    }
    cerr << endl;
    cerr << "Performing cleanup...\n";
    mesh->compute_neighbors();
    mesh->remove_delaunay(nn, 0);
    mesh->remove_delaunay(nn+1, 0);
    mesh->remove_delaunay(nn+2, 0);
    mesh->remove_delaunay(nn+3, 0);
    mesh->remove_delaunay(nn+4, 0);
    mesh->remove_delaunay(nn+5, 0);
    mesh->remove_delaunay(nn+6, 0);
    mesh->remove_delaunay(nn+7, 0);
    mesh->pack_all();
    double vol=0;
    cerr << "There are " << mesh->elems.size() << " elements" << endl;
    for(int i=0;i<mesh->elems.size();i++){
	double dv=mesh->elems[i]->volume();
	if (dv<0.00000001) cerr << "Warning: elements with tiny volume!\n";
	else vol+=dv;
    }
    cerr << "Total volume: " << vol << endl;
}





// Generate pts.size() random points in a tetra spanned by v0, v1, v2, v3.
// Thanks to Peter-Pike, Dean, and Peter Shirley for the algorithm.
void BldEEGMesh::randomPointsInTetra(Array1<Point> &pts, const Point &v0, 
				     const Point &v1, const Point &v2, 
				     const Point &v3) {
    double s, t, u;
    double alpha, beta, gamma;
    double p0, p1, p2, p3;

    for (int i=0; i<pts.size(); i++) {
	s=mr();
	t=mr();
	u=mr();
	alpha=1-pow(s, 0.33333333);
	beta=1-sqrt(t);
	gamma=u;
	p0=1-alpha;
	p1=beta*alpha*(1-gamma);
	p2=alpha*(1-beta);
	p3=beta*alpha*gamma;
	pts[i]=AffineCombination(v0, p0, v1, p1, v2, p2, v3, p3);
    }
}


// Generate num points randomly distributed through the SegFld.  Don't
// put any points in air regions (type == 0).
void BldEEGMesh::genPts(SegFldHandle sf, SurfTree *st, int num, 
			MeshHandle mesh) {
    mesh->nodes.resize(0);
    sf->audit();
    sf->printComponents();
    Array1<Point> pts;
    BBox bb;
    for (int iii=0; iii<st->points.size(); iii++)
	bb.extend(st->points[iii]);
    Point min, max;
    min = bb.min();
    max = bb.max();
    Vector dv=max-min;
    min = min-dv*.001;
    max = max+dv*.001;
//    sf->get_bounds(min, max);
    Vector v(max-min);
    double vol=v.x()*v.y()*v.z();
    vol/=(3*num);
    double side=pow(vol, 0.3333333);
    Point curr;
//    int pidx=0;

    int ci, cj, ck;
    int inum=v.x()/side;
    double dx=v.x()/inum;
    int jnum=v.y()/side;
    double dy=v.y()/jnum;
    int knum=v.z()/side;
    double dz=v.z()/knum;

    cerr <<" min="<<min<<" max="<<max<<"  dx="<<dx<<" dy="<<dy<<" dz="<<dz<<"\n";
    cerr << "minx="<<min.x()+dx/2.<<"  maxx="<<min.x()+dx/2.+dx*inum<<"\n";
    cerr << "miny="<<min.y()+dy/2.<<"  maxy="<<min.y()+dy/2.+dy*jnum<<"\n";
    cerr << "minz="<<min.z()+dz/2.<<"  maxz="<<min.z()+dz/2.+dz*knum<<"\n";

//    cerr << "sf dims: "<<sf->grid.dim1()<<", "<<sf->grid.dim2()<<", "<<sf->grid.dim3()<<"\n";
    Array3<char> visited(inum,jnum,knum);
    visited.initialize(0);

    for (int ii=0; ii<st->points.size(); ii++) {
	int i=(st->points[ii].x()-min.x())/dx;
	int j=(st->points[ii].y()-min.y())/dy;
	int k=(st->points[ii].z()-min.z())/dz;
	if (!visited(i,j,k)) {
	    mesh->nodes.add(NodeHandle(new Node(st->points[ii])));
	    visited(i,j,k)=1;
	}
    }
    curr.x(min.x()+dx/2.);
    for (int i=0; i<inum; i++, curr.x(curr.x()+dx)) {
	curr.y(min.y()+dy/2.);
	for (int j=0; j<jnum; j++, curr.y(curr.y()+dy)) {
	    curr.z(min.z()+dz/2.);
	    for (int k=0; k<knum; k++, curr.z(curr.z()+dz)) {
		if (!visited(i,j,k)) {
		    Point p=curr+Vector((mr()-.5)*dx*.8, 
					(mr()-.5)*dy*.8, 
					(mr()-.5)*dz*.8);
		    sf->locate(p, ci, cj, ck);
//		cerr << ci<<" "<<cj<<" "<<ck<<"\n";
		    if (sf->get_type(sf->comps[sf->grid(ci, cj, ck)]) == 0) 
			continue;
//		cerr << "Found a non-air point!\n";
		    mesh->nodes.add(NodeHandle(new Node(p)));
//		pts[pidx]=p;
//		pidx++;
		}
	    }
	}
    }
//    cerr << "pidx="<<pidx<<"\n";
//    pts.resize(pidx);
//    char fpts[100];
//    sprintf(fpts, "%s.pts", oname);
//    cerr <<"Generated "<<pidx<<" points -- writing them to file: "<<fpts<<"\n";
//    FILE *Fpts=fopen(fpts, "wt");
//    fprintf(Fpts, "%d\n", pts.size());
//    for (i=0; i<pts.size(); i++) {
//	fprintf(Fpts, "%lf  %lf  %lf\n", pts[i]);
//    }
//    fclose(Fpts);
}


// For each element, sample it at npoint, see what material type is
// there -- for the greyMatlIdx component (and those contained inside
// of it), tag it as material 6.  Find out which is most popular and
// label the element that type.
void BldEEGMesh::classifyElements(SegFldHandle sf, Mesh *m, SurfTree* st, 
				  int greyMatlIdx, int whiteMatlIdx) {
    Array1<int> popularity(7);
    Array1<Point> samples(10);
    Point v0, v1, v2, v3;
    int ci, cj, ck;
    cerr << "These are the components(materials) interior to "<<greyMatlIdx<<"(4): ";
    for (int i=0; i<st->inner[greyMatlIdx].size(); i++) {
	int inComp=st->inner[greyMatlIdx][i];
	int type=sf->get_type(sf->comps[inComp]);
	cerr <<inComp<<"("<<type<<") ";
    }
    cerr << "\n";
    cerr << "These are the components(materials) interior to "<<whiteMatlIdx<<"(5): ";
    for (i=0; i<st->inner[whiteMatlIdx].size(); i++) {
	int inComp=st->inner[whiteMatlIdx][i];
	int type=sf->get_type(sf->comps[inComp]);
	cerr <<inComp<<"("<<type<<") ";
    }
    cerr << "\n";
//    cerr << "m->elems.size()="<<m->elems.size()<<"\n";
    for (i=0; i<m->elems.size(); i++) {
//	cerr << "..."<<i <<"\n";
	popularity.initialize(0);
	Element *e=m->elems[i];
	int tied=1;
	int mostPopIdx=0;
	int n0, n1, n2, n3;
	n0=e->n[0]; n1=e->n[1]; n2=e->n[2]; n3=e->n[3];
	while (tied) {
//	    cerr << "tied... ";
	    randomPointsInTetra(samples, m->nodes[n0]->p, m->nodes[n1]->p,
				m->nodes[n2]->p, m->nodes[n3]->p);
	    for (int j=0; j<samples.size(); j++) {
		sf->locate(samples[j], ci, cj, ck);
		int comp=sf->grid(ci,cj,ck);
		int type=sf->get_type(sf->comps[comp]);
		if (comp == greyMatlIdx || comp == whiteMatlIdx) {
//		    cerr << "another of type 6, ";
		    popularity[6] = popularity[6]+1;
		} else {	
		    for (int k=0; k<st->inner[greyMatlIdx].size(); k++) {
			if (st->inner[greyMatlIdx][k] == comp) {
			    break;
			}
		    }
		    if (k != st->inner[greyMatlIdx].size()) {
			popularity[6]=popularity[6]+1;
//			cerr << "grey type 6, ";
		    } else {
			for (k=0; k<st->inner[whiteMatlIdx].size(); k++) {
			    if (st->inner[whiteMatlIdx][k] == comp) {
				break;
			    }
			}
			if (k != st->inner[whiteMatlIdx].size()) {
			    popularity[6]=popularity[6]+1;
			} else {
			    popularity[type]=popularity[type]+1;
			}
		    }
		}
	    }
	    tied=0;
	    int mostPopAmt=popularity[0];
	    for (j=1; j<popularity.size(); j++) {
		if (mostPopAmt > popularity[j]) continue;
		if (mostPopAmt == popularity[j]) { tied=1; continue; }
		mostPopAmt = popularity[j];
		mostPopIdx = j;
		tied=0;
	    }
//	    cerr << "mostPopAmt="<<mostPopAmt<<" Idx="<<mostPopIdx<<" ";
	}
	sf->locate(AffineCombination(m->nodes[n0]->p, .25, 
				     m->nodes[n1]->p, .25,
				     m->nodes[n2]->p, .25,
				     m->nodes[n3]->p, .25), ci, cj, ck);
//	int comp=sf->grid(ci,cj,ck);
//	int type=sf->get_type(sf->comps[comp]);
//	cerr << "centroid type is "<<type<<" mostPopIdx is "<<mostPopIdx<<"\n";
	e->cond=mostPopIdx;
    }
}	

// For all elements of type 6, set them to null, and store their
// nodes in the "grey nodes" list.  for all elements of type "air",
// just remove them.  for all other elements, add their nodes to the
// interior nodes list.
// Then find any nodes in both lists and save those indices to the BC 
// cortex list.
// Go through all of the nodes, and build a mapping which won't include
// unattached nodes.
// For all elements, renumber according to the mapping.  Do the same with
// the BC cortex list.
void BldEEGMesh::removeAirAndGreyMatlElems(Mesh *m, Array1<int>& bcCortex) {
    Array1<int> greyNodes(m->nodes.size());
    greyNodes.initialize(0);
    Array1<int> interiorNodes(m->nodes.size());
    interiorNodes.initialize(0);
    for (int i=0; i<m->elems.size(); i++) {
	Element *e=m->elems[i];
	if (e->cond == 6) {
	    greyNodes[e->n[0]]=1;
	    greyNodes[e->n[1]]=1;
	    greyNodes[e->n[2]]=1;
	    greyNodes[e->n[3]]=1;
	    m->elems[i]=0;
	} else if (e->cond == 0) { 
	    m->elems[i]=0;
	} else {
	    interiorNodes[e->n[0]]=1;
	    interiorNodes[e->n[1]]=1;
	    interiorNodes[e->n[2]]=1;
	    interiorNodes[e->n[3]]=1;
	}
    }
    Array1<int> map(m->nodes.size());
    int cnt=0;
    for (i=0; i<m->nodes.size(); i++) {
	if (greyNodes[i] && interiorNodes[i]) bcCortex.add(cnt);
	if (interiorNodes[i]) {
	    m->nodes[cnt]->p = m->nodes[i]->p;
	    map[i]=cnt;
	    cnt++;
	} else {
	    map[i]=-32;
	}
    }
    m->nodes.resize(cnt);
    cnt=0;
    for (i=0; i<m->elems.size(); i++) {
	Element *e=m->elems[i];
	if (e) {
	    e->n[0]=map[e->n[0]];
	    e->n[1]=map[e->n[1]];
	    e->n[2]=map[e->n[2]];
	    e->n[3]=map[e->n[3]];	    
	    m->elems[cnt]=m->elems[i];
	    cnt++;
	}
    }
    m->elems.resize(cnt);
}

// Find the nearest node from the mesh to each of the electrodes.
// For each, give that node a Dirichlet boundary condition, and save it
// to the bcScalp array as well.
void BldEEGMesh::applyScalpBCs(Mesh *m, TriSurface *ts, Array1<int>& bcScalp,
			       const Array1<int>& bcCortex) {
//    cerr << "locating "<<bcScalp.size()<<" electrode sites...\n";
    Array1<int> used(m->nodes.size());
    used.initialize(0);
    for (int i=0; i<bcCortex.size(); i++) used[bcCortex[i]]=1;
    for (i=0; i<ts->bcIdx.size(); i++) {
	int invalid=1;
	double dist;
	int closest;
	double bcVal;
	Point bcPt(ts->points[ts->bcIdx[i]]);
	bcVal=ts->bcVal[i];
	for (int j=0; j<m->nodes.size(); j++) {
	    if (!used[j]) {
		double d=(m->nodes[j]->p - bcPt).length2();
		if (invalid || d<dist) {
		    invalid=0;
		    dist=d;
		    closest=j;
		}
	    }
	}
	used[closest]=1;
	bcScalp.add(closest);
	m->nodes[closest]->bc = new DirichletBC(SurfaceHandle(ts), bcVal);
    }
}

int BldEEGMesh::findLargestGreyMatterIdx(SegFldHandle sf) {
    int invalid=1;
    int largestIdx;
    int largestAmt;
    for (int i=0; i<sf->comps.size(); i++) {
	int component=sf->comps[i];
	if (sf->get_type(component) == 4) {
	    int amt=sf->get_size(component);
	    if (invalid || (largestAmt<amt)) {
		invalid=0;
		largestAmt=amt;
		largestIdx=i;
	    }
	}
    }
    return largestIdx;
}
		
int BldEEGMesh::findLargestWhiteMatterIdx(SegFldHandle sf) {
    int invalid=1;
    int largestIdx;
    int largestAmt;
    for (int i=0; i<sf->comps.size(); i++) {
	int component=sf->comps[i];
	if (sf->get_type(component) == 5) {
	    int amt=sf->get_size(component);
	    if (invalid || (largestAmt<amt)) {
		invalid=0;
		largestAmt=amt;
		largestIdx=i;
	    }
	}
    }
    return largestIdx;
}

void BldEEGMesh::reorderPts(Mesh *m, const Array1<int>& bcScalp, 
			    const Array1<int>& bcCortex) {

    for (int ii=0; ii<bcScalp.size(); ii++) {
	for (int jj=0; jj<bcCortex.size(); jj++) {
	    if (bcScalp[ii] == bcCortex[jj]) {
		cerr << "ERROR:  bcScalp["<<ii<<"] = bcCortex["<<jj<<"] = "<<bcCortex[jj]<<"\n";
	    }
	}
    }

    Array1<int> map(m->nodes.size());
    Array1<int> invMap(m->nodes.size());
    map.initialize(-1);

    for (int i=0; i<bcScalp.size(); i++) {
	map[bcScalp[i]]=i;
	invMap[i]=bcScalp[i];
    }
    int fromEnd=m->nodes.size()-bcCortex.size();
    for (i=bcCortex.size()-1; i>=0; i--) {
	map[bcCortex[i]]=fromEnd+i;
	invMap[fromEnd+i]=bcCortex[i];
    }
    int curr=bcScalp.size();
    for (i=0; i<m->nodes.size(); i++) {
	if (map[i] == -1) {
	    map[i] = curr;
	    invMap[curr]=i;
	    curr++;
	}
    }
    
    Array1<Point> p(m->nodes.size());
    for (i=0; i<m->nodes.size(); i++) p[i]=m->nodes[invMap[i]]->p;
    for (i=0; i<m->nodes.size(); i++) m->nodes[i]->p = p[i];
    for (i=0; i<m->elems.size(); i++) {
	Element *e=m->elems[i];
	for (int j=0; j<4; j++) e->n[j]=map[e->n[j]];
    }
}

void BldEEGMesh::execute()
{
    int numPts=10000;
    
    SegFldHandle sf;
    if (!iseg->get(sf))
	return;
    if (!sf.get_rep()) {
	cerr << "Error: empty seg fld\n";
	return;
    }
    
    SurfaceHandle sh;
    if (!istree->get(sh))
	return;
    if (!sh.get_rep()) {
	cerr << "Error: empty surftree\n";
	return;
    }
    SurfTree *st=sh->getSurfTree();
    if (!st) {
	cerr << "Error: surface isn't a surftree\n";
	return;
    }

    sh=0;
    if (!itrisurf->get(sh))
	return;
    if (!sh.get_rep()) {
	cerr << "Error: empty trisurf\n";
	return;
    }
    TriSurface *ts=sh->getTriSurface();
    if (!ts) {
	cerr << "Error: surface isn't a trisurface\n";
	return;
    }

    Mesh *m = new Mesh;
    m->cond_tensors.resize(6);
    m->cond_tensors[0].resize(6);
    m->cond_tensors[0].initialize(0);
    m->cond_tensors[0][0]=m->cond_tensors[0][3]=m->cond_tensors[0][5]=AIR_CONDUCTIVITY;

    m->cond_tensors[1].resize(6);
    m->cond_tensors[1].initialize(0);
    m->cond_tensors[1][0]=m->cond_tensors[1][3]=m->cond_tensors[1][5]=SKIN_CONDUCTIVITY;

    m->cond_tensors[2].resize(6);
    m->cond_tensors[2].initialize(0);
    m->cond_tensors[2][0]=m->cond_tensors[2][3]=m->cond_tensors[2][5]=BONE_CONDUCTIVITY;

    m->cond_tensors[3].resize(6);
    m->cond_tensors[3].initialize(0);
    m->cond_tensors[3][0]=m->cond_tensors[3][3]=m->cond_tensors[3][5]=CSF_CONDUCTIVITY;

    m->cond_tensors[4].resize(6);
    m->cond_tensors[4].initialize(0);
    m->cond_tensors[4][0]=m->cond_tensors[4][3]=m->cond_tensors[4][5]=GREY_CONDUCTIVITY;
    
    m->cond_tensors[5].resize(6);
    m->cond_tensors[5].initialize(0);
    m->cond_tensors[5][0]=m->cond_tensors[5][3]=m->cond_tensors[5][5]=WHITE_CONDUCTIVITY;
    
    MeshHandle mesh(m);
    genPts(sf, st, numPts, mesh);
    cerr << "Tesselating the points (thanks Steve!)...\n";
    tess(mesh);

    int greyMatlIdx=findLargestGreyMatterIdx(sf);
    int whiteMatlIdx=findLargestWhiteMatterIdx(sf);
    cerr << "The largest grey matter component is: "<<greyMatlIdx<<"\n";
    cerr << "The largest white matter component is: "<<whiteMatlIdx<<"\n";
    cerr << "Calling classify elements...\n";
    classifyElements(sf, m, st, greyMatlIdx, whiteMatlIdx);
    Array1<int> bcCortex;
//    cerr << "Calling removeAirAndGreyMatlElems...\n";
    removeAirAndGreyMatlElems(m, bcCortex);
    // might want to output here first... ??
    Array1<int> bcScalp;
    cerr << "Calling applyScalpBCs...\n";
    applyScalpBCs(m, ts, bcScalp, bcCortex);

    FILE *f=fopen("/tmp/scalpPts", "wt");
    fprintf(f, "%d\n", bcScalp.size());
    for (int i=0; i<bcScalp.size(); i++) {
	Point p(m->nodes[bcScalp[i]]->p);
	fprintf(f, "%lf %lf %lf\n", p.x(), p.y(), p.z());
    }
    fclose(f);
    
    f=fopen("/tmp/bc", "wt");
    for (i=0; i<bcScalp.size(); i++) {
	double v=m->nodes[bcScalp[i]]->bc->value;
	fprintf(f, "%lf\n", v);
    }
    fclose(f);

    cerr << "\n\n\n***** Just wrote output file for inverse solver: "<<name<<"\n";
    cerr <<       "      after running SCIRun to build the .matrix file, call:\n";
    cerr <<       "      'MatrixToMat probName "<<bcScalp.size()<<" "<<m->nodes.size()-bcScalp.size()-bcCortex.size()<<" "<<bcCortex.size()<<"'\n";
    cerr << 	  "      and then run 'solve probName'\n";

    f=fopen("/tmp/cortexPts", "wt");
    fprintf(f, "%d\n", bcCortex.size());
    for (i=0; i<bcCortex.size(); i++) {
	Point p(m->nodes[bcCortex[i]]->p);
	fprintf(f, "%lf %lf %lf\n", p.x(), p.y(), p.z());
    }
    fclose(f);

    reorderPts(m, bcScalp, bcCortex);
    
    // blow away the Dirich boundary conditions for building the inverse matrix
    for (i=0; i<m->nodes.size(); i++) 
	if (m->nodes[i]->bc != 0) m->nodes[i]->bc=0;

    MeshHandle mh(m);
    omesh->send(mh);
    
    cerr << "Mesh has been built and output -- "<<mesh->nodes.size()<<" nodes, "<<mesh->elems.size()<<" elements.\n";
}
