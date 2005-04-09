
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
 *   August 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Tester/RigorousTest.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Classlib/Queue.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColumnMatrix.h>
#include <Datatypes/ColumnMatrixPort.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/SegFld.h>
#include <Datatypes/SegFldPort.h>
#include <Datatypes/SurfTree.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/TriSurface.h>
#include <Geometry/BBox.h>
#include <Geometry/Point.h>
#include <Classlib/Array1.h>
#include <Classlib/Array2.h>
#include <Classlib/Array3.h>
#include <Malloc/Allocator.h>
#include <Math/MusilRNG.h>
#include <TCL/TCLvar.h>

#include <iostream.h>
#include <stdio.h>

using sci::Mesh;
using sci::NodeHandle;
using sci::Node;
using sci::Element;
using sci::DirichletBC;

int airCtr=0;
int othrCtr=0;
void dump_mesh(Mesh *m);

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
    MeshOPort * omesh;
    SurfaceOPort* ostree;
    TCLint npts;
    int AddBC;
    TCLint ADDBC;
    int NoBrain;
    TCLint NOBRAIN;
public:
    BldEEGMesh(const clString& id);
    BldEEGMesh(const BldEEGMesh&, int deep);
    virtual ~BldEEGMesh();
    virtual Module* clone(int deep);
    virtual void execute();

    void randomPointsInTetra(Array1<Point> &pts, const Point &v0, 
			     const Point &v1,const Point &v2,const Point &v3);
    void genPtsAndTets(SegFldHandle sf, SurfTree *st, int num, 
		       Mesh *mesh);
    void classifyElements(SegFldHandle sf, Mesh *m, SurfTree* st, 
			  int greyMatlIdx, int whiteMatlIdx);
    void removeAirAndGreyMatlElems(Mesh *m, Array1<int>& cortexBCMeshNodes);
    void applyScalpBCs(Mesh *m, SurfTree *st, Array1<int>& scalpBCMeshNodes,
 		        const Array1<int>& cortexBCMeshNodes);
    void findCortexNodesInSTree(Mesh *m, SurfTree* st, int startCNode,
					const Array1<int>& cortexBCMeshNodes);
    int findLargestGreyMatterIdx(SegFldHandle sf);
    int findLargestWhiteMatterIdx(SegFldHandle sf);
    void reorderPts(Mesh *m, const Array1<int>& scalpBCMeshNodes, 
		    const Array1<int>& cortexBCMeshNodes);
    void setConductivities(Mesh *m);
    MusilRNG mr;
};

extern "C" {
Module* make_BldEEGMesh(const clString& id)
{
    return new BldEEGMesh(id);
}
}

BldEEGMesh::BldEEGMesh(const clString& id)
: Module("BldEEGMesh", id, Filter), npts("npts", id, this),
  ADDBC("ADDBC", id, this), NOBRAIN("NOBRAIN", id, this)
{
    iseg=new SegFldIPort(this, "SegIn", SegFldIPort::Atomic);
    add_iport(iseg);
    istree=new SurfaceIPort(this, "SurfTreeIn", SurfaceIPort::Atomic);
    add_iport(istree);
    omesh=new MeshOPort(this, "MeshOut", MeshIPort::Atomic);
    add_oport(omesh);
    ostree=new SurfaceOPort(this, "SurfTreeOut", SurfaceIPort::Atomic);
    add_oport(ostree);
}

BldEEGMesh::BldEEGMesh(const BldEEGMesh& copy, int deep)
: Module(copy, deep), npts("npts", id, this),
  ADDBC("ADDBC", id, this), NOBRAIN("NOBRAIN", id, this)
{
}

BldEEGMesh::~BldEEGMesh()
{
}

Module* BldEEGMesh::clone(int deep)
{
    return new BldEEGMesh(*this, deep);
}

// Generate pts.size() random points in a tetra spanned by v0, v1, v2, v3.
// Thanks to Peter-Pike, Dean, and Peter Shirley for the algorithm.
void BldEEGMesh::randomPointsInTetra(Array1<Point> &pts, const Point &v0, 
				     const Point &v1, const Point &v2, 
				     const Point &v3) {
    double alpha, beta, gamma;

    for (int i=0; i<pts.size(); i++) {
	alpha = pow(mr(),1.0/3.0);
	beta = sqrt(mr());
	gamma = mr();

	// let the compiler do the sub-expression stuff...

	pts[i]=AffineCombination(v0,1-alpha,
				 v1,beta*alpha*(1-gamma),
				 v2,alpha*(1-beta),
				 v3,beta*alpha*gamma);
	pts[i]=AffineCombination(v0,.25,
				 v1,.25,
				 v2,.25,
				 v3,.25);
    }
}

// Generate num points randomly distributed through the SegFld.  Don't
// put any points in air regions (type == 0).
// Teselate these points by creating five tets from each cube.
void BldEEGMesh::genPtsAndTets(SegFldHandle sf, SurfTree *st, int num, 
			       Mesh* mesh) {
    BBox bb;
    for (int iii=0; iii<st->nodes.size(); iii++)
	bb.extend(st->nodes[iii]);
    Point min, max;
    min = bb.min();
    max = bb.max();
    int minaa, minbb, mincc, maxaa, maxbb, maxcc;
    sf->locate(min, minaa, minbb, mincc);
    sf->locate(max, maxaa, maxbb, maxcc); maxaa++; maxbb++; maxcc++;
    if (minaa<0) {cerr << "Warning minaa was: "<<minaa<<"\n"; minaa=0;}
    if (minbb<0) {cerr << "Warning minbb was: "<<minbb<<"\n"; minbb=0;}
    if (mincc<0) {cerr << "Warning mincc was: "<<mincc<<"\n"; mincc=0;}
    if (maxaa>sf->nx) {cerr << "Warning maxaa was: "<<maxaa<<"\n"; maxaa=sf->nx;}
    if (maxbb>sf->ny) {cerr << "Warning maxbb was: "<<maxbb<<"\n"; maxbb=sf->ny;}
    if (maxcc>sf->nz) {cerr << "Warning maxcc was: "<<maxcc<<"\n"; maxcc=sf->nz;}


    double num_in=0;

    for (int aa=minaa; aa<=maxaa; aa++)
        for (int bb=minbb; bb<=maxbb; bb++)
            for (int cc=mincc; cc<=maxcc; cc++) {
		int type=sf->get_type(sf->comps[sf->grid(aa, bb, cc)]);
		if (type != 0 && type != 4 && type != 5) num_in++;
	    }
    double density = num_in/((maxaa-minaa+1)*(maxbb-minbb+1)*(maxcc-mincc+1));
    mesh->nodes.resize(0);
    sf->audit();
    sf->printComponents();
    Array1<Point> pts;
    Vector v(max-min);
    min = min+v*.001;
    max = max-v*.001;
    v*=1.002;

    double vol=v.x()*v.y()*v.z();
    double numNodesRequired=num/density;
    double numNodesPerSide=pow(numNodesRequired, 1./3.);
    double numCellsPerSide=numNodesPerSide-1;
    double numCells=numCellsPerSide*numCellsPerSide*numCellsPerSide;
    double cellVolume=vol/numCells;
    double lengthPerCellSide=pow(cellVolume, 1./3.);

#if 0
    int inum=15;
    int jnum=15;
    int knum=15;
#endif

    int inum=v.x()/lengthPerCellSide+1;
    int jnum=v.y()/lengthPerCellSide+1;
    int knum=v.z()/lengthPerCellSide+1;

    double dx=v.x()/(inum-1);
    double dy=v.y()/(jnum-1);
    double dz=v.z()/(knum-1);

    cerr << "minx="<<min.x()<<"  maxx="<<min.x()+dx*(inum-1)<<"\n";
    cerr << "miny="<<min.y()<<"  maxy="<<min.y()+dy*(jnum-1)<<"\n";
    cerr << "minz="<<min.z()<<"  maxz="<<min.z()+dz*(knum-1)<<"\n";
    cerr << "inum="<<inum<<"  jnum="<<jnum<<"  knum="<<knum<<"  dx="<<dx<<"  dy="<<dy<<"  dz="<<dz<<"\n";

    Point curr(min);
    int currIdx=0;
    Array3<int> nodes(inum, jnum, knum);

    for (int i=0; i<inum; i++, curr.x(curr.x()+dx)) {
	curr.y(min.y());
	for (int j=0; j<jnum; j++, curr.y(curr.y()+dy)) {
	    curr.z(min.z());
	    for (int k=0; k<knum; k++, curr.z(curr.z()+dz)) {
		nodes(i,j,k)=currIdx++;
		mesh->nodes.add(NodeHandle(new Node(curr)));
	    }
	}
    }

    Array1<Element *> e(5);
    Array1<int> c(8);
    for (i=0; i<inum-1; i++) {
	for (int j=0; j<jnum-1; j++) {
	    for (int k=0; k<knum-1; k++) {
		c[0]=nodes(i,j,k);
		c[1]=nodes(i+1,j,k);
		c[2]=nodes(i+1,j+1,k);
		c[3]=nodes(i,j+1,k);
		c[4]=nodes(i,j,k+1);
		c[5]=nodes(i+1,j,k+1);
		c[6]=nodes(i+1,j+1,k+1);
		c[7]=nodes(i,j+1,k+1);
		if ((i+j+k)%2) {
		    e[0]=new Element(mesh, c[0], c[1], c[2], c[5]);
		    e[1]=new Element(mesh, c[0], c[2], c[3], c[7]);
		    e[2]=new Element(mesh, c[0], c[2], c[5], c[7]);
		    e[3]=new Element(mesh, c[0], c[4], c[5], c[7]);
		    e[4]=new Element(mesh, c[2], c[5], c[6], c[7]);
		} else {
		    e[0]=new Element(mesh, c[1], c[0], c[3], c[4]);
		    e[1]=new Element(mesh, c[1], c[3], c[2], c[6]);
		    e[2]=new Element(mesh, c[1], c[3], c[4], c[6]);
		    e[3]=new Element(mesh, c[1], c[5], c[4], c[6]);
		    e[4]=new Element(mesh, c[3], c[4], c[7], c[6]);
		}
		mesh->elems.add(e[0]); 
		mesh->elems.add(e[1]); 
		mesh->elems.add(e[2]); 
		mesh->elems.add(e[3]); 
		mesh->elems.add(e[4]); 
	    }
	}
    }
}

// For each element, sample it at npoint, see what material type is
// there -- for the greyMatlIdx component (and those contained inside
// of it), tag it as material 6.  Find out which is most popular and
// label the element that type.
void BldEEGMesh::classifyElements(SegFldHandle sf, Mesh *m, SurfTree* st, 
				  int greyMatlIdx, int whiteMatlIdx) {

    Array1<int> scalp(m->nodes.size());
    scalp.initialize(0);

    Array1<int> popularity(7);
    Array1<Point> samples(6);
//    Point v0, v1, v2, v3;
    int ci, cj, ck;
    cerr << "These are the components(materials) interior to "<<greyMatlIdx<<"(4): ";
    for (int i=0; i<st->surfI[greyMatlIdx].inner.size(); i++) {
	int inComp=st->surfI[greyMatlIdx].inner[i];
	int type=sf->get_type(sf->comps[inComp]);
	cerr <<inComp<<"("<<type<<") ";
    }
    cerr << "\n";
    cerr << "These are the components(materials) interior to "<<whiteMatlIdx<<"(5): ";
    for (i=0; i<st->surfI[whiteMatlIdx].inner.size(); i++) {
	int inComp=st->surfI[whiteMatlIdx].inner[i];
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

		if (NoBrain) {
		    if (comp == greyMatlIdx || comp == whiteMatlIdx) {
//		        cerr << "another of type 6, ";
			popularity[6] = popularity[6]+1;
		    } else {	
			for (int k=0; k<st->surfI[greyMatlIdx].inner.size(); 
			     k++) {
			    if (st->surfI[greyMatlIdx].inner[k] == comp) {
				break;
			    }
			}
			if (k != st->surfI[greyMatlIdx].inner.size()) {
			    popularity[6]=popularity[6]+1;
//			    cerr << "grey type 6, ";
			} else {
			    for (k=0; k<st->surfI[whiteMatlIdx].inner.size(); 
				 k++) {
				if (st->surfI[whiteMatlIdx].inner[k] == comp) {
				    break;
				}
			    }
			    if (k != st->surfI[whiteMatlIdx].inner.size()) {
				popularity[6]=popularity[6]+1;
			    } else {
				popularity[type]=popularity[type]+1;
			    }
			}
		    }
		} else {
		    popularity[type]=popularity[type]+1;
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

	if (sf->grid(ci,cj,ck) == 0) {airCtr++; m->elems[i]=0; scalp[n0]=1; scalp[n1]=1; scalp[n2]=1; scalp[n3]=1;} else {othrCtr++;
//	int comp=sf->grid(ci,cj,ck);
//	int type=sf->get_type(sf->comps[comp]);
//	cerr << "centroid type is "<<type<<" mostPopIdx is "<<mostPopIdx<<"\n";
	e->cond=mostPopIdx; }
    }
    m->compute_neighbors();

    Array1<int> scalp2;

    // this is just pack_nodes, but we have to keep track of the scalp array
    int nnodes=m->nodes.size();
    int idx=0;
    Array1<int> map(nnodes);
    for(i=0;i<nnodes;i++){
	NodeHandle& n=m->nodes[i];
	if(n.get_rep() && n->elems.size()){
	    map[i]=idx;
	    scalp2.add(scalp[i]);
	    m->nodes[idx++]=n;
	} else {
	    map[i]=-1234;
	}
    }
    m->nodes.resize(idx);
    
    int nelems=m->elems.size();
    for(i=0;i<nelems;i++){
	Element* e=m->elems[i];
	if(e){
	    for(int j=0;j<4;j++){
		if(map[e->n[j]]==-1234)
		    cerr << "Warning: pointing to old node: " << e->n[j] << endl;
		e->n[j]=map[e->n[j]];
	    }
	}
    }
    m->pack_elems();

#if 0
    int cnt=0;
    for (i=0; i<scalp2.size(); i++) if (scalp2[i]) cnt++;
    FILE *fout=fopen("/home/ari/scratch1/dweinste/data/new/head.bdry", "wt");
    fprintf(fout, "%d\n", cnt);
    for (i=0; i<scalp2.size(); i++) if (scalp2[i]) fprintf(fout, "%d\n", i+1);
    cerr << "Outside error was "<<airCtr*100./(airCtr+othrCtr)<<" percent.\n";
    fclose(fout);
#endif

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
void BldEEGMesh::removeAirAndGreyMatlElems(Mesh *m, Array1<int>& cortexBCMeshNodes) {
    cerr << "START:  m->nodes.size()="<<m->nodes.size()<<"\n";

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


	    if (NoBrain) {
	    // this will remove all of the brain!
		m->elems[i]=0;
	    }


	} else if (e->cond == 0) { 
	    m->elems[i]=0;
	} else {
	    interiorNodes[e->n[0]]=1;
	    interiorNodes[e->n[1]]=1;
	    interiorNodes[e->n[2]]=1;
	    interiorNodes[e->n[3]]=1;
	}
    }

    // basically, we want to do a "Mesh::pack_all()" here, but we need
    // to keep track of the node mapping so we can put the cortical nodes
    // into cortexBCMeshNodes.  so, instead we do a Mesh::compute_neighbors(), 
    // we cut & paste Mesh::pack_nodes() with a couple extra lines to
    // track cortex nodes, and then we call Mesh::pack_elems()

    m->compute_neighbors();

    int nnodes=m->nodes.size();
    int idx=0;
    Array1<int> map(nnodes);
    for(i=0;i<nnodes;i++){
	NodeHandle& n=m->nodes[i];
//	if(n.get_rep()) {
	if(n.get_rep() && n->elems.size()){
	    map[i]=idx;
	    m->nodes[idx]=n;

	    // here's the extra code to track the cortex node numbers...

	    if (greyNodes[i] && interiorNodes[i]) {
                cortexBCMeshNodes.add(idx);
//                cerr << "cortex point: "<<m->nodes[idx]->p<<"\n";
            }
	    idx++;
	} else {
	    map[i]=-1234;
	}
    }
    m->nodes.resize(idx);
    int nelems=m->elems.size();
    for(i=0;i<nelems;i++){
	Element* e=m->elems[i];
	if(e){
	    for(int j=0;j<4;j++){
		if(map[e->n[j]]==-1234)
		    cerr << "Warning: pointing to old node: " << e->n[j] << endl;
		e->n[j]=map[e->n[j]];
	    }
	}
    }

    m->pack_elems();
    cerr << "FINISH:  m->nodes.size()="<<m->nodes.size()<<"\n";
}

// surftree has the scalp bc's already
// find the nearest mesh node, store its index in scalpBCMeshNodes and
//    give that node a BC as well.

void BldEEGMesh::applyScalpBCs(Mesh *m, SurfTree *st, 
			       Array1<int>& scalpBCMeshNodes,
			        const Array1<int>& cortexBCMeshNodes) {
    Array1<int> used(m->nodes.size());
    used.initialize(0);
    for (int i=0; i<cortexBCMeshNodes.size(); i++) 
	used[cortexBCMeshNodes[i]]=1;
    for (i=0; i<scalpBCMeshNodes.size(); i++) {
	int invalid=1;
	double dist;
	int closest;
	double bcVal;
	Point bcPt(st->nodes[st->idx[i]]);
	bcVal=st->data[i];
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
	scalpBCMeshNodes[i]=closest;

	if (AddBC) {
	    m->nodes[closest]->bc = new DirichletBC(SurfaceHandle(0), bcVal);
	}
    }
}

void BldEEGMesh::findCortexNodesInSTree(Mesh *m, SurfTree* st, int startCNode,
					const Array1<int>& cortexBCMeshNodes){
    Array1<int> isCortex(st->nodes.size());
    isCortex.initialize(0);
    int cortexIdx=-1;
    for (int i=0; i<st->surfI.size(); i++) {
	if (st->surfI[i].name == "cortex") cortexIdx=i;
    }
    if (cortexIdx == -1) {
	error("Error: no cortex in SurfTree!");
	return;
    }
    for (i=0; i<st->surfI[cortexIdx].faces.size(); i++) {
	TSElement *e=st->faces[st->surfI[cortexIdx].faces[i]];
	isCortex[e->i1]=isCortex[e->i2]=isCortex[e->i3]=1;
    }
    Array1<int> ctxPts;
    for (i=0; i<isCortex.size(); i++) if (isCortex[i]) {
	ctxPts.add(i);
//	cerr << "cortex point: "<<st->points[i]<<"\n";
    }
//    cerr << "NscalpPts="<<st->bcIdx.size();
    for (i=0; i<cortexBCMeshNodes.size(); i++) {
	Point p(m->nodes[cortexBCMeshNodes[i]]->p);
//	cerr << "p="<<p<<"  ";
	int idx=ctxPts[0];
	double d=(p-st->nodes[idx]).length2();
	for (int j=1; j<ctxPts.size(); j++) {
	    double dd=(p-st->nodes[ctxPts[j]]).length2();
	    if (dd<d) {
		d=dd; 
		idx=ctxPts[j];
	    }
	}
//	cerr << "closest surface point is: "<<st->points[idx]<<"  d="<<d<<"\n";
	st->idx.add(idx);
	st->data.add(0);
    }
//    cerr << "NbdryPts="<<st->bcIdx.size();

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

void BldEEGMesh::reorderPts(Mesh *m, const Array1<int>& scalpBCMeshNodes, 
			    const Array1<int>& cortexBCMeshNodes) {

    Array1<int> map, invMap;
    
    cerr << "in reorder pts... scalpBCMeshNodes.size()="<<scalpBCMeshNodes.size()<<"  cortexBCMeshNodes.size()="<<cortexBCMeshNodes.size()<<"  m->nodes.size()="<<m->nodes.size()<<"\n";

    for (int ii=0; ii<scalpBCMeshNodes.size(); ii++) {
	for (int jj=0; jj<cortexBCMeshNodes.size(); jj++) {
	    if (scalpBCMeshNodes[ii] == cortexBCMeshNodes[jj]) {
		cerr << "ERROR:  scalpBCMeshNodes["<<ii<<"] = cortexBCMeshNodes["<<jj<<"] = "<<cortexBCMeshNodes[jj]<<"\n";
	    }
	}
    }

    map.resize(m->nodes.size());
    invMap.resize(m->nodes.size());
    map.initialize(-1);

    for (int i=0; i<scalpBCMeshNodes.size(); i++) {
//	cerr << "i="<<i<<"  scalpBCMeshNodes[i]="<<scalpBCMeshNodes[i]<<"\n";
	map[scalpBCMeshNodes[i]]=i;
	invMap[i]=scalpBCMeshNodes[i];
        scalpBCMeshNodes[i]=i;
    }
    int fromEnd=m->nodes.size()-cortexBCMeshNodes.size();

    for (i=0; i<cortexBCMeshNodes.size(); i++) {
//	cerr << "i="<<i<<"   cortexBCMeshNodes[i]="<<cortexBCMeshNodes[i]<<"\n";
	map[cortexBCMeshNodes[i]]=fromEnd+i;
	m->nodes[cortexBCMeshNodes[i]]->fluxBC=1;
	invMap[fromEnd+i]=cortexBCMeshNodes[i];
        cortexBCMeshNodes[i]=fromEnd+i;
    }
    int curr=scalpBCMeshNodes.size();
    for (i=0; i<m->nodes.size(); i++) {
	if (map[i] == -1) {
	    map[i] = curr;
	    invMap[curr]=i;
	    curr++;
	}
    }
    Array1<NodeHandle> nh(m->nodes.size());
    for (i=0; i<m->nodes.size(); i++) nh[i]=m->nodes[invMap[i]];
    for (i=0; i<m->nodes.size(); i++) m->nodes[i]=nh[i];
    for (i=0; i<m->elems.size(); i++) {
	Element *e=m->elems[i];
	for (int j=0; j<4; j++) e->n[j]=map[e->n[j]];
    }
}

void BldEEGMesh::setConductivities(Mesh *m) {

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
}

void BldEEGMesh::execute()
{
    int numPts=npts.get();
    SegFldHandle sf;
    NoBrain = NOBRAIN.get();
    AddBC = ADDBC.get();
    update_state(NeedData);
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

    if (st->typ != SurfTree::NodeValuesSome) {
	cerr << "Error - BldEEGMesh needs data of type NodeValuesSome in STree.\n";
	return;
    }

    update_state(JustStarted);
    Mesh *m = new Mesh;

    setConductivities(m);

    Point min, max;
    sf->get_bounds(min,max);
    cerr << "SF (min,max) = "<<min<<" "<<max<<"\n";
    
    BBox bb;
    for (int iii=0; iii<st->nodes.size(); iii++) bb.extend(st->nodes[iii]);
    cerr << "Surf (min,max) = "<<bb.min()<<" "<<bb.max()<<"\n";


    int greyMatlIdx=findLargestGreyMatterIdx(sf);
    int whiteMatlIdx=findLargestWhiteMatterIdx(sf);
    cerr << "The largest grey matter component is: "<<greyMatlIdx<<"\n";
    cerr << "The largest white matter component is: "<<whiteMatlIdx<<"\n";
    cerr << "Generating points and tets...\n";
    genPtsAndTets(sf, st, numPts, m);
    cerr << "Calssifying elements...\n";
    classifyElements(sf, m, st, greyMatlIdx, whiteMatlIdx);
    cerr << "Done classifiying!\n";
    
#if 0
    Piostream* stream = scinew BinaryPiostream(clString("/home/ari/scratch1/dweinste/data/new/head.mesh"), Piostream::Write);
    Pio(*stream, MeshHandle(m));
    delete stream;
#endif

    Array1<int> cortexBCMeshNodes;

    // new we'll query a button on the interface for whether we want
    // a volume or a surface-to-surface problem


    removeAirAndGreyMatlElems(m, cortexBCMeshNodes);

    Array1<int> scalpBCMeshNodes(st->idx.size());
    applyScalpBCs(m, st, scalpBCMeshNodes, cortexBCMeshNodes);
    findCortexNodesInSTree(m, st, m->nodes.size()-cortexBCMeshNodes.size(),
			   cortexBCMeshNodes);

    reorderPts(m, scalpBCMeshNodes, cortexBCMeshNodes);

    cerr << "Mesh has been built and output -- "<<m->nodes.size()<<" nodes, "<<m->elems.size()<<" elements.\n";
    for (int i=0; i<m->elems.size(); i++) {
	m->elems[i]->mesh = m;
	m->elems[i]->orient();
	m->elems[i]->compute_basis();
    }
    m->compute_neighbors();
    omesh->send(m);
    ostree->send(st);
}
