
/*
 *  MorphMesher3d.cc:  Convert a surface into geoemtry
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/Surface.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/TriSurface.h>
#include <Classlib/Queue.h>
#include <Classlib/HashTable.h>
#include <Classlib/BitArray1.h>
#include <Geometry/Point.h>
#include <Classlib/String.h>
#include <Geometry/BBox.h>
#include <Classlib/Array1.h>

#define FACE1 8
#define FACE2 4
#define FACE3 2
#define FACE4 1
#define ALLFACES (FACE1|FACE2|FACE3|FACE4)

struct MCubeTable {
    int which_case;
    int permute[8];
    int nbrs;
};

#include <Modules/Visualization/mcube.h>

#define NUM_LAYERS	10

class MorphMesher3d : public Module {
    int in_node_base;
    int in_elem_base;
    int out_node_base;
    int out_elem_base;
    int cond_index;
    Array1<SurfaceIPort*> isurfaces;
    MeshOPort* omesh;
    void mesh_single_surf(const Array1<SurfaceHandle> &surfaces, Mesh*);
    void morph_mesher_3d(const Array1<SurfaceHandle> &surfaces, Mesh*);
    void lace_surfaces(TriSurface *out, TriSurface *in, Mesh *mesh);
    void lace_surfaces(const SurfaceHandle &out, TriSurface *in, Mesh *mesh);
    void lace_surfaces(const Point &mid, Mesh *mesh);
    int iso_cube_ext(Point *ov, double *oval, TriSurface* surf);
    double get_value(int x, int y, int z, double t, TriSurface* surf, Point p);
public:
    MorphMesher3d(const clString& id);
    MorphMesher3d(const MorphMesher3d&, int deep);
    virtual ~MorphMesher3d();
    virtual Module* clone(int deep);
    virtual void connection(Module::ConnectionMode, int, int);
    virtual void execute();
};

static Module* make_MorphMesher3d(const clString& id)
{
    return new MorphMesher3d(id);
}

static RegisterModule db1("Surfaces", "MorphMesher3d", make_MorphMesher3d);
static RegisterModule db2("Mesh", "MorphMesher3d", make_MorphMesher3d);
static RegisterModule db3("Dave", "MorphMesher3d", make_MorphMesher3d);

MorphMesher3d::MorphMesher3d(const clString& id)
: Module("MorphMesher3d", id, Filter)
{
    // Create the input port
    isurfaces.add(new SurfaceIPort(this, "Surface", SurfaceIPort::Atomic));
    add_iport(isurfaces[0]);
    omesh=new MeshOPort(this, "Mesh", MeshIPort::Atomic);
    add_oport(omesh);
}

MorphMesher3d::MorphMesher3d(const MorphMesher3d&copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("MorphMesher3d::MorphMesher3d");
}

MorphMesher3d::~MorphMesher3d()
{
}

Module* MorphMesher3d::clone(int deep)
{
    return new MorphMesher3d(*this, deep);
}


void MorphMesher3d::connection(ConnectionMode mode, int which_port,
			       int output)
{
    if (output) return;
    if (mode==Disconnected) {
	remove_iport(which_port);
	delete isurfaces[which_port];
	isurfaces.remove(which_port);
    } else {
	SurfaceIPort* si=new SurfaceIPort(this,"Surface",SurfaceIPort::Atomic);
	add_iport(si);
	isurfaces.add(si);
    }
}

void MorphMesher3d::execute()
{
    Array1<SurfaceHandle> surfs(isurfaces.size()-1);
    for (int flag=0, i=0; i<isurfaces.size()-1; i++)
	if (!isurfaces[i]->get(surfs[i]))
	    flag=1;
    if (flag) return;
    if (surfs[0]->getTriSurface()==0) {
	error("MorphMesher3d only works with TriSurfaces");
	return;
    }
    Mesh* mesh=new Mesh;
    morph_mesher_3d(surfs, mesh);
    omesh->send(mesh);
}

int MorphMesher3d::iso_cube_ext(Point *ov,double* oval,TriSurface*surf){
    int mask=0;
    for(int idx=1;idx<=8;idx++){
	if(oval[idx]<0)
	    mask|=1<<(idx-1);
    }
    MCubeTable* tab=&mcube_table[mask];
    double val[9];
    Point v[9];
    for(idx=1;idx<=8;idx++){
	val[idx]=oval[tab->permute[idx-1]];
	v[idx]=ov[tab->permute[idx-1]];
    }
    int wcase=tab->which_case;
    switch(wcase){
    case 0:
	break;
    case 1:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    surf->cautious_add_triangle(p1,p2,p3);
	}
	break;
    case 2:
	{
	    Point p1(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p2(Interpolate(v[2], v[6], val[2]/(val[2]-val[6])));
	    Point p3(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    surf->cautious_add_triangle(p1,p2,p3);
	    Point p4(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    surf->cautious_add_triangle(p3,p4,p1);
	}
	break;
    case 3:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    surf->cautious_add_triangle(p1,p2,p3);
	    Point p4(Interpolate(v[3], v[2], val[3]/(val[3]-val[2])));
	    Point p5(Interpolate(v[3], v[7], val[3]/(val[3]-val[7])));
	    Point p6(Interpolate(v[3], v[4], val[3]/(val[3]-val[4])));
	    surf->cautious_add_triangle(p4,p5,p6);
	}
	break;
    case 4:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    surf->cautious_add_triangle(p1,p2,p3);
	    Point p4(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    Point p5(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    Point p6(Interpolate(v[7], v[6], val[7]/(val[7]-val[6])));
	    surf->cautious_add_triangle(p4,p5,p6);
	}
	break;
    case 5:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[5], v[1], val[5]/(val[5]-val[1])));
	    surf->cautious_add_triangle(p1,p2,p3);
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    surf->cautious_add_triangle(p4,p3,p2);
	    Point p5(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    surf->cautious_add_triangle(p5,p4,p2);
	}
	break;
    case 6:
	{
	    Point p1(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p2(Interpolate(v[2], v[6], val[2]/(val[2]-val[6])));
	    Point p3(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    surf->cautious_add_triangle(p1,p2,p3);
	    Point p4(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    surf->cautious_add_triangle(p3,p4,p1);
	    Point p5(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    Point p6(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    Point p7(Interpolate(v[7], v[6], val[7]/(val[7]-val[6])));
	    surf->cautious_add_triangle(p5,p6,p7);
	}
	break;
    case 7:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[2], v[6], val[2]/(val[2]-val[6])));
	    surf->cautious_add_triangle(p1,p2,p3);
	    Point p4(Interpolate(v[4], v[1], val[4]/(val[4]-val[1])));
	    Point p5(Interpolate(v[4], v[3], val[4]/(val[4]-val[3])));
	    Point p6(Interpolate(v[4], v[8], val[4]/(val[4]-val[8])));
	    surf->cautious_add_triangle(p4,p5,p6);
	    Point p7(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    Point p8(Interpolate(v[7], v[6], val[7]/(val[7]-val[6])));
	    Point p9(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    surf->cautious_add_triangle(p7,p8,p9);
	}
	break;
    case 8:
	{
	    Point p1(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    surf->cautious_add_triangle(p1,p2,p3);
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    surf->cautious_add_triangle(p4,p1,p3);
	}
	break;
    case 9:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    surf->cautious_add_triangle(p1,p2,p3);
	    Point p4(Interpolate(v[8], v[7], val[8]/(val[8]-val[7])));
	    surf->cautious_add_triangle(p1,p3,p4);
	    Point p5(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    surf->cautious_add_triangle(p1,p4,p5);
	    Point p6(Interpolate(v[8], v[4], val[8]/(val[8]-val[4])));
	    surf->cautious_add_triangle(p5,p4,p6);
	}
	break;
    case 10:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[4], v[3], val[4]/(val[4]-val[3])));
	    Point p3(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    surf->cautious_add_triangle(p1,p2,p3);
	    Point p4(Interpolate(v[4], v[8], val[4]/(val[4]-val[8])));
	    surf->cautious_add_triangle(p2,p4,p3);
	    Point p5(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p6(Interpolate(v[6], v[5], val[6]/(val[6]-val[5])));
	    Point p7(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    surf->cautious_add_triangle(p5,p6,p7);
	    Point p8(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    surf->cautious_add_triangle(p2,p8,p3);
	}
	break;
    case 11:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p3(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    surf->cautious_add_triangle(p1,p2,p3);
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    surf->cautious_add_triangle(p1,p3,p4);
	    Point p5(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    surf->cautious_add_triangle(p1,p4,p5);
	    Point p6(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    surf->cautious_add_triangle(p4,p3,p6);
	}
	break;
    case 12:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[5], v[1], val[5]/(val[5]-val[1])));
	    surf->cautious_add_triangle(p1,p2,p3);
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    surf->cautious_add_triangle(p3, p2, p4);
	    Point p5(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    surf->cautious_add_triangle(p5, p2, p5);
	    Point p6(Interpolate(v[4], v[1], val[4]/(val[4]-val[1])));
	    Point p7(Interpolate(v[4], v[3], val[4]/(val[4]-val[3])));
	    Point p8(Interpolate(v[4], v[8], val[4]/(val[4]-val[8])));
	    surf->cautious_add_triangle(p6, p7, p8);
	}
	break;
    case 13:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    surf->cautious_add_triangle(p1, p2, p3);
	    Point p4(Interpolate(v[3], v[2], val[3]/(val[3]-val[2])));
	    Point p5(Interpolate(v[3], v[7], val[3]/(val[3]-val[7])));
	    Point p6(Interpolate(v[3], v[4], val[3]/(val[3]-val[4])));
	    surf->cautious_add_triangle(p4, p5, p6);
	    Point p7(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p8(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    Point p9(Interpolate(v[6], v[5], val[6]/(val[6]-val[5])));
	    surf->cautious_add_triangle(p7, p8, p9);
	    Point p10(Interpolate(v[8], v[5], val[8]/(val[8]-val[5])));
	    Point p11(Interpolate(v[8], v[7], val[8]/(val[8]-val[7])));
	    Point p12(Interpolate(v[8], v[4], val[8]/(val[8]-val[4])));
	    surf->cautious_add_triangle(p10, p11, p12);
	}
	break;
    case 14:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    surf->cautious_add_triangle(p1, p2, p3);
	    Point p4(Interpolate(v[8], v[4], val[8]/(val[8]-val[4])));
	    surf->cautious_add_triangle(p1, p3, p4);
	    Point p5(Interpolate(v[5], v[1], val[5]/(val[5]-val[1])));
	    surf->cautious_add_triangle(p1, p4, p5);
	    Point p6(Interpolate(v[8], v[7], val[8]/(val[8]-val[7])));
	    surf->cautious_add_triangle(p3, p6, p4);
	}
	break;
    default:
	error("Bad case in marching cubes!\n");
	break;
    }
    return(tab->nbrs);
}

double MorphMesher3d::get_value(int x, int y, int z, double t,
				TriSurface *surf, Point p) {

    Point a((double)x, (double)y, (double)z);
    double dp=(a-p).length();
    Array1<int> res;
    double ds=surf->distance(a,res);
    return((1-t)*ds-t*dp);
}

void MorphMesher3d::mesh_single_surf(const Array1<SurfaceHandle> &surfs, 
				     Mesh *mesh) {

    TriSurface* ts=surfs[0]->getTriSurface();
    BBox bb;
    Point mid;
    for (int i=0; i<ts->points.size(); i++) {
	mid=Point(0,0,0)+(ts->points[i]-(-mid));
	bb.extend(ts->points[i]);
    }
    mid.x(mid.x()/i);
    mid.y(mid.y()/i);
    mid.z(mid.z()/i);
    double radius=(bb.max()-bb.min()).length()/2;
    int CUBE_SIZE=(int) (radius/NUM_LAYERS);
    TriSurface *last_surf=0;
    TriSurface *new_surf=new TriSurface;
    for (i=0; i<NUM_LAYERS; i++) {
	double t=(i+1)/(NUM_LAYERS+1);
	Point s((ts->points[0])-((ts->points[0]-mid)*t));
	int px, py, pz;
	int nx, ny, nz, xmin, ymin, zmin;
	xmin=(int)bb.min().x(); ymin=(int)bb.min().y(); zmin=(int)bb.min().z();
	nx=(int)(bb.max().x()-xmin); ny=(int)(bb.max().y()-ymin); 
	nz=(int)(bb.max().z()-zmin);
	px=(int)s.x();
	py=(int)s.y();
	pz=(int)s.z();
	HashTable<int, int> visitedPts;
	Queue<int> surfQ;
	int pLoc=((((pz-zmin)*ny)+(py-ymin))*nx)+(px-xmin);
	int dummy;
	visitedPts.insert(pLoc, 0);
	surfQ.append(pLoc);
	while(!surfQ.is_empty()) {
	    pLoc=surfQ.pop();
	    pz=pLoc/(nx*ny)+zmin;
	    dummy=pLoc%(nx*ny);
	    py=dummy/nx+ymin;
	    px=dummy%nx+xmin;
	    Point ov[9];
	    double oval[9];
	    ov[1]=Point(px, py, pz);
	    ov[2]=Point(px+CUBE_SIZE, py, pz);
	    ov[3]=Point(px+CUBE_SIZE, py+CUBE_SIZE, pz);
	    ov[4]=Point(px, py+CUBE_SIZE, pz);
	    ov[5]=Point(px, py, pz+CUBE_SIZE);
	    ov[6]=Point(px+CUBE_SIZE, py, pz+CUBE_SIZE);
	    ov[7]=Point(px+CUBE_SIZE, py+CUBE_SIZE, pz+CUBE_SIZE);
	    ov[8]=Point(px, py+CUBE_SIZE, pz+CUBE_SIZE);
	    oval[1]=get_value(px,py,pz,t,ts,mid);
	    oval[2]=get_value(px+CUBE_SIZE,py,pz,t,ts,mid);
	    oval[3]=get_value(px+CUBE_SIZE,py+CUBE_SIZE,pz,t,ts,mid);
	    oval[4]=get_value(px,py+CUBE_SIZE,pz,t,ts,mid);
	    oval[5]=get_value(px,py,pz+CUBE_SIZE,t,ts,mid);
	    oval[6]=get_value(px+CUBE_SIZE,py,pz+CUBE_SIZE,t,ts,mid);
	    oval[7]=get_value(px+CUBE_SIZE,py+CUBE_SIZE,pz+CUBE_SIZE,t,ts,mid);
	    oval[8]=get_value(px,py+CUBE_SIZE,pz+CUBE_SIZE,t,ts,mid);
	    int nbrs=iso_cube_ext(ov, oval, new_surf);
	    if ((nbrs & 1) && (px!=xmin)) {
		pLoc-=1;
		if (!visitedPts.lookup(pLoc, dummy)) {
		    visitedPts.insert(pLoc, 0);
		    surfQ.append(pLoc);
		}
		pLoc+=1;
	    }
	    if ((nbrs & 2) && (px!=xmin+nx-2)) {
		pLoc+=1;
		if (!visitedPts.lookup(pLoc, dummy)) {
		    visitedPts.insert(pLoc, 0);
		    surfQ.append(pLoc);
		}
		pLoc-=1;
	    }
	    if ((nbrs & 8) && (py!=ymin)) {
		pLoc-=nx;
		if (!visitedPts.lookup(pLoc, dummy)) {
		    visitedPts.insert(pLoc, 0);
		    surfQ.append(pLoc);
		}
		pLoc+=nx;
	    }
	    if ((nbrs & 4) && (py!=ymin+ny-2)) {
		pLoc+=nx;
		if (!visitedPts.lookup(pLoc, dummy)) {
		    visitedPts.insert(pLoc, 0);
		    surfQ.append(pLoc);
		}
		pLoc-=nx;
	    }
	    if ((nbrs & 16) && (pz!=zmin)) {
		pLoc-=nx*ny;
		if (!visitedPts.lookup(pLoc, dummy)) {
		    visitedPts.insert(pLoc, 0);
		    surfQ.append(pLoc);
		}
		pLoc+=nx*ny;
	    }
	    if ((nbrs & 32) && (pz!=zmin+nz-2)) {
		pLoc+=nx*ny;
		if (!visitedPts.lookup(pLoc, dummy)) {
		    visitedPts.insert(pLoc, 0);
		    surfQ.append(pLoc);
		}
		pLoc-=nx*ny;
	    }
	}
	if (i==0) {
	    lace_surfaces(surfs[0], new_surf, mesh);
	} else {
	    lace_surfaces(last_surf, new_surf, mesh);
	}
	if (i==NUM_LAYERS-1) {
	    lace_surfaces(mid, mesh);
	}
	last_surf=new_surf;
    }
}


// We have our surfaces, and the mesh that we're building.  Now, we have
// to mesh between the outside surface (the 0th one), and the others.

void MorphMesher3d::morph_mesher_3d(const Array1<SurfaceHandle> &surfs, 
				    Mesh *mesh) {

    mesh->cond_tensors.add(surfs[0]->conductivity);
    if (surfs.size()==1)	// only an exterior surface, solid inside
	mesh_single_surf(surfs, mesh);
    else {
	error("Haven't implemented interior surfaces yet in MM3d");
    }
}


// need to add all the points of out to the mesh, and also add the
// triangles with the first three nodes set to the triangle vertices
// need to store the base index -- the difference between a point's 
// index in the surface points array and the mesh points array.

void MorphMesher3d::lace_surfaces(const SurfaceHandle &outHand, TriSurface* in,
				  Mesh *mesh) {
    in_node_base=in_elem_base=0;
    cond_index=mesh->cond_tensors.size();
    TriSurface* out=outHand->getTriSurface();

    Array1<double> *cond=new Array1<double>(6);
    for (int i=0; i<6; i++)
	(*cond)[i]=out->conductivity[i];
    mesh->cond_tensors.add(*cond);

    for (i=0; i<out->points.size(); i++)
	mesh->nodes.add(new Node(out->points[i]));

    for (i=0; i<out->elements.size(); i++) {
	mesh->elems.add(new Element(mesh, out->elements[i]->i1,
				    out->elements[i]->i2,
				    out->elements[i]->i3, -1));
	mesh->elems[i]->cond=cond_index;
    }

    lace_surfaces(out, in, mesh);
}

void MorphMesher3d::lace_surfaces(const Point &mid, Mesh *mesh) {

    out_node_base=in_node_base;
    out_elem_base=in_elem_base;
    in_node_base=mesh->nodes.size();
    in_elem_base=mesh->elems.size();
    int ptid=in_node_base;

    mesh->nodes.add(new Node(mid));
    for (int i=out_elem_base; i<in_elem_base; i++)
	mesh->elems[i]->n[3]=ptid;
}


// The idea here is that the mesh already contains all of the triangles from
// the outter surface, and those triangles are stored as incomplete elements.
// We will go through and add all of the triangles and points from the inner
// surface, and will lace these two together.
// More specifically, we will complete the mesh elements from the outter to the
// inner, create finished elements from the inner to the outter, and buld
// partial elements from the inner (which will be completed the next time
// this method is called).

void MorphMesher3d::lace_surfaces(TriSurface *out, TriSurface *in, Mesh* mesh){
    out_node_base=in_node_base;
    out_elem_base=in_elem_base;
    in_node_base=mesh->nodes.size();
    in_elem_base=mesh->elems.size();
    
    // first add all of the new points and triangles from the inner surface
    for (int i=0; i<in->points.size(); i++)
	mesh->nodes.add(new Node(in->points[i]));
    for (i=0; i<in->elements.size(); i++) {
	mesh->elems.add(new Element(mesh, out->elements[i]->i1+in_node_base,
				    out->elements[i]->i2+in_node_base,
				    out->elements[i]->i3+in_node_base, -1));
	mesh->elems[i+in_elem_base]->cond=cond_index;
    }

    // now lace all of the outer triangles to an inner point
    for (i=0; i<out->elements.size(); i++) {
	int pid=in->get_closest_vertex_id(out->points[out->elements[i]->i1],
					  out->points[out->elements[i]->i2],
					  out->points[out->elements[i]->i3]);
	mesh->elems[i+out_elem_base]->n[3]=pid+in_node_base;
    }

    // now lace all of the inner triangles to an outer point
    for (i=0; i<in->elements.size(); i++) {
	int pid=out->get_closest_vertex_id(in->points[out->elements[i]->i1],
					   in->points[out->elements[i]->i2],
					   in->points[out->elements[i]->i3]);
	mesh->elems[i+in_elem_base]->n[3]=pid+out_node_base;
    }
}

