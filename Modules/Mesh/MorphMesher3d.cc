
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
#include <Datatypes/Surface.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/TriSurface.h>
#include <Classlib/Queue.h>
#include <Classlib/HashTable.h>
#include <Classlib/BitArray1.h>
#include <Classlib/String.h>
#include <Classlib/Array1.h>
#include <Geometry/Point.h>
#include <Geometry/BBox.h>
#include <Geometry/Grid.h>
#include <Malloc/Allocator.h>
#include <Math/Expon.h>
#include <TCL/TCLvar.h>
#include <stdlib.h>		//rand()
///
//#include <Datatypes/ScalarFieldRG.h>
//#include <Datatypes/ScalarFieldPort.h>
///

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

class MorphMesher3d : public Module {
    int in_node_base;
    int in_elem_base;
    int out_node_base;
    int out_elem_base;
    int cond_index;
    TCLint num_layers;

    Array1<SurfaceIPort*> isurfaces;
    MeshOPort* omesh;
///
//    ScalarFieldOPort* ofield;
//    ScalarFieldRG* field;
///
    void mesh_single_surf(const Array1<SurfaceHandle> &surfaces, Mesh*);
    void mesh_mult_surfs(const Array1<SurfaceHandle> &surfs, Mesh *mesh);
    void morph_mesher_3d(const Array1<SurfaceHandle> &surfaces, Mesh*);
    void lace_surfaces(TriSurface *out, TriSurface *in, Mesh *mesh);
    void lace_surfaces(const SurfaceHandle &out, TriSurface *in, Mesh *mesh);
    void lace_surfaces(const Point &mid, Mesh *mesh);
    int iso_cube_ext(Point *ov, double *oval, TriSurface* surf);
    double get_value(double x, double y, double z, double t, TriSurface* surf,
		     Point p);
    void find_a_crossing(Point *s, TriSurface *ts, const Point &p, double t);
    double get_value(double x, double y, double z, double t, 
		     TriSurface* inner, TriSurface *outer);
    void find_a_crossing(Point *s, TriSurface* , TriSurface*, 
			 double t);
public:
    MorphMesher3d(const clString& id);
    MorphMesher3d(const MorphMesher3d&, int deep);
    virtual ~MorphMesher3d();
    virtual Module* clone(int deep);
    virtual void connection(Module::ConnectionMode, int, int);
    virtual void execute();
};

extern "C" {
Module* make_MorphMesher3d(const clString& id)
{
    return scinew MorphMesher3d(id);
}
};

MorphMesher3d::MorphMesher3d(const clString& id)
: Module("MorphMesher3d", id, Filter), num_layers("num_layers", id, this)
{
    // Create the input port
    isurfaces.add(scinew SurfaceIPort(this, "Surface", SurfaceIPort::Atomic));
    add_iport(isurfaces[0]);
    omesh=scinew MeshOPort(this, "Mesh", MeshIPort::Atomic);
    add_oport(omesh);
///
//    ofield=scinew ScalarFieldOPort(this, "Field", ScalarFieldIPort::Atomic);
//    add_oport(ofield);
///
}

MorphMesher3d::MorphMesher3d(const MorphMesher3d&copy, int deep)
: Module(copy, deep), num_layers("num_layers", id, this)
{
    NOT_FINISHED("MorphMesher3d::MorphMesher3d");
}

MorphMesher3d::~MorphMesher3d()
{
}

Module* MorphMesher3d::clone(int deep)
{
    return scinew MorphMesher3d(*this, deep);
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
	SurfaceIPort* si=scinew SurfaceIPort(this,"Surface",SurfaceIPort::Atomic);
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
    Mesh* mesh=scinew Mesh;
///
//    field=scinew ScalarFieldRG();
///
    morph_mesher_3d(surfs, mesh);
    omesh->send(MeshHandle(mesh));
///
//    ofield->send(field);
///
}

void MorphMesher3d::find_a_crossing(Point *s, TriSurface *ts, const Point &p, 
				    double t){
    Point pout;
    Array1<int> res;
    /*double dist=*/ts->distance(p,res,&pout);

    *s=AffineCombination(p,t,pout,(1-t));
}

void MorphMesher3d::find_a_crossing(Point *s, TriSurface *outer,
				    TriSurface *inner, double t) {
    Point pout;
//    double dist;
    Array1<int> res;

// This way might be too slow...
//
//    int index=-1;
//    for (int i=0; i<inner.points.size(); i++) {
//	double tdist=outer.distance(inner.points[i], res, &pout);
//	if ((index==-1) || (Abs(tdist)<Abs(dist))) {
//	    index=i;
//	    dist=tdist;
//	    *s=pout;
//	}
//    }
//    *s=AffineCombination(inner.points[i],t,*s,(1-t));

// But this way isn't always correct -- ie it fails for some concave inner ts.
    outer->distance(inner->points[0],res,&pout);
    *s=AffineCombination(inner->points[0],t,pout,(1-t));
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
	    surf->cautious_add_triangle(p1,p2,p3,1);
	}
	break;
    case 2:
	{
	    Point p1(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p2(Interpolate(v[2], v[6], val[2]/(val[2]-val[6])));
	    Point p3(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    surf->cautious_add_triangle(p1,p2,p3,1);
	    Point p4(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    surf->cautious_add_triangle(p3,p4,p1,1);
	}
	break;
    case 3:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    surf->cautious_add_triangle(p1,p2,p3,1);
	    Point p4(Interpolate(v[3], v[2], val[3]/(val[3]-val[2])));
	    Point p5(Interpolate(v[3], v[7], val[3]/(val[3]-val[7])));
	    Point p6(Interpolate(v[3], v[4], val[3]/(val[3]-val[4])));
	    surf->cautious_add_triangle(p4,p5,p6,1);
	}
	break;
    case 4:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    surf->cautious_add_triangle(p1,p2,p3,1);
	    Point p4(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    Point p5(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    Point p6(Interpolate(v[7], v[6], val[7]/(val[7]-val[6])));
	    surf->cautious_add_triangle(p4,p5,p6,1);
	}
	break;
    case 5:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[5], v[1], val[5]/(val[5]-val[1])));
	    surf->cautious_add_triangle(p1,p2,p3,1);
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    surf->cautious_add_triangle(p4,p3,p2,1);
	    Point p5(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    surf->cautious_add_triangle(p5,p4,p2,1);
	}
	break;
    case 6:
	{
	    Point p1(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p2(Interpolate(v[2], v[6], val[2]/(val[2]-val[6])));
	    Point p3(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    surf->cautious_add_triangle(p1,p2,p3,1);
	    Point p4(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    surf->cautious_add_triangle(p3,p4,p1,1);
	    Point p5(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    Point p6(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    Point p7(Interpolate(v[7], v[6], val[7]/(val[7]-val[6])));
	    surf->cautious_add_triangle(p5,p6,p7,1);
	}
	break;
    case 7:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[2], v[6], val[2]/(val[2]-val[6])));
	    surf->cautious_add_triangle(p1,p2,p3,1);
	    Point p4(Interpolate(v[4], v[1], val[4]/(val[4]-val[1])));
	    Point p5(Interpolate(v[4], v[3], val[4]/(val[4]-val[3])));
	    Point p6(Interpolate(v[4], v[8], val[4]/(val[4]-val[8])));
	    surf->cautious_add_triangle(p4,p5,p6,1);
	    Point p7(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    Point p8(Interpolate(v[7], v[6], val[7]/(val[7]-val[6])));
	    Point p9(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    surf->cautious_add_triangle(p7,p8,p9,1);
	}
	break;
    case 8:
	{
	    Point p1(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    surf->cautious_add_triangle(p1,p2,p3,1);
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    surf->cautious_add_triangle(p4,p1,p3,1);
	}
	break;
    case 9:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    surf->cautious_add_triangle(p1,p2,p3,1);
	    Point p4(Interpolate(v[8], v[7], val[8]/(val[8]-val[7])));
	    surf->cautious_add_triangle(p1,p3,p4,1);
	    Point p5(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    surf->cautious_add_triangle(p1,p4,p5,1);
	    Point p6(Interpolate(v[8], v[4], val[8]/(val[8]-val[4])));
	    surf->cautious_add_triangle(p5,p4,p6,1);
	}
	break;
    case 10:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[4], v[3], val[4]/(val[4]-val[3])));
	    Point p3(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    surf->cautious_add_triangle(p1,p2,p3,1);
	    Point p4(Interpolate(v[4], v[8], val[4]/(val[4]-val[8])));
	    surf->cautious_add_triangle(p2,p4,p3,1);
	    Point p5(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p6(Interpolate(v[6], v[5], val[6]/(val[6]-val[5])));
	    Point p7(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    surf->cautious_add_triangle(p5,p6,p7,1);
	    Point p8(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    surf->cautious_add_triangle(p2,p8,p3,1);
	}
	break;
    case 11:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p3(Interpolate(v[7], v[3], val[7]/(val[7]-val[3])));
	    surf->cautious_add_triangle(p1,p2,p3,1);
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    surf->cautious_add_triangle(p1,p3,p4,1);
	    Point p5(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    surf->cautious_add_triangle(p1,p4,p5,1);
	    Point p6(Interpolate(v[7], v[8], val[7]/(val[7]-val[8])));
	    surf->cautious_add_triangle(p4,p3,p6,1);
	}
	break;
    case 12:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[5], v[1], val[5]/(val[5]-val[1])));
	    surf->cautious_add_triangle(p1,p2,p3,1);
	    Point p4(Interpolate(v[5], v[8], val[5]/(val[5]-val[8])));
	    surf->cautious_add_triangle(p3, p2, p4,1);
	    Point p5(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    surf->cautious_add_triangle(p4, p2, p5);
	    Point p6(Interpolate(v[4], v[1], val[4]/(val[4]-val[1])));
	    Point p7(Interpolate(v[4], v[3], val[4]/(val[4]-val[3])));
	    Point p8(Interpolate(v[4], v[8], val[4]/(val[4]-val[8])));
	    surf->cautious_add_triangle(p6, p7, p8,1);
	}
	break;
    case 13:
	{
	    Point p1(Interpolate(v[1], v[2], val[1]/(val[1]-val[2])));
	    Point p2(Interpolate(v[1], v[5], val[1]/(val[1]-val[5])));
	    Point p3(Interpolate(v[1], v[4], val[1]/(val[1]-val[4])));
	    surf->cautious_add_triangle(p1, p2, p3,1);
	    Point p4(Interpolate(v[3], v[2], val[3]/(val[3]-val[2])));
	    Point p5(Interpolate(v[3], v[7], val[3]/(val[3]-val[7])));
	    Point p6(Interpolate(v[3], v[4], val[3]/(val[3]-val[4])));
	    surf->cautious_add_triangle(p4, p5, p6,1);
	    Point p7(Interpolate(v[6], v[2], val[6]/(val[6]-val[2])));
	    Point p8(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    Point p9(Interpolate(v[6], v[5], val[6]/(val[6]-val[5])));
	    surf->cautious_add_triangle(p7, p8, p9,1);
	    Point p10(Interpolate(v[8], v[5], val[8]/(val[8]-val[5])));
	    Point p11(Interpolate(v[8], v[7], val[8]/(val[8]-val[7])));
	    Point p12(Interpolate(v[8], v[4], val[8]/(val[8]-val[4])));
	    surf->cautious_add_triangle(p10, p11, p12,1);
	}
	break;
    case 14:
	{
	    Point p1(Interpolate(v[2], v[1], val[2]/(val[2]-val[1])));
	    Point p2(Interpolate(v[2], v[3], val[2]/(val[2]-val[3])));
	    Point p3(Interpolate(v[6], v[7], val[6]/(val[6]-val[7])));
	    surf->cautious_add_triangle(p1, p2, p3,1);
	    Point p4(Interpolate(v[8], v[4], val[8]/(val[8]-val[4])));
	    surf->cautious_add_triangle(p1, p3, p4,1);
	    Point p5(Interpolate(v[5], v[1], val[5]/(val[5]-val[1])));
	    surf->cautious_add_triangle(p1, p4, p5,1);
	    Point p6(Interpolate(v[8], v[7], val[8]/(val[8]-val[7])));
	    surf->cautious_add_triangle(p3, p6, p4,1);
	}
	break;
    default:
	error("Bad case in marching cubes!\n");
	break;
    }
    return(tab->nbrs);
}

double MorphMesher3d::get_value(double x, double y, double z, double t,
				TriSurface *surf, Point p) {

    Point a(x,y,z);
    double dp=(a-p).length();	// this is always going to be positive   
    Array1<int> res;
    double ds=surf->distance(a,res);
//    if (ds>0) ds=-ds;		// force distance to surface to be negative
    return((1-t)*ds+t*dp);
}

double MorphMesher3d::get_value(double x, double y, double z, double t,
				TriSurface *inner, 
				TriSurface *outer) {
    Point a(x,y,z);
    Array1<int> res;
    double din=inner->distance(a,res);
//    if (din<0) din=-din;	// force distance to inner to be positive
    res.remove_all();
    double dout=outer->distance(a,res);
//    if (dout>0) dout=-dout;	// force distance to outer to be negative
    return((1-t)*dout+t*din);
}

void count_grid(const Grid &grid, int tris) {
    int total=0;
    Point min(grid.get_min());
    Array1<int> count(400);
    for (int i=0; i<400; i++) {
	count[i]=0;
    }
    for (int a=0; a<grid.dim1(); a++) {
	for (int b=0; b<grid.dim2(); b++) {
	    for (int c=0; c<grid.dim3(); c++) {
		Array1<int>* e=grid.get_members(a,b,c);
		if (e) {
		    int tmp=e->size();
		    count[tmp]++;
		    total+=tmp;
		}
	    }
	}
    }
    cerr << "   Total # of triangles: " << tris << "\n";
    cerr << "   Total # of intersections: " << total << "\n";
    cerr << "   Density: " << total*1./tris << "\n";
    cerr << "   Grid size = (" << grid.dim1() << "," << grid.dim2() << "," << grid.dim3() << ")\n";
    cerr << "   Min. Point = (" << min.x() << "," << min.y() << "," << min.z()
	<< ")\n";
    cerr << "   Spacing = " << grid.get_spacing() << "\n";
    for (i=0; i<400; i++) {
	if (count[i] != 0) {
	    cerr << "     " << i << ": " << count[i] << "\n";
	}
    }
    cerr << "\n\n";
}


void MorphMesher3d::mesh_mult_surfs(const Array1<SurfaceHandle> &surfs, 
				    Mesh *mesh) {
    if (surfs.size() > 2) {
	ASSERT(!"Can't deal with more than one inner-surface yet!");
    }
    BBox bb;

    TriSurface* outer=surfs[0]->getTriSurface();
    TriSurface* inner=surfs[1]->getTriSurface();
    if(!outer->grid)
	outer->construct_grid();
    double gr_sp=outer->grid->get_spacing();

    inner->construct_grid(outer->grid->dim1()+1, outer->grid->dim2()+1, 
			  outer->grid->dim3()+1, 
			  outer->grid->get_min()+
			  Vector((rand()%100)/(-100.)*gr_sp,
				 (rand()%100)/(-100.)*gr_sp,
				 (rand()%100)/(-100.)*gr_sp),
			  gr_sp);

    for (int i=0; i<outer->points.size(); i++) {
	bb.extend(outer->points[i]);
    }
    double radius=(bb.max()-bb.min()).length()/2;

    // we need to scale our cube size for the marching cubes -- we want 
    // them approx. as big as the triangles on the outside surface.
    // We determine the # of triangles on the surface, estimate the surface
    // area as if it were a sphere (4*Pi*r^2), and take the sqrt of the
    // quotient to approx. the scale for each side of our marching cube.
    // Sqrt(#tris/(4*Pi*r^2)) ~= Sqrt(#tris)/(r*3.5)   We double this since
    // the average triangle will have an area closer to .5 than 1

    double scale=Sqrt(outer->elements.size())/(radius*2);
    TriSurface *last_surf=0;
    TriSurface *new_surf;

//cerr << "Outer Grid:\n";
//count_grid(*outer->grid, outer->elements.size());

    for (i=0; i<num_layers.get(); i++) {
        new_surf=scinew TriSurface;
	new_surf->construct_grid(outer->grid->dim1()+1, outer->grid->dim2()+1, 
				 outer->grid->dim3()+1, 
				 outer->grid->get_min()+
				    Vector((rand()%100)/(-100.)*gr_sp,
					   (rand()%100)/(-100.)*gr_sp,
					   (rand()%100)/(-100.)*gr_sp),
				 gr_sp);
        double t=(i+1.0)/(num_layers.get()+1.0);
cerr << "MorphMesher starting on t=" << t << "\n";	
	Array1<int> res;

	Point s;
	find_a_crossing(&s, outer, inner, t);
	s=s*scale;

	int px, py, pz;
	int nx, ny, nz, xmin, ymin, zmin;
	xmin=Floor(bb.min().x()*scale); 
	ymin=Floor(bb.min().y()*scale); 
	zmin=Floor(bb.min().z()*scale);
	nx=Floor(bb.max().x()*scale-xmin+1); 
	ny=Floor(bb.max().y()*scale-ymin+1); 
	nz=Floor(bb.max().z()*scale-zmin+1);
	px=Floor(s.x());
	py=Floor(s.y());
	pz=Floor(s.z());
	HashTable<int, int> visitedPts;
	Queue<int> surfQ;
///
//    field->resize(nx,ny,nz);
//    for (int xx=0; xx<nx; xx++)
//	for (int yy=0; yy<ny; yy++)
//	    for (int zz=0; zz<nz; zz++)
//		field->grid(xx,yy,zz)=get_value((xmin+xx)/scale,
//						(ymin+yy)/scale,
//						(zmin+zz)/scale,
//						 t,inner,outer);
//   field->set_minmax(Point(xmin/scale,ymin/scale,zmin/scale),
//		      Point((xmin+nx-1)/scale,(ymin+ny-1)/scale,
//			    (zmin+nz-1)/scale));
///
	int pLoc=((((pz-zmin)*ny)+(py-ymin))*nx)+(px-xmin);
	int dummy;
	visitedPts.insert(pLoc, 0);
	surfQ.append(pLoc);
// just to be sure, we'll add all of its neighbors as starting points as well
	if (px!=xmin) {
	    pLoc-=1;
	    visitedPts.insert(pLoc,0);
	    surfQ.append(pLoc);
	    pLoc+=1;
	}
	if (px!=xmin+nx-1) {
	    pLoc+=1;
	    visitedPts.insert(pLoc,0);
	    surfQ.append(pLoc);
	    pLoc-=1;
	}
	if (py!=ymin) {
	    pLoc-=nx;
	    visitedPts.insert(pLoc,0);
	    surfQ.append(pLoc);
	    pLoc+=nx;
	}
	if (py!=ymin+ny-1) {
	    pLoc+=ny;
	    visitedPts.insert(pLoc,0);
	    surfQ.append(pLoc);
	    pLoc-=ny;
	}
	if (pz!=zmin) {
	    pLoc-=nx*ny;
	    visitedPts.insert(pLoc,0);
	    surfQ.append(pLoc);
	    pLoc+=nx*ny;
	}
	if (pz!=zmin+nz-1) {
	    pLoc+=nx*ny;
	    visitedPts.insert(pLoc,0);
	    surfQ.append(pLoc);
	    pLoc-=nx*ny;
	}

	int ndone=0;
	while(!surfQ.is_empty()) {
	    update_progress(ndone, surfQ.length()+ndone);
	    ndone++;
	    pLoc=surfQ.pop();
	    ASSERT((pLoc < nx*ny*nz) && (pLoc >= 0));
	    pz=pLoc/(nx*ny)+zmin;
	    dummy=pLoc%(nx*ny);
	    py=dummy/nx+ymin;
	    px=dummy%nx+xmin;
//cerr << "\nVisiting: " << pLoc <<" (" << px << "," << py << "," << pz << ")\n";
	    Point ov[9];
	    double oval[9];
	    ov[1]=Point(px/scale, py/scale, pz/scale);
	    ov[2]=Point((px+1)/scale, py/scale, pz/scale);
	    ov[3]=Point((px+1)/scale, (py+1)/scale, pz/scale);
	    ov[4]=Point(px/scale, (py+1)/scale, pz/scale);
	    ov[5]=Point(px/scale, py/scale, (pz+1)/scale);
	    ov[6]=Point((px+1)/scale, py/scale, (pz+1)/scale);
	    ov[7]=Point((px+1)/scale, (py+1)/scale, (pz+1)/scale);
	    ov[8]=Point(px/scale, (py+1)/scale, (pz+1)/scale);
	    oval[1]=get_value(px/scale,py/scale,pz/scale,t,inner,outer);
	    oval[2]=get_value((px+1)/scale,py/scale,pz/scale,t,inner,outer);
	    oval[3]=get_value((px+1)/scale,(py+1)/scale,pz/scale,t,inner,outer);
	    oval[4]=get_value(px/scale,(py+1)/scale,pz/scale,t,inner,outer);
	    oval[5]=get_value(px/scale,py/scale,(pz+1)/scale,t,inner,outer);
	    oval[6]=get_value((px+1)/scale,py/scale,(pz+1)/scale,t,inner,outer);
	    oval[7]=get_value((px+1)/scale,(py+1)/scale,(pz+1)/scale,t,inner,outer);
	    oval[8]=get_value(px/scale,(py+1)/scale,(pz+1)/scale,t,inner,outer);
	    int nbrs=iso_cube_ext(ov, oval, new_surf);
//cerr << "Nbrs: " << nbrs; 

	    if ((nbrs & 1) && (px!=xmin)) {
//cerr << " -x";
		pLoc-=1;
		if (!visitedPts.lookup(pLoc, dummy)) {
//cerr << "(" << pLoc << ")";
		    visitedPts.insert(pLoc, 0);
		    surfQ.append(pLoc);
		}
		pLoc+=1;
	    }
	    if ((nbrs & 2) && (px!=nx+xmin-1)) {
//cerr << " +x";
		pLoc+=1;
		if (!visitedPts.lookup(pLoc, dummy)) {
//cerr << "(" << pLoc << ")";
		    visitedPts.insert(pLoc, 0);
		    surfQ.append(pLoc);
		}
		pLoc-=1;
	    }
	    if ((nbrs & 8) && (py!=ymin)) {
//cerr << " -y";
		pLoc-=nx;
		if (!visitedPts.lookup(pLoc, dummy)) {
//cerr << "(" << pLoc << ")";
		    visitedPts.insert(pLoc, 0);
		    surfQ.append(pLoc);
		}
		pLoc+=nx;
	    }
	    if ((nbrs & 4) && (py!=ny+ymin-1)) {
//cerr << " +y";
		pLoc+=nx;
		if (!visitedPts.lookup(pLoc, dummy)) {
//cerr << "(" << pLoc << ")";
		    visitedPts.insert(pLoc, 0);
		    surfQ.append(pLoc);
		}
		pLoc-=nx;
	    }
	    if ((nbrs & 16) && (pz!=zmin)) {
//cerr << " -z";
		pLoc-=nx*ny;
		if (!visitedPts.lookup(pLoc, dummy)) {
//cerr << "(" << pLoc << ")";
		    visitedPts.insert(pLoc, 0);
		    surfQ.append(pLoc);
		}
		pLoc+=nx*ny;
	    }
	    if ((nbrs & 32) && (pz!=nz+zmin-1)) {
//cerr << " +z";
		pLoc+=nx*ny;
		if (!visitedPts.lookup(pLoc, dummy)) {
//cerr << "(" << pLoc << ")";
		    visitedPts.insert(pLoc, 0);
		    surfQ.append(pLoc);
		}
		pLoc-=nx*ny;
	    }
	}

//cerr << "Grid t=" << t << ":\n";
//count_grid(*new_surf->grid, new_surf->elements.size());

	if (!new_surf->elements.size())
	    ASSERT(!"Didn't get any elements on this surface!");
	if (i==0) {
	    lace_surfaces(surfs[0], new_surf, mesh);
	    last_surf=new_surf;
	} else {
	    lace_surfaces(last_surf, new_surf, mesh);
	    delete last_surf;
	    last_surf=new_surf;
	}
	if (i==num_layers.get()-1) {
	    lace_surfaces(last_surf, surfs[1]->getTriSurface(), mesh);
	    delete last_surf;
	}
	if (abort_flag) return;
    }

//cerr << "Inner Grid:\n";
//count_grid(*inner->grid, inner->elements.size());

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
    mid+=Vector(0.00001,0.00001,0.00001);
    mid.x(mid.x()/i);
    mid.y(mid.y()/i);
    mid.z(mid.z()/i);
    double radius=(bb.max()-bb.min()).length()/2;

    // we need to scale our cube size for the marching cubes -- we want 
    // them approx. as big as the triangles on the outside surface.
    // We determine the # of triangles on the surface, estimate the surface
    // area as if it were a sphere (4*Pi*r^2), and take the sqrt of the
    // quotient to approx. the scale for each side of our marching cube.
    // Sqrt(#tris/(4*Pi*r^2)) ~= Sqrt(#tris)/(r*3.5)   We double this since
    // the average triangle will have an area closer to .5 than 1

    double scale=Sqrt(ts->elements.size())/(radius*2);
    TriSurface *last_surf=0;
    TriSurface *new_surf;

    for (i=0; i<num_layers.get(); i++) {
        new_surf=scinew TriSurface;
        double gr_sp=ts->grid->get_spacing();
	new_surf->construct_grid(ts->grid->dim1()+1, ts->grid->dim2()+1, 
				 ts->grid->dim3()+1, 
				 ts->grid->get_min()+
				    Vector((rand()%100)/(-100.)*gr_sp,
					   (rand()%100)/(-100.)*gr_sp,
					   (rand()%100)/(-100.)*gr_sp),
				 gr_sp);
        double t=(i+1.0)/(num_layers.get()+1.0);
	Array1<int> res;

	Point s;
	find_a_crossing(&s, ts, mid, t);
	s=s*scale;

	int px, py, pz;
	int nx, ny, nz, xmin, ymin, zmin;
	xmin=Floor(bb.min().x()*scale); 
	ymin=Floor(bb.min().y()*scale); 
	zmin=Floor(bb.min().z()*scale);
	nx=Floor(bb.max().x()*scale-xmin+1); 
	ny=Floor(bb.max().y()*scale-ymin+1); 
	nz=Floor(bb.max().z()*scale-zmin+1);
	px=Floor(s.x());
	py=Floor(s.y());
	pz=Floor(s.z());
	HashTable<int, int> visitedPts;
	Queue<int> surfQ;
	int pLoc=((((pz-zmin)*ny)+(py-ymin))*nx)+(px-xmin);
	int dummy;
	visitedPts.insert(pLoc, 0);
	surfQ.append(pLoc);
	while(!surfQ.is_empty()) {
	    pLoc=surfQ.pop();
	    ASSERT((pLoc < nx*ny*nz) && (pLoc >= 0));
	    pz=pLoc/(nx*ny)+zmin;
	    dummy=pLoc%(nx*ny);
	    py=dummy/nx+ymin;
	    px=dummy%nx+xmin;
	    Point ov[9];
	    double oval[9];
	    ov[1]=Point(px/scale, py/scale, pz/scale);
	    ov[2]=Point((px+1)/scale, py/scale, pz/scale);
	    ov[3]=Point((px+1)/scale, (py+1)/scale, pz/scale);
	    ov[4]=Point(px/scale, (py+1)/scale, pz/scale);
	    ov[5]=Point(px/scale, py/scale, (pz+1)/scale);
	    ov[6]=Point((px+1)/scale, py/scale, (pz+1)/scale);
	    ov[7]=Point((px+1)/scale, (py+1)/scale, (pz+1)/scale);
	    ov[8]=Point(px/scale, (py+1)/scale, (pz+1)/scale);
	    oval[1]=get_value(px/scale,py/scale,pz/scale,t,ts,mid);
	    oval[2]=get_value((px+1)/scale,py/scale,pz/scale,t,ts,mid);
	    oval[3]=get_value((px+1)/scale,(py+1)/scale,pz/scale,t,ts,mid);
	    oval[4]=get_value(px/scale,(py+1)/scale,pz/scale,t,ts,mid);
	    oval[5]=get_value(px/scale,py/scale,(pz+1)/scale,t,ts,mid);
	    oval[6]=get_value((px+1)/scale,py/scale,(pz+1)/scale,t,ts,mid);
	    oval[7]=get_value((px+1)/scale,(py+1)/scale,(pz+1)/scale,t,ts,mid);
	    oval[8]=get_value(px/scale,(py+1)/scale,(pz+1)/scale,t,ts,mid);
	    int nbrs=iso_cube_ext(ov, oval, new_surf);
	    if ((nbrs & 1) && (px!=xmin)) {
		pLoc-=1;
		if (!visitedPts.lookup(pLoc, dummy)) {
		    visitedPts.insert(pLoc, 0);
		    surfQ.append(pLoc);
		}
		pLoc+=1;
	    }
	    if ((nbrs & 2) && (px!=nx-1)) {
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
	    if ((nbrs & 4) && (py!=ny-1)) {
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
	    if ((nbrs & 32) && (pz!=nz-1)) {
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
	    last_surf=new_surf;
	} else {
	    lace_surfaces(last_surf, new_surf, mesh);
	    delete last_surf;
	    last_surf=new_surf;
	}
	if (i==(num_layers.get()-1)) {
	    lace_surfaces(mid, mesh);
	    delete last_surf;
	}
	if (abort_flag) return;
    }
}


// We have our surfaces, and the mesh that we're building.  Now, we have
// to mesh between the outside surface (the 0th one), and the others.

void MorphMesher3d::morph_mesher_3d(const Array1<SurfaceHandle> &surfs, 
				    Mesh *mesh) {

    mesh->cond_tensors.add(surfs[0]->conductivity);
    for (int i=0; i<surfs.size(); i++) {
	TriSurface *ts=surfs[i]->getTriSurface();
	if (!ts->is_directed())
	    error("Can't run MorphMesher3d::morph_mesher_3d with undirected surfaces");
    }
    if (surfs.size()==1)	// only an exterior surface, solid inside
	mesh_single_surf(surfs, mesh);
    else {
	mesh_mult_surfs(surfs, mesh);
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

    Array1<double> *cond=scinew Array1<double>(6);
    if (out->conductivity.size()!=0) {
	for (int i=0; i<6; i++) {
	    (*cond)[i]=out->conductivity[i];
	}
    } else {
	(*cond)[0]=(*cond)[3]=(*cond)[5]=1;
	(*cond)[1]=(*cond)[2]=(*cond)[4]=0;
    }
    mesh->cond_tensors.add(*cond);

    for (int i=0; i<out->points.size(); i++)
	mesh->nodes.add(NodeHandle(new Node(out->points[i])));

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

    mesh->nodes.add(NodeHandle(new Node(mid)));
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
	mesh->nodes.add(NodeHandle(new Node(in->points[i])));
    for (i=0; i<in->elements.size(); i++) {
	mesh->elems.add(new Element(mesh, in->elements[i]->i1+in_node_base,
				    in->elements[i]->i2+in_node_base,
				    in->elements[i]->i3+in_node_base, -1));
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
	int pid=out->get_closest_vertex_id(in->points[in->elements[i]->i1],
					   in->points[in->elements[i]->i2],
					   in->points[in->elements[i]->i3]);
	mesh->elems[i+in_elem_base]->n[3]=pid+out_node_base;
    }
}


