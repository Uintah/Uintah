
#include <Packages/rtrt/Core/Grid2.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/MiscMath.h>
#include <iostream>
#include <stdlib.h>

using namespace rtrt;
using namespace std;
using SCIRun::Thread;
using SCIRun::Time;

namespace rtrt {

Grid2::Grid2(Object* obj, int nsides)
    : Object(0), obj(obj), nsides(nsides)
{
    grid=0;
}

Grid2::~Grid2()
{
}

static inline void calc_se(const BBox& obj_bbox, const BBox& bbox,
			   const Vector& diag,
			   int nx, int ny, int nz,
			   int& sx, int& sy, int& sz,
			   int& ex, int& ey, int& ez)
{
    Vector s((obj_bbox.min()-bbox.min())/diag);
    Vector e((obj_bbox.max()-bbox.min())/diag);
    sx = (s.x() < -1e-6) ? (int)(s.x()*nx) - 1 : (int)(s.x()*nx - 1e-7);
    sy = (s.y() < -1e-6) ? (int)(s.y()*ny) - 1 : (int)(s.y()*ny - 1e-7);
    sz = (s.z() < -1e-6) ? (int)(s.z()*nz) - 1 : (int)(s.z()*nz - 1e-7);
    ex = (e.x() < -1e-6) ? (int)(e.x()*nx) - 1 : (int)(e.x()*nx - 1e-7);
    ey = (e.y() < -1e-6) ? (int)(e.y()*ny) - 1 : (int)(e.y()*ny - 1e-7);
    ez = (e.z() < -1e-6) ? (int)(e.z()*nz) - 1 : (int)(e.z()*nz - 1e-7);
}

void Grid2::recompute_bbox ()
{
}

void Grid2::rebuild ()
{
  int maxradius = 0;

  cerr << "Rebuilding grid\n";
  double time=Time::currentSeconds();

  bbox = logical_bbox;        // Grid is now expanded to include logical bbox
  Vector diag (logical_bbox.diagonal());

  int ngrid = nx * ny * nz;
  Array1<Object *>*new_grid = new Array1<Object*>[ngrid];
  for (int n = 0; n < ngrid; n++)
  {
    new_grid[n]    = Array1<Object *>(1, 1, -1);
    new_grid[n][0] = (Object *)1;
  }

  for(int i=0;i<prims.size();i++)
  {
    BBox obj_bbox;
    prims[i]->compute_bounds(obj_bbox, maxradius);
    prims[i]->set_grid_position (1, 0);
    int sx, sy, sz, ex, ey, ez;
    calc_se(obj_bbox, bbox, diag, nx, ny, nz, sx, sy, sz, ex, ey, ez);
    for(int x=sx;x<=ex;x++)
      for(int y=sy;y<=ey;y++)
      {
	int idx = x * nynz + y * nz + sz;
	for(int z=sz;z<=ez;z++){
	  new_grid[idx].add (prims[i]);
	  new_grid[idx][0] = (Object *)((long)new_grid[idx][0] + 1);
	  idx++;
	}
      }
  }

  Array1<Object *>* old_grid = grid;
  grid = new_grid;
  if (old_grid)
    delete [] old_grid;

  cerr << "Rebuilding took " << Time::currentSeconds() - time << " seconds\n";
}

void Grid2::preprocess(double maxradius, int& pp_offset, int& scratchsize)
{
    cerr << "Building grid\n";
    double time=Time::currentSeconds();
    obj->preprocess(maxradius, pp_offset, scratchsize);
    cerr << "1/7 Preprocess took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();

    obj->collect_prims(prims);
    cerr << "2/7 Collect prims took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();

    bbox.reset();
    obj->compute_bounds(bbox, maxradius);
    cerr << "3/7 Compute bounds took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();

    int ncells=nsides*nsides*nsides;
    bbox.extend(bbox.min()-Vector(1.e-3, 1.e-3, 1.e-3));
    bbox.extend(bbox.max()+Vector(1.e-3, 1.e-3, 1.e-3));
    Vector diag(bbox.diagonal());
    bbox.extend(bbox.min()-diag*1.e-3);
    bbox.extend(bbox.max()+diag*1.e-3);
    logical_bbox = bbox;
    diag         = bbox.diagonal();
    double volume=diag.x()*diag.y()*diag.z();
    double c=cbrt(ncells/volume);
    nx= Max ((int)(c*diag.x()+0.5), 2);
    ny= Max ((int)(c*diag.y()+0.5), 2);
    nz= Max ((int)(c*diag.z()+0.5), 2);
    nynz=ny*nz;
    int ngrid=nx*ny*nz;
    cerr << "Computing " << nx << 'x' << ny << 'x' << nz << " grid for " << ngrid << " cells (wanted " << ncells << ")\n";

#if 0
    for (int i = 0; i < prims.size(); i++)
      if (prims[i]->is_motionspline ())
	prims[i]->set_scene (this);
#endif

    rebuild ();
    cerr << "Done building grid for " << prims.size() << " primitives.\n";
}

void Grid2::intersect(Ray& ray, HitInfo& hit,
		      DepthStats* st, PerProcessorContext* ppc)
{
    int    phys_far_x;
    int    phys_far_y;
    int    phys_far_z;
    int    phys_near_x;
    int    phys_near_y;
    int    phys_near_z;
    int    logic_far_x;
    int    logic_far_y;
    int    logic_far_z;

    const  Vector dir(ray.direction());
    const  Point orig(ray.origin());
    Point  min(bbox.min());
    Point  max(bbox.max());
    Vector diag(bbox.diagonal());
    Point  lmin  (logical_bbox.min());
    Point  lmax  (logical_bbox.max());

    Vector logical_min ((logical_bbox.min()-min)/diag);
    Vector logical_max ((logical_bbox.max()-min)/diag);

    double inv_dir=1./dir.x();
    double MIN, MAX, lMIN, lMAX;
    int dix_dx, ddx;
    if(dir.x() > 0){
	MIN         = inv_dir * (min.x()  - orig.x());
	MAX         = inv_dir * (max.x()  - orig.x());
	lMIN        = inv_dir * (lmin.x() - orig.x());
	lMAX        = inv_dir * (lmax.x() - orig.x());
	dix_dx      =  1;
	ddx         =  1;
	phys_near_x =  0;
	phys_far_x  = nx - 1;
	logic_far_x = (logical_max.x() < 0) ? (int)(logical_max.x()*nx)
                                            : (int)(logical_max.x()*nx) + 1;
    } else {
	MIN         = inv_dir * (max.x()  - orig.x());
	MAX         = inv_dir * (min.x()  - orig.x());
	lMIN        = inv_dir * (lmax.x() - orig.x());
	lMAX        = inv_dir * (lmin.x() - orig.x());
	dix_dx      = -1;
	ddx         =  0;
	phys_near_x = nx - 1;
	phys_far_x  = 0;
	logic_far_x = (logical_min.x() < 0) ? (int)(logical_min.x()*nx) - 2
	                                    : (int)(logical_min.x()*nx) - 1;
    }

    inv_dir = 1. / dir.y();
    double y0, y1, ly0, ly1;
    int diy_dy, ddy;
    if(dir.y() > 0){
	y0          = inv_dir * (min.y()  - orig.y());
	y1          = inv_dir * (max.y()  - orig.y());
	ly0         = inv_dir * (lmin.y() - orig.y());
	ly1         = inv_dir * (lmax.y() - orig.y());
	diy_dy      =  1;
	ddy         =  1;
	phys_near_y =  0;
	phys_far_y  = ny - 1;
	logic_far_y = (logical_max.y() < 0) ? (int)(logical_max.y()*ny)
	                                    : (int)(logical_max.y()*ny) + 1;
    } else {
	y0          = inv_dir * (max.y()  - orig.y());
	y1          = inv_dir * (min.y()  - orig.y());
	ly0         = inv_dir * (lmax.y() - orig.y());
	ly1         = inv_dir * (lmin.y() - orig.y());
	diy_dy      = -1;
	ddy         =  0;
	phys_near_y = ny - 1;
	phys_far_y  = 0;
	logic_far_y = (logical_min.y() < 0) ? (int)(logical_min.y()*ny) - 2
	                                    : (int)(logical_min.y()*ny) - 1;
    }
    if(y0>MIN)
	MIN=y0;
    if(y1<MAX)
	MAX=y1;
    if(ly0>lMIN)
	lMIN=ly0;
    if(ly1<lMAX)
	lMAX=ly1;
    if(lMAX<lMIN)
	return;

    inv_dir = 1. / dir.z();
    double z0, z1, lz0, lz1;
    int diz_dz, ddz;
    if(dir.z() > 0){
	z0          = inv_dir * (min.z()  - orig.z());
	z1          = inv_dir * (max.z()  - orig.z());
	lz0         = inv_dir * (lmin.z() - orig.z());
	lz1         = inv_dir * (lmax.z() - orig.z());
	diz_dz      =  1;
	ddz         =  1;
	phys_near_z =  0;
	phys_far_z  = nz - 1;
	logic_far_z = (logical_max.z() < 0) ? (int)(logical_max.z()*nz)
	                                    : (int)(logical_max.z()*nz) + 1;
    } else {
	z0          = inv_dir * (max.z()  - orig.z());
	z1          = inv_dir * (min.z()  - orig.z());
	lz0         = inv_dir * (lmax.z() - orig.z());
	lz1         = inv_dir * (lmin.z() - orig.z());
	diz_dz      = -1;
	ddz         =  0;
	phys_near_z = nz - 1;
	phys_far_z  =  0;
	logic_far_z = (logical_min.z() < 0) ? (int)(logical_min.z()*nz) - 2
 	                                    : (int)(logical_min.z()*nz) - 1;
   }
    if(z0>MIN)
	MIN=z0;
    if(z1<MAX)
	MAX=z1;
    if(lz0>lMIN)
	lMIN=lz0;
    if(lz1<lMAX)
	lMAX=lz1;
    if(lMAX<lMIN)
	return;

    double t;
    if(MIN > 1.e-6){
	t=MIN;
    } else if(MAX > 1.e-6){
	t=0;
    }

    double lt;
    if(lMIN > 1.e-6){
	lt=lMIN;
    } else if(lMAX > 1.e-6){
	lt=0;
    } else
        return;
    if(lt>1.e29)
	return;

    Point  lp(orig+dir*lt);                  // Logical grid
    Vector ls((lp-min)/diag);
    int lix        = (ls.x() < 0) ? (int)(ls.x()*nx) - 1 : (int)(ls.x()*nx);
    int liy        = (ls.y() < 0) ? (int)(ls.y()*ny) - 1 : (int)(ls.y()*ny);
    int liz        = (ls.z() < 0) ? (int)(ls.z()*nz) - 1 : (int)(ls.z()*nz);
    int ix         = lix;
    int iy         = liy;
    int iz         = liz;
    while (ix >= nx)   ix -= nx;
    while (ix < 0)     ix += nx;
    while (iy >= ny)   iy -= ny;
    while (iy < 0)     iy += ny;
    while (iz >= nz)   iz -= nz;
    while (iz < 0)     iz += nz;
    int phys_min_x = (phys_near_x < phys_far_x) ? phys_near_x : phys_far_x;
    int phys_min_y = (phys_near_y < phys_far_y) ? phys_near_y : phys_far_y;
    int phys_min_z = (phys_near_z < phys_far_z) ? phys_near_z : phys_far_z;
    int phys_max_x = (phys_near_x > phys_far_x) ? phys_near_x : phys_far_x;
    int phys_max_y = (phys_near_y > phys_far_y) ? phys_near_y : phys_far_y;
    int phys_max_z = (phys_near_z > phys_far_z) ? phys_near_z : phys_far_z;
    int logical_x  = (lix < phys_min_x) | (lix > phys_max_x);
    int logical_y  = (liy < phys_min_y) | (liy > phys_max_y);
    int logical_z  = (liz < phys_min_z) | (liz > phys_max_z);
    int logical    = (logical_x | logical_y | logical_z) + 1;
    double x       = min.x() + diag.x() * double(lix + ddx) / nx;
    double y       = min.y() + diag.y() * double(liy + ddy) / ny;
    double z       = min.z() + diag.z() * double(liz + ddz) / nz;
    double next_x  = (x - orig.x()) / dir.x();
    double next_y  = (y - orig.y()) / dir.y();
    double next_z  = (z - orig.z()) / dir.z();
    double dtdx    = dix_dx * diag.x() / nx / dir.x();
    double dtdy    = diy_dy * diag.y() / ny / dir.y();
    double dtdz    = diz_dz * diag.z() / nz / dir.z();
    hit.min_t      = MAXDOUBLE;
    
    Object **cell;
    while (hit.min_t > t) {
	cell = grid[ix * nynz + iy * nz + iz].get_objs();
	for(int i = 1; i < (long)cell[0]; i++)
	  if (cell[i]->get_id () & logical)
	    cell[i]->intersect(ray, hit, st, ppc);
	if(next_x < next_y && next_x < next_z){
	    lix      += dix_dx;
	    if (lix == logic_far_x)
	      return;
	    ix        = lix;
	    while (ix >= nx)   ix -= nx;
	    while (ix < 0)     ix += nx;
	    t         = next_x;
	    logical_x = (lix < phys_min_x) | (lix > phys_max_x);
	    next_x   += dtdx;
	} else if(next_y < next_z){
	    liy      += diy_dy;
	    if (liy == logic_far_y)
	      return;
	    iy        = liy;
	    while (iy >= ny)   iy -= ny;
	    while (iy < 0)     iy += ny;
	    t         = next_y;
	    logical_y = (liy < phys_min_y) | (liy > phys_max_y);
	    next_y   += dtdy;
	} else {
	    liz      += diz_dz;
	    if (liz == logic_far_z)
	      return;
	    iz        = liz;
	    while (iz >= nz)   iz -= nz;
	    while (iz < 0)     iz += nz;
	    t         = next_z;
	    logical_z = (liz < phys_min_z) | (liz > phys_max_z);
	    next_z   += dtdz;
	}
	logical = (logical_x | logical_y | logical_z) + 1;
    }
}

void Grid2::animate(double, bool&)
{
    //obj->animate(t);
}

void Grid2::collect_prims(Array1<Object*>& prims)
{
    prims.add(this);
}

void Grid2::compute_bounds(BBox& bbox, double offset)
{
    obj->compute_bounds(bbox, offset);
}

Vector Grid2::normal(const Point&, const HitInfo&)
{
    cerr << "Error: Grid normal should not be called!\n";
    return Vector(0,0,0);
}

void Grid2::remove(Object* object, const BBox& obj_bbox)
{
  Array1<Object *> voxel;
  int sx, sy, sz, ex, ey, ez, px, py, pz;
  int idx;

  calc_se(obj_bbox, bbox, bbox.diagonal(), nx, ny, nz, sx, sy, sz, ex, ey, ez);
  for(int x=sx;x<=ex;x++)
    for(int y=sy;y<=ey;y++)
      for(int z=sz;z<=ez;z++)
      {
	px = x;
	py = y;
	pz = z;
	while (px >= nx) px -= nx;  // Appears to be faster than taking (px + largenx)%nx
	while (px < 0)   px += nx;
	while (py >= ny) py -= ny;
	while (py < 0)   py += ny;
	while (pz >= nz) pz -= nz;
	while (pz < 0)   pz += nz;

	idx = px * nynz + py * nz + pz;
	voxel = grid[idx];
	int object_count = (long)voxel[0];
	for (int c = 1; c < object_count; c++)
	  if (object == voxel[c])
	  {
	    grid[idx].remove (c);
	    object_count--;
	  }
	grid[idx][0] = (Object *)((int)object_count);
      }
}

void Grid2::insert(Object* object, const BBox& obj_bbox)
{
  int sx, sy, sz, ex, ey, ez, px, py, pz;
  int idx;
  int tag_in  = 0;
  int tag_out = 0;

  calc_se(obj_bbox, bbox, bbox.diagonal(), nx, ny, nz, sx, sy, sz, ex, ey, ez);
  logical_bbox.extend (obj_bbox);

  for(int x=sx;x<=ex;x++){
    for(int y=sy;y<=ey;y++){
      for(int z=sz;z<=ez;z++){
	px = x;
	py = y;
	pz = z;
	while (px >= nx) px -= nx;  // Appears to be faster than taking (px + largenx)%nx
	while (px < 0)   px += nx;
	while (py >= ny) py -= ny;
	while (py < 0)   py += ny;
	while (pz >= nz) pz -= nz;
	while (pz < 0)   pz += nz;
	idx      = px * nynz + py * nz + pz;
	tag_out += ((px != x) || (py != y) || (pz != z));
	tag_in  += ((px == x) && (py == y) && (pz == z));
	grid[idx].add(object);
	grid[idx][0] = (Object *)((long)grid[idx][0] + 1);
      }
    }
  }
  object->set_grid_position (tag_in > 0, tag_out > 0);
}

} // end namespace


/*


#if 1
    cerr << "Physical bounding box:\n";
    bbox.print();
    cerr << "Logical bounding box:\n";
    logical_bbox.print();
#endif

#if 1
      cerr << "Start logicals: " << logical_x << " " << logical_y << " " << logical_z << "\n";
      cerr << "Spos " << sx << " " << sy << " " << sz << "\n";
      cerr << "Lpos " << lix << " " << liy << " " << liz << "\n";
      cerr << "Ipos " << ix << " " << iy << " " << iz << "\n";
      cerr << "Lfar " << logic_far_x << " " << logic_far_y << " " << logic_far_z << "\n";
#endif

    cerr << "Physical near: " << phys_near_x << " " << phys_near_y << " " << phys_near_z << "\n";

    cerr << "Physical far : " << phys_far_x << " " << phys_far_y << " " << phys_far_z << "\n";

*/
