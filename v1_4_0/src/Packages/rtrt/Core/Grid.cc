
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <iostream>
#include <stdlib.h>

using namespace rtrt;
using SCIRun::Thread;
using SCIRun::Time;

Grid::Grid(Object* obj, int nsides)
    : Object(0), obj(obj), nsides(nsides)
{
    grid=0;
    counts=0;
}

Grid::~Grid()
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
    sx=(int)(s.x()*nx);
    sy=(int)(s.y()*ny);
    sz=(int)(s.z()*nz);
    ex=(int)(e.x()*nx);
    ey=(int)(e.y()*ny);
    ez=(int)(e.z()*nz);
    if(sx < 0 || ex >= nx){
	cerr << "NX out of bounds!\n";
	cerr << "sx=" << sx << ", ex=" << ex << '\n';
	cerr << "e=" << e << '\n';
	cerr << "obj_bbox=" << obj_bbox.min() << ", " << obj_bbox.max() << '\n';
	cerr << "bbox=" << bbox.min() << ", " << bbox.max() << '\n';
	cerr << "diag=" << diag << '\n';
	exit(1);
    }
    if(sy < 0 || ey >= ny){
	cerr << "NY out of bounds!\n";
	cerr << "sy=" << sy << ", ey=" << ey << '\n';
	exit(1);
    }
    if(sz < 0 || ez >= nz){
	cerr << "NZ out of bounds!\n";
	cerr << "sz=" << sz << ", ez=" << ez << '\n';
	exit(1);
    }
}

void Grid::preprocess(double maxradius, int& pp_offset, int& scratchsize)
{
    cerr << "Building grid\n";
    double time=Time::currentSeconds();
    obj->preprocess(maxradius, pp_offset, scratchsize);
    cerr << "1/7 Preprocess took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();

    Array1<Object*> prims;
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
    bbox.extend(bbox.max()+diag*1.e-3);
    bbox.extend(bbox.min()-diag*1.e-3);
    diag=bbox.diagonal();
    double volume=diag.x()*diag.y()*diag.z();
    double c=cbrt(ncells/volume);
    nx=(int)(c*diag.x()+0.5);
    ny=(int)(c*diag.y()+0.5);
    nz=(int)(c*diag.z()+0.5);
    if(nx<2)
	nx=2;
    if(ny<2)
	ny=2;
    if(nz<2)
	nz=2;
    int ngrid=nx*ny*nz;
    cerr << "Computing " << nx << 'x' << ny << 'x' << nz << " grid for " << ngrid << " cells (wanted " << ncells << ")\n";


    if(counts)
	delete[] counts;
    if(grid)
	delete[] grid;
    counts=new int[2*ngrid];
    for(int i=0;i<ngrid*2;i++)
	counts[i]=0;

    double itime=time;
    int nynz=ny*nz;
    for(int i=0;i<prims.size();i++){
	double tnow=Time::currentSeconds();
	if(tnow-itime > 5.0){
	    cerr << i << "/" << prims.size() << '\n';
	    itime=tnow;
	}
	BBox obj_bbox;
	prims[i]->compute_bounds(obj_bbox, maxradius);
	int sx, sy, sz, ex, ey, ez;
	calc_se(obj_bbox, bbox, diag, nx, ny, nz, sx, sy, sz, ex, ey, ez);
	for(int x=sx;x<=ex;x++){
	    for(int y=sy;y<=ey;y++){
		int idx=x*nynz+y*nz+sz;
		for(int z=sz;z<=ez;z++){
		    counts[idx*2+1]++;
		    idx++;
		}
	    }
	}
    }

    cerr << "4/7 Counting cells took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();
    int total=0;
    for(int i=0;i<ngrid;i++){
	int count=counts[i*2+1];
	counts[i*2]=total;
	total+=count;
    }
    cerr << "Allocating " << total << " grid cells (" << double(total)/prims.size() << " per object, " << double(total)/ngrid << " per cell)\n";
    grid=new Object*[total];
    for(int i=0;i<total;i++)
	grid[i]=0;
    cerr << "5/7 Calculating offsets took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();
    itime=time;
    Array1<int> current(ngrid);
    current.initialize(0);
    for(int i=0;i<prims.size();i++){
	double tnow=Time::currentSeconds();
	if(tnow-itime > 5.0){
	    cerr << i << "/" << prims.size() << '\n';
	    itime=tnow;
	}
	BBox obj_bbox;
	prims[i]->compute_bounds(obj_bbox, maxradius);
	int sx, sy, sz, ex, ey, ez;
	calc_se(obj_bbox, bbox, diag, nx, ny, nz, sx, sy, sz, ex, ey, ez);
	for(int x=sx;x<=ex;x++){
	    for(int y=sy;y<=ey;y++){
		int idx=x*nynz+y*nz+sz;
		for(int z=sz;z<=ez;z++){
		    int cur=current[idx];
		    int pos=counts[idx*2]+cur;
		    grid[pos]=prims[i];
		    current[idx]++;
		    idx++;
		}
	    }
	}
    }
    cerr << "6/7 Filling grid took " << Time::currentSeconds()-time << " seconds\n";
    time=Time::currentSeconds();
    for(int i=0;i<ngrid;i++){
	if(current[i] != counts[i*2+1]){
	    cerr << "OOPS!\n";
	    cerr << "current: " << current[i] << '\n';
	    cerr << "counts: " << counts[i*2+1] << '\n';
	    exit(1);
	}
    }
    for(int i=0;i<total;i++){
	if(!grid[i]){
	    cerr << "OOPS: grid[" << i << "]==0!\n";
	    exit(1);
	}
    }
    cerr << "7/7 Verifying grid took " << Time::currentSeconds()-time << " seconds\n";
    cerr << "Done building grid\n";
}

void Grid::intersect(const Ray& ray, HitInfo& hit,
		    DepthStats* st, PerProcessorContext* ppc)
{
    const Vector dir(ray.direction());
    const Point orig(ray.origin());
    Point min(bbox.min());
    Point max(bbox.max());
    Vector diag(bbox.diagonal());
    double MIN, MAX;
    double inv_dir=1./dir.x();
    int didx_dx, dix_dx;
    int ddx;
    int nynz=ny*nz;
    if(dir.x() > 0){
	MIN=inv_dir*(min.x()-orig.x());
	MAX=inv_dir*(max.x()-orig.x());
	didx_dx=nynz;
	dix_dx=1;
	ddx=1;
    } else {
	MIN=inv_dir*(max.x()-orig.x());
	MAX=inv_dir*(min.x()-orig.x());
	didx_dx=-nynz;
	dix_dx=-1;
	ddx=0;
    }	
    double y0, y1;
    int didx_dy, diy_dy;
    int ddy;
    if(dir.y() > 0){
	double inv_dir=1./dir.y();
	y0=inv_dir*(min.y()-orig.y());
	y1=inv_dir*(max.y()-orig.y());
	didx_dy=nz;
	diy_dy=1;
	ddy=1;
    } else if(dir.y() <-1.e-6){
	double inv_dir=1./dir.y();
	y0=inv_dir*(max.y()-orig.y());
	y1=inv_dir*(min.y()-orig.y());
	didx_dy=-nz;
	diy_dy=-1;
	ddy=0;
    }
    if(y0>MIN)
	MIN=y0;
    if(y1<MAX)
	MAX=y1;
    if(MAX<MIN)
	return;
    
    double z0, z1;
    int didx_dz, diz_dz;
    int ddz;
    if(dir.z() > 0){
	double inv_dir=1./dir.z();
	z0=inv_dir*(min.z()-orig.z());
	z1=inv_dir*(max.z()-orig.z());
	didx_dz=1;
	diz_dz=1;
	ddz=1;
    } else {
	double inv_dir=1./dir.z();
	z0=inv_dir*(max.z()-orig.z());
	z1=inv_dir*(min.z()-orig.z());
	didx_dz=-1;
	diz_dz=-1;
	ddz=0;
    }
    if(z0>MIN)
	MIN=z0;
    if(z1<MAX)
	MAX=z1;
    if(MAX<MIN)
	return;
    double t;
    if(MIN > 1.e-6){
	t=MIN;
    } else if(MAX > 1.e-6){
	t=0;
    } else {
	return;
    }
    if(t>1.e29)
	return;
    Point p(orig+dir*t);
    Vector s((p-min)/diag);
    int ix=(int)(s.x()*nx);
    int iy=(int)(s.y()*ny);
    int iz=(int)(s.z()*nz);
    if(ix>=nx)
	ix--;
    if(iy>=ny)
	iy--;
    if(iz>=nz)
	iz--;
    if(ix<0)
	ix++;
    if(iy<0)
	iy++;
    if(iz<0)
	iz++;

    int idx=ix*nynz+iy*nz+iz;

    double next_x, next_y, next_z;
    double dtdx, dtdy, dtdz;
    double x=min.x()+diag.x()*double(ix+ddx)/nx;
    next_x=(x-orig.x())/dir.x();
    dtdx=dix_dx*diag.x()/nx/dir.x();
    double y=min.y()+diag.y()*double(iy+ddy)/ny;
    next_y=(y-orig.y())/dir.y();
    dtdy=diy_dy*diag.y()/ny/dir.y();
    double z=min.z()+diag.z()*double(iz+ddz)/nz;
    next_z=(z-orig.z())/dir.z();
    dtdz=diz_dz*diag.z()/nz/dir.z();
    for(;;){
	int n=counts[idx*2+1];
	int s=counts[idx*2];
	for(int i=0;i<n;i++){
	    grid[s+i]->intersect(ray, hit, st, ppc);
	}
	if(next_x < next_y && next_x < next_z){
	    // Step in x...
	    t=next_x;
	    next_x+=dtdx;
	    ix+=dix_dx;
	    idx+=didx_dx;
	    if(ix<0 || ix>=nx)
		return;
	} else if(next_y < next_z){
	    t=next_y;
	    next_y+=dtdy;
	    iy+=diy_dy;
	    idx+=didx_dy;
	    if(iy<0 || iy>=ny)
		return;
	} else {
	    t=next_z;
	    next_z+=dtdz;
	    iz+=diz_dz;
	    idx+=didx_dz;
	    if(iz<0 || iz >=nz)
		return;
	}
	if(hit.min_t < t)
	    break;
    }
}

void Grid::light_intersect(Light*, const Ray& lightray,
			  HitInfo& hit, double, Color&,
			  DepthStats* ds, PerProcessorContext* ppc)
{
    intersect(lightray, hit, ds, ppc);
}

void Grid::animate(double, bool&)
{
    //obj->animate(t);
}

void Grid::collect_prims(Array1<Object*>& prims)
{
    prims.add(this);
}

void Grid::compute_bounds(BBox& bbox, double offset)
{
    obj->compute_bounds(bbox, offset);
}

Vector Grid::normal(const Point&, const HitInfo&)
{
    cerr << "Error: Grid normal should not be called!\n";
    return Vector(0,0,0);
}
