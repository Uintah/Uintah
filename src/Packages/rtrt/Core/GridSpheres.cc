
#include <Packages/rtrt/Core/GridSpheres.h>

#include <Packages/rtrt/Core/GridSpheresDpy.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/visinfo/visinfo.h>

#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Runnable.h>

#include <iostream>

#include <GL/glx.h>
#include <GL/glu.h>
#include <X11/Xlib.h>

#include <stdlib.h>

using namespace rtrt;
using namespace SCIRun;
using namespace std;


//class Barrier;
//class Scene;
//class Stats;
//class Worker;

namespace rtrt {
  struct MCell {
    int nspheres;
    float* max;
    float* min;
  };
} // end namespace rtrt

GridSpheres::GridSpheres(float* spheres, float* /*inmin*/, float* /*inmax*/,
			 int nspheres, int ndata, int cellsize, int depth,
			 float radius, int nmatls, Material** matls,
			 string *var_names)
  : Object(this),
    spheres(spheres),
    nspheres(nspheres), ndata(ndata), cellsize(cellsize), depth(depth),
    radius(radius), nmatls(nmatls), var_names(var_names), preprocessed(false)
{
  counts=0;
  cells=0;
  min=new float[ndata+3];
  max=new float[ndata+3];
  // if minimum data copy it over
  // otherwise compute it manually
  //  if(inmin){
#if 0
  if(false){
    for(int i=0;i<ndata+3;i++){
      min[i]=inmin[i];
      max[i]=inmax[i];
    }
  } else {
#endif
    cerr << "Recomputing min/max for GridSpheres\n";
    float* p=spheres;
    for(int j = 0; j < ndata+3; j++){
      min[j]=MAXFLOAT;
      max[j]=-MAXFLOAT;
    }
    for(int i = 0; i < nspheres; i++){
      for(int j = 0; j < ndata+3; j++){
	min[j]=Min(min[j], p[j]);
	max[j]=Max(max[j], p[j]);
      }
      p+=ndata+3;
    }
#if 0
  }
#endif
  iradius=1./radius;
  this->matls=new Material*[nmatls];
  for(int i=0;i<nmatls;i++)
    this->matls[i]=matls[i];
  if (var_names != 0) {
    this->var_names=new string[ndata+3];
    for(int i=0;i<(ndata+3);i++)
      this->var_names[i]=var_names[i];
  }
}

GridSpheres::~GridSpheres()
{
}

void 
GridSpheres::io(SCIRun::Piostream &str)
{
  ASSERTFAIL("Pio for GridSpheres not implemented");
}

static inline void calc_se(float* p, const BBox& bbox,
			   const Vector& diag,
			   float radius,
			   int totalcells,
			   int& sx, int& sy, int& sz,
			   int& ex, int& ey, int& ez)
{
  Vector s((Point(p[0]-radius, p[1]-radius, p[2]-radius)-bbox.min())/diag);
  Vector e((Point(p[0]+radius, p[1]+radius, p[2]+radius)-bbox.min())/diag);
  sx=(int)(s.x()*totalcells);
  sy=(int)(s.y()*totalcells);
  sz=(int)(s.z()*totalcells);
  ex=(int)(e.x()*totalcells);
  ey=(int)(e.y()*totalcells);
  ez=(int)(e.z()*totalcells);
  if(sx < 0 || ex >= totalcells){
    cerr << "NX out of bounds!\n";
    cerr << "sx=" << sx << ", ex=" << ex << '\n';
    cerr << "e=" << e << '\n';
    cerr << "bbox=" << bbox.min() << ", " << bbox.max() << '\n';
    cerr << "diag=" << diag << '\n';
    exit(1);
  }
  if(sy < 0 || ey >= totalcells){
    cerr << "NY out of bounds!\n";
    cerr << "sy=" << sy << ", ey=" << ey << '\n';
    exit(1);
  }
  if(sz < 0 || ez >= totalcells){
    cerr << "NZ out of bounds!\n";
    cerr << "sz=" << sz << ", ez=" << ez << '\n';
    exit(1);
  }
}

int GridSpheres::map_idx(int ix, int iy, int iz, int depth)
{
  if(depth==0){
    return ix*cellsize*cellsize+iy*cellsize+iz;
  } else {
    int cellidx=map_idx(ix/cellsize, iy/cellsize, iz/cellsize, depth-1);
    int nx=ix%cellsize;
    int ny=iy%cellsize;
    int nz=iz%cellsize;
    return cellidx*cellsize*cellsize*cellsize+nx*cellsize*cellsize+ny*cellsize+nz;
  }
}

static void genmap(int depth, int idx, int cs, int*& map){
  idx*=cs*cs*cs;
  if(depth==0){
    for(int x=0;x<cs;x++){
      for(int y=0;y<cs;y++){
	for(int z=0;z<cs;z++){
	  *map++=idx++;
	}
      }
    }
  } else {
    for(int x=0;x<cs;x++){
      for(int y=0;y<cs;y++){
	for(int z=0;z<cs;z++){
	  genmap(depth-1, idx, cs, map);
	  idx++;
	}
      }
    }
  }
}

void GridSpheres::preprocess(double, int&, int&)
{
  if (preprocessed) return;
  preprocessed = true;
  cerr << "Building GridSpheres\n";
  float time=SCIRun::Time::currentSeconds();
  
  cerr << "min: " << min[0] << ", " << min[1] << ", " << min[2] << '\n';
  cerr << "max: " << max[0] << ", " << max[1] << ", " << max[2] << '\n';
  bbox.reset();
  bbox.extend(Point(min[0]-radius, min[1]-radius, min[2]-radius));
  bbox.extend(Point(max[0]+radius, max[1]+radius, max[2]+radius));
  bbox.extend(bbox.min()-Vector(1.e-3, 1.e-3, 1.e-3));
  bbox.extend(bbox.max()+Vector(1.e-3, 1.e-3, 1.e-3));
  Vector diag(bbox.diagonal());
  bbox.extend(bbox.max()+diag*1.e-3);
  bbox.extend(bbox.min()-diag*1.e-3);
  diag=bbox.diagonal();
  int totalcells=1;
  for(int i=0;i<=depth;i++)
    totalcells*=cellsize;
  totalsize=totalcells*totalcells*totalcells;
  cerr << "Computing " << totalcells << 'x' << totalcells << 'x' << totalcells << " grid for " << totalsize << " cells\n";
  
  
  if(counts)
    delete[] counts;
  if(cells)
    delete[] cells;
  counts=new int[2*totalsize];
  for(int i=0;i<totalsize*2;i++)
    counts[i]=0;
  cerr << "0/6 Allocation took " << SCIRun::Time::currentSeconds()-time << " seconds\n";
  time=SCIRun::Time::currentSeconds();
  
  int* map=new int[totalsize];
#if 0
  int* tmpmap=map;
  genmap(depth, 0, cellsize, tmpmap);
#else
  int idx=0;
  for(int x=0;x<totalcells;x++){
    for(int y=0;y<totalcells;y++){
      for(int z=0;z<totalcells;z++){
	map[idx++]=map_idx(x,y,z,depth);
      }
    }
  }
#endif
  cerr << "1/6 Generating map took " << SCIRun::Time::currentSeconds()-time << " seconds\n";
  time=SCIRun::Time::currentSeconds();
  
  double itime=time;
  float* p=spheres;
  int tc2=totalcells*totalcells;
  for(int i=0;i<nspheres;i++){
    double tnow=SCIRun::Time::currentSeconds();
    if(tnow-itime > 5.0){
      cerr << i << "/" << nspheres << '\n';
      itime=tnow;
    }
    int sx, sy, sz, ex, ey, ez;
    calc_se(p, bbox, diag, radius, totalcells, sx, sy, sz, ex, ey, ez);
    int idx_x=sx*tc2;
    for(int x=sx;x<=ex;x++){
      int idx_y=idx_x+sy*totalcells;
      idx_x+=tc2;
      for(int y=sy;y<=ey;y++){
	int idx=idx_y+sz;
	idx_y+=totalcells;
	for(int z=sz;z<=ez;z++){
	  int aidx=map[idx++];
	  counts[aidx*2+1]++;
	}
      }
    }
    //cerr << '\n';
    p+=3+ndata;
  }
  
  cerr << "2/6 Counting cells took " << SCIRun::Time::currentSeconds()-time << " seconds\n";
  time=SCIRun::Time::currentSeconds();
  int total=0;
  for(int i=0;i<totalsize;i++){
    int count=counts[i*2+1];
    counts[i*2]=total;
    total+=count;
  }
  cerr << "Allocating " << total << " grid cells (" << double(total)/nspheres << " per object, " << double(total)/totalsize << " per cell)\n";
  cells=new int[total];
  for(int i=0;i<total;i++)
    cells[i]=-1234;
  cerr << "3/6 Calculating offsets took " << SCIRun::Time::currentSeconds()-time << " seconds\n";
  time=SCIRun::Time::currentSeconds();
  itime=time;
  Array1<int> current(totalsize);
  current.initialize(0);
  p=spheres;
  for(int i=0;i<nspheres;i++){
    double tnow=SCIRun::Time::currentSeconds();
    if(tnow-itime > 5.0){
      cerr << i << "/" << nspheres << '\n';
      itime=tnow;
    }
    int sx, sy, sz, ex, ey, ez;
    calc_se(p, bbox, diag, radius, totalcells, sx, sy, sz, ex, ey, ez);
    for(int x=sx;x<=ex;x++){
      for(int y=sy;y<=ey;y++){
	int idx=x*totalcells*totalcells+y*totalcells+sz;
	for(int z=sz;z<=ez;z++){
	  int aidx=map[idx++];
	  int cur=current[aidx]++;
	  int pos=counts[aidx*2]+cur;
	  cells[pos]=i*(ndata+3);
	}
      }
    }
    p+=3+ndata;
  }
  cerr << "4/6 Filling grid took " << SCIRun::Time::currentSeconds()-time << " seconds\n";
  delete[] map;
  time=SCIRun::Time::currentSeconds();
  for(int i=0;i<totalsize;i++){
    if(current[i] != counts[i*2+1]){
      cerr << "OOPS!\n";
      cerr << "current: " << current[i] << '\n';
      cerr << "counts: " << counts[i*2+1] << '\n';
      exit(1);
    }
#if 0
    cerr << "cell " << i << ": ";
    for(int j=0;j<counts[i*2+1];j++){
      int idx=counts[i*2]+j;
      cerr << cells[idx] << ' ';
    }
    cerr << '\n';
#endif
  }
  for(int i=0;i<total;i++){
    if(cells[i]==-1234){
      cerr << "OOPS: cells[" << i << "]==-1234!\n";
      exit(1);
    }
  }
  cerr << "5/6 Verifying grid took " << SCIRun::Time::currentSeconds()-time << " seconds\n";
  time=SCIRun::Time::currentSeconds();
  if(depth==0){
    macrocells=0;
  } else {
    macrocells=new MCell*[depth+1];
    macrocells[0]=0;
    int size=cellsize*cellsize*cellsize;
    int n=ndata+3;
    for(int d=depth;d>=1;d--){
      MCell* p=macrocells[d]=new MCell[size];
      float* mm=new float[size*n*2];
      for(int i=0;i<size;i++){
	p->min=mm;
	mm+=n;
	p->max=mm;
	mm+=n;
	p++;
      }
      size*=cellsize*cellsize*cellsize;
    }
    MCell top;
    calc_mcell(depth, 0, top);
    if(top.nspheres != total){
      cerr << "Mcell went wrong!\n";
      cerr << "mcell: " << top.nspheres << '\n';
      cerr << "total: " << total << '\n';
    }
    cerr << "6/6 Calculating macrocells took " << SCIRun::Time::currentSeconds()-time << " seconds\n";
  }
  cerr << "Done building GridSpheres\n";
  icellsize=1./cellsize;
}

void GridSpheres::calc_mcell(int depth, int startidx, MCell& mcell)
{
  mcell.nspheres=0;
  int n=ndata+3;
  mcell.min=new float[n*2];
  mcell.max=mcell.min+n;
  for(int i=0;i<n;i++){
    mcell.min[i]=MAXFLOAT;
    mcell.max[i]=-MAXFLOAT;
  }
  int cellsize3=cellsize*cellsize*cellsize;
  if(depth==0){
    for(int i=0;i<cellsize3;i++){
      int idx=startidx+i;
      int nsph=counts[idx*2+1];
      mcell.nspheres+=nsph;
      int s=counts[idx*2];
      for(int j=0;j<nsph;j++){
	float* p=spheres+cells[s+j];
	for(int k=0;k<n;k++){
	  if(p[k]<mcell.min[k])
	    mcell.min[k]=p[k];
	  if(p[k]>mcell.max[k])
	    mcell.max[k]=p[k];
	}
      }
    }
  } else {
    MCell* mcells=macrocells[depth];
    for(int i=0;i<cellsize3;i++){
      int idx=startidx+i;
      calc_mcell(depth-1, idx*cellsize*cellsize*cellsize, mcells[idx]);
      mcell.nspheres+=mcells[idx].nspheres;
      for(int j=0;j<n;j++){
	if(mcells[idx].min[j] < mcell.min[j])
	  mcell.min[j]=mcells[idx].min[j];
	if(mcells[idx].max[j] > mcell.max[j])
	  mcell.max[j]=mcells[idx].max[j];
      }
    }
  }
}

#if 0
void GridSpheres::intersect(Ray& ray, HitInfo& hit,
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
  } else {
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
  if(iy<1)
    iy++;
  if(iz<1)
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
    int j=0;
    int n=ndata+3;
    if(mcell->nspheres==0)
      j=n;
    for(;j<n;j++){
      if(mcell->max[j]<range_begin[j])
	break;
      if(mcell->min[j]>range_end[j])
	break;
    }
    if(j==n){
      for(;;){
	int nsph=counts[idx*2+1];
	int s=counts[idx*2];
	st->sphere_isect+=n;
	for(int i=0;i<nsph;i++){
	  float* p=spheres+cells[s+i];
	  int j;
	  for(j=0;j<n;j++){
	    if(p[j]<range_begin[j])
	      break;
	    if(p[j]>range_end[j])
	      break;
	  }
	  if(j==n){
	    Vector OC=Point(p[0], p[1], p[2])-ray.origin();
	    double tca=Dot(OC, ray.direction());
	    double l2oc=OC.length2();
	    double rad2=radius*radius;
	    if(l2oc <= rad2){
	      // Inside the sphere
	      double t2hc=rad2-l2oc+tca*tca;
	      double thc=sqrt(t2hc);
	      double t=tca+thc;
	      if(hit.hit(this, t)){
		int* cell=(int*)hit.scratchpad;
		*cell=cells[s+i];
	      }
	      st->sphere_hit++;
	    } else {
	      if(tca < 0.0){
				// Behind ray, no intersections...
	      } else {
		double t2hc=rad2-l2oc+tca*tca;
		if(t2hc <= 0.0){
		  // Ray misses, no intersections
		} else {
		  double thc=sqrt(t2hc);
		  if(hit.hit(this, tca-thc)){
		    int* cell=(int*)hit.scratchpad;
		    *cell=cells[s+i];
		  }
		}
	      }
	    }
	  }
	}
	if(next_x < next_y && next_x < next_z){
	  // Step in x...
	  t=next_x;
	  next_x+=dtdx;
	  ix+=dix_dx;
	  idx+=didx_dx;
	  if(ix<mcellminx || ix>=mcellmaxx)
	    return;
	} else if(next_y < next_z){
	  t=next_y;
	  next_y+=dtdy;
	  iy+=diy_dy;
	  idx+=didx_dy;
	  if(iy<mcellminu || iy>=mcellmaxy)
	    return;
	} else {
	  t=next_z;
	  next_z+=dtdz;
	  iz+=diz_dz;
	  idx+=didx_dz;
	  if(iz<mcellminz || iz >=mcellmaxz)
	    return;
	}
	if(hit.min_t < t)
	  return;
      } else {
	
      }
    }
  }
}
#endif

#if 1
void GridSpheres::intersect(Ray& ray, HitInfo& hit,
			    DepthStats* st, PerProcessorContext* ppc)
{
  const Vector dir(ray.direction());
  const Point orig(ray.origin());
  //cerr << "orig: " << orig << '\n';
  //cerr << "dir: " << dir << '\n';
  Point min(bbox.min());
  Point max(bbox.max());
  //cerr << "min: " << min << ", max: " << max << '\n';
  Vector diag(bbox.diagonal());
  double MIN, MAX;
  double inv_dir=1./dir.x();
  int didx_dx, dix_dx;
  int ddx;
  if(dir.x() > 0){
    MIN=inv_dir*(min.x()-orig.x());
    MAX=inv_dir*(max.x()-orig.x());
    didx_dx=cellsize*cellsize;
    dix_dx=1;
    ddx=1;
  } else {
    MIN=inv_dir*(max.x()-orig.x());
    MAX=inv_dir*(min.x()-orig.x());
    didx_dx=-cellsize*cellsize;
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
    didx_dy=cellsize;
    diy_dy=1;
    ddy=1;
  } else {
    double inv_dir=1./dir.y();
    y0=inv_dir*(max.y()-orig.y());
    y1=inv_dir*(min.y()-orig.y());
    didx_dy=-cellsize;
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
  int ix=(int)(s.x()*cellsize);
  int iy=(int)(s.y()*cellsize);
  int iz=(int)(s.z()*cellsize);
  if(ix>=cellsize)
    ix--;
  if(iy>=cellsize)
    iy--;
  if(iz>=cellsize)
    iz--;
  if(ix<0)
    ix++;
  if(iy<0)
    iy++;
  if(iz<0)
    iz++;
  
  int idx=ix*cellsize*cellsize+iy*cellsize+iz;
  
  double next_x, next_y, next_z;
  double dtdx, dtdy, dtdz;
  double x=min.x()+diag.x()*double(ix+ddx)/cellsize;
  next_x=(x-orig.x())/dir.x();
  dtdx=dix_dx*diag.x()/cellsize/dir.x();
  double y=min.y()+diag.y()*double(iy+ddy)/cellsize;
  next_y=(y-orig.y())/dir.y();
  dtdy=diy_dy*diag.y()/cellsize/dir.y();
  double z=min.z()+diag.z()*double(iz+ddz)/cellsize;
  next_z=(z-orig.z())/dir.z();
  dtdz=diz_dz*diag.z()/cellsize/dir.z();
  
  Vector cellcorner((orig-min)/(diag)*cellsize);
  Vector celldir(dir/(diag)*cellsize);
  
  isect(depth, t, dtdx, dtdy, dtdz, next_x, next_y, next_z,
	idx, ix, iy, iz, dix_dx, diy_dy, diz_dz,
	didx_dx, didx_dy, didx_dz,
	cellcorner, celldir,
	ray, hit, st, ppc);
}

#else

void GridSpheres::intersect_print(Ray& ray, HitInfo& hit,
				  DepthStats* st, PerProcessorContext* ppc)
{
  int totalcells=1;
  for(int i=0;i<=depth;i++)
    totalcells*=cellsize;
  const Vector dir(ray.direction());
  const Point orig(ray.origin());
  Point min(bbox.min());
  Point max(bbox.max());
  Vector diag(bbox.diagonal());
  double MIN, MAX;
  double inv_dir=1./dir.x();
  int dix_dx;
  int ddx;
  if(dir.x() > 0){
    MIN=inv_dir*(min.x()-orig.x());
    MAX=inv_dir*(max.x()-orig.x());
    dix_dx=1;
    ddx=1;
  } else {
    MIN=inv_dir*(max.x()-orig.x());
    MAX=inv_dir*(min.x()-orig.x());
    dix_dx=-1;
    ddx=0;
  }	
  double y0, y1;
  int diy_dy;
  int ddy;
  if(dir.y() > 0){
    double inv_dir=1./dir.y();
    y0=inv_dir*(min.y()-orig.y());
    y1=inv_dir*(max.y()-orig.y());
    diy_dy=1;
    ddy=1;
  } else {
    double inv_dir=1./dir.y();
    y0=inv_dir*(max.y()-orig.y());
    y1=inv_dir*(min.y()-orig.y());
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
  int diz_dz;
  int ddz;
  if(dir.z() > 0){
    double inv_dir=1./dir.z();
    z0=inv_dir*(min.z()-orig.z());
    z1=inv_dir*(max.z()-orig.z());
    diz_dz=1;
    ddz=1;
  } else {
    double inv_dir=1./dir.z();
    z0=inv_dir*(max.z()-orig.z());
    z1=inv_dir*(min.z()-orig.z());
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
  int ix=(int)(s.x()*totalcells);
  int iy=(int)(s.y()*totalcells);
  int iz=(int)(s.z()*totalcells);
  if(ix>=totalcells)
    ix--;
  if(iy>=totalcells)
    iy--;
  if(iz>=totalcells)
    iz--;
  if(ix<0)
    ix++;
  if(iy<0)
    iy++;
  if(iz<0)
    iz++;
  //cerr << "Start: ix=" << ix << ", iy=" << iy << ", iz=" << iz << '\n';
  //cerr << "t=" << t << ", p=" << p << '\n';
  //cerr << "bbox: " << min << ", " << max << '\n';
  
  double next_x, next_y, next_z;
  double dtdx, dtdy, dtdz;
  double x=min.x()+diag.x()*double(ix+ddx)/totalcells;
  next_x=(x-orig.x())/dir.x();
  dtdx=dix_dx*diag.x()/totalcells/dir.x();
  double y=min.y()+diag.y()*double(iy+ddy)/totalcells;
  next_y=(y-orig.y())/dir.y();
  dtdy=diy_dy*diag.y()/totalcells/dir.y();
  double z=min.z()+diag.z()*double(iz+ddz)/totalcells;
  next_z=(z-orig.z())/dir.z();
  dtdz=diz_dz*diag.z()/totalcells/dir.z();
  int n=ndata+3;
  for(;;){
    //cerr << "ix=" << ix << ", iy=" << iy << ", iz=" << iz;
    int aidx=map_idx(ix, iy, iz, depth);
    //cerr << " aidx=" << aidx << ": ";
    int nsph=counts[aidx*2+1];
    int s=counts[aidx*2];
    st->sphere_isect+=nsph;
    for(int i=0;i<nsph;i++){
      float* p=spheres+cells[s+i];
      //cerr << cells[s+i] << ' ';
      int j;
      for(j=0;j<n;j++){
	if(p[j]<range_begin[j])
	  break;
	if(p[j]>range_end[j])
	  break;
      }
      if(j==n){
	Vector OC=Point(p[0], p[1], p[2])-ray.origin();
	double tca=Dot(OC, ray.direction());
	double l2oc=OC.length2();
	double rad2=radius*radius;
	if(l2oc <= rad2){
	  // Inside the sphere
	  double t2hc=rad2-l2oc+tca*tca;
	  double thc=sqrt(t2hc);
	  double t=tca+thc;
	  if(hit.hit(this, t)){
	    int* cell=(int*)hit.scratchpad;
	    *cell=cells[s+i];
	  }
	  st->sphere_hit++;
	} else {
	  if(tca < 0.0){
	    // Behind ray, no intersections...
	  } else {
	    double t2hc=rad2-l2oc+tca*tca;
	    if(t2hc <= 0.0){
	      // Ray misses, no intersections
	    } else {
	      double thc=sqrt(t2hc);
	      if(hit.hit(this, tca-thc)){
		int* cell=(int*)hit.scratchpad;
		*cell=cells[s+i];
	      }
	    }
	  }
	}
      } else {
	cerr << "BAILED\n";
	for(j=0;j<n;j++){
	  cerr << j << ": " << p[j] << ", " << range_begin[j] << ", " << range_end[j] << '\n';
	}
      }
    }
    //cerr << '\n';
    if(next_x < next_y && next_x < next_z){
      // Step in x...
      t=next_x;
      next_x+=dtdx;
      ix+=dix_dx;
      if(ix<0 || ix>=totalcells)
	return;
    } else if(next_y < next_z){
      t=next_y;
      next_y+=dtdy;
      iy+=diy_dy;
      if(iy<0 || iy>=totalcells)
	return;
    } else {
      t=next_z;
      next_z+=dtdz;
      iz+=diz_dz;
      if(iz<0 || iz >=totalcells)
	return;
    }
    if(hit.min_t < t)
      return;
  }
}

void GridSpheres::intersect(Ray& ray, HitInfo& hit,
			    DepthStats* st, PerProcessorContext* ppc)
{
  int totalcells=1;
  for(int i=0;i<=depth;i++)
    totalcells*=cellsize;
  const Vector dir(ray.direction());
  const Point orig(ray.origin());
  Point min(bbox.min());
  Point max(bbox.max());
  Vector diag(bbox.diagonal());
  double MIN, MAX;
  double inv_dir=1./dir.x();
  int dix_dx;
  int ddx;
  if(dir.x() > 0){
    MIN=inv_dir*(min.x()-orig.x());
    MAX=inv_dir*(max.x()-orig.x());
    dix_dx=1;
    ddx=1;
  } else {
    MIN=inv_dir*(max.x()-orig.x());
    MAX=inv_dir*(min.x()-orig.x());
    dix_dx=-1;
    ddx=0;
  }	
  double y0, y1;
  int diy_dy;
  int ddy;
  if(dir.y() > 0){
    double inv_dir=1./dir.y();
    y0=inv_dir*(min.y()-orig.y());
    y1=inv_dir*(max.y()-orig.y());
    diy_dy=1;
    ddy=1;
  } else {
    double inv_dir=1./dir.y();
    y0=inv_dir*(max.y()-orig.y());
    y1=inv_dir*(min.y()-orig.y());
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
  int diz_dz;
  int ddz;
  if(dir.z() > 0){
    double inv_dir=1./dir.z();
    z0=inv_dir*(min.z()-orig.z());
    z1=inv_dir*(max.z()-orig.z());
    diz_dz=1;
    ddz=1;
  } else {
    double inv_dir=1./dir.z();
    z0=inv_dir*(max.z()-orig.z());
    z1=inv_dir*(min.z()-orig.z());
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
  int ix=(int)(s.x()*totalcells);
  int iy=(int)(s.y()*totalcells);
  int iz=(int)(s.z()*totalcells);
  if(ix>=totalcells)
    ix--;
  if(iy>=totalcells)
    iy--;
  if(iz>=totalcells)
    iz--;
  if(ix<0)
    ix++;
  if(iy<0)
    iy++;
  if(iz<0)
    iz++;
  
  double next_x, next_y, next_z;
  double dtdx, dtdy, dtdz;
  double x=min.x()+diag.x()*double(ix+ddx)/totalcells;
  next_x=(x-orig.x())/dir.x();
  dtdx=dix_dx*diag.x()/totalcells/dir.x();
  double y=min.y()+diag.y()*double(iy+ddy)/totalcells;
  next_y=(y-orig.y())/dir.y();
  dtdy=diy_dy*diag.y()/totalcells/dir.y();
  double z=min.z()+diag.z()*double(iz+ddz)/totalcells;
  next_z=(z-orig.z())/dir.z();
  dtdz=diz_dz*diag.z()/totalcells/dir.z();
  int n=ndata+3;
  for(;;){
    int aidx=map_idx(ix, iy, iz, depth);
    int nsph=counts[aidx*2+1];
    int s=counts[aidx*2];
    st->sphere_isect+=nsph;
    for(int i=0;i<nsph;i++){
      float* p=spheres+cells[s+i];
      int j;
      for(j=0;j<n;j++){
	if(p[j]<range_begin[j])
	  break;
	if(p[j]>range_end[j])
	  break;
      }
      if(j==n){
	Vector OC=Point(p[0], p[1], p[2])-ray.origin();
	double tca=Dot(OC, ray.direction());
	double l2oc=OC.length2();
	double rad2=radius*radius;
	if(l2oc <= rad2){
	  // Inside the sphere
	  double t2hc=rad2-l2oc+tca*tca;
	  double thc=sqrt(t2hc);
	  double t=tca+thc;
	  if(hit.hit(this, t)){
	    int* cell=(int*)hit.scratchpad;
	    *cell=cells[s+i];
	  }
	  st->sphere_hit++;
	} else {
	  if(tca < 0.0){
	    // Behind ray, no intersections...
	  } else {
	    double t2hc=rad2-l2oc+tca*tca;
	    if(t2hc <= 0.0){
	      // Ray misses, no intersections
	    } else {
	      double thc=sqrt(t2hc);
	      if(hit.hit(this, tca-thc)){
		int* cell=(int*)hit.scratchpad;
		*cell=cells[s+i];
	      }
	    }
	  }
	}
      }
    }
    if(next_x < next_y && next_x < next_z){
      // Step in x...
      t=next_x;
      next_x+=dtdx;
      ix+=dix_dx;
      if(ix<0 || ix>=totalcells)
	break;
    } else if(next_y < next_z){
      t=next_y;
      next_y+=dtdy;
      iy+=diy_dy;
      if(iy<0 || iy>=totalcells)
	break;
    } else {
      t=next_z;
      next_z+=dtdz;
      iz+=diz_dz;
      if(iz<0 || iz >=totalcells)
	break;
    }
    if(hit.min_t < t)
      break;
  }
#if 0
  {
    float* p=spheres;
    double old_min=hit.min_t;
    int min_i=-1234;
    for(int i=0;i<nspheres;i++){
      Vector OC=Point(p[0], p[1], p[2])-ray.origin();
      double tca=Dot(OC, ray.direction());
      double l2oc=OC.length2();
      double rad2=radius*radius;
      if(l2oc <= rad2){
	// Inside the sphere
	double t2hc=rad2-l2oc+tca*tca;
	double thc=sqrt(t2hc);
	double t=tca+thc;
	if(hit.hit(this, t)){
	  int* cell=(int*)hit.scratchpad;
	  *cell=i*3;
	  min_i=i;
	}
	st->sphere_hit++;
      } else {
	if(tca < 0.0){
	  // Behind ray, no intersections...
	} else {
	  double t2hc=rad2-l2oc+tca*tca;
	  if(t2hc <= 0.0){
	    // Ray misses, no intersections
	  } else {
	    double thc=sqrt(t2hc);
	    if(hit.hit(this, tca-thc)){
	      int* cell=(int*)hit.scratchpad;
	      *cell=i*3;
	      min_i=i;
	    }
	  }
	}
      }
      p+=3+ndata;
    }
    if(hit.min_t != old_min){
      cerr << "OLD: " << old_min << '\n';
      cerr << "NEW: " << hit.min_t << '\n';
      cerr << "Sphere: " << min_i*3 << '\n';
      intersect_print(ray, hit, st, ppc);
    }
  }
#endif
}	

#endif

#if 0
static void compare(int depth, char* tag, const Vector& have, const Vector& want)
{
  if(have != want){
    //cerr << depth << " - " << tag << ", have: " << have << ", want: " << want << '\n';
  }
}

static void compare(int depth, char* tag, int have, int want)
{
  if(have != want){
    //cerr << depth << " - " << tag << ", have: " << have << ", want: " << want << '\n';
  }
}

static void compare(int depth, char* tag, double have, double want)
{
  if(have != want){
    //cerr << depth << " - " << tag << ", have: " << have << ", want: " << want << '\n';
  }
}

static void compare(int depth, char* tag, float have, float want)
{
  if(have != want){
    //cerr << depth << " - " << tag << ", have: " << have << ", want: " << want << '\n';
  }
}
#endif

void GridSpheres::isect(int depth, double t,
			double dtdx, double dtdy, double dtdz,
			double next_x, double next_y, double next_z,
			int idx, int ix, int iy, int iz,
			int dix_dx, int diy_dy, int diz_dz,
			int didx_dx, int didx_dy, int didx_dz,
			const Vector& cellcorner, const Vector& celldir,
			const Ray& ray, HitInfo& hit,
			DepthStats* st, PerProcessorContext* ppc)
{
  //cerr << "Starting depth " << depth << '\n';
  int n=ndata+3;
  if(depth==0){
    for(;;){
#if 1
      if(didx_dy<-cellsize || didx_dy>cellsize){
	cerr << "didx_dy corrupt: " << didx_dy << '\n';
      }
#endif
      int nsph=counts[idx*2+1];
      //cerr << "t=" << t << ": " << ix << ", " << iy << ", " << iz << ", " << next_x << ", " << next_y << ", " << next_z << ", " << idx << "(" << nsph << " spheres)\n";
      int s=counts[idx*2];
      for(int i=0;i<nsph;i++){
	float* p=spheres+cells[s+i];
	int j;
	for(j=0;j<n;j++){
	  if(p[j] < dpy->range_begin[j])
	    break;
	  if(p[j] > dpy->range_end[j])
	    break;
	}
	if(j==n){
	  st->sphere_isect++;
	  Vector OC=Point(p[0], p[1], p[2])-ray.origin();
	  double tca=Dot(OC, ray.direction());
	  double l2oc=OC.length2();
	  double rad2=radius*radius;
	  if(l2oc <= rad2){
	    // Inside the sphere
	    double t2hc=rad2-l2oc+tca*tca;
	    double thc=sqrt(t2hc);
	    double t=tca+thc;
	    if(hit.hit(this, t)){
	      int* cell=(int*)hit.scratchpad;
	      *cell=cells[s+i];
	    }
	    st->sphere_hit++;
	  } else {
	    if(tca < 0.0){
	      // Behind ray, no intersections...
	    } else {
	      double t2hc=rad2-l2oc+tca*tca;
	      if(t2hc <= 0.0){
				// Ray misses, no intersections
	      } else {
		double thc=sqrt(t2hc);
		if(hit.hit(this, tca-thc)){
		  int* cell=(int*)hit.scratchpad;
		  *cell=cells[s+i];
		}
	      }
	    }
	  }
	} else {
	  //cerr << "Bailed on sphere\n";
	}
      }
      if(next_x < next_y && next_x < next_z){
	// Step in x...
	t=next_x;
	next_x+=dtdx;
	ix+=dix_dx;
	idx+=didx_dx;
	if(ix<0 || ix>=cellsize)
	  break;
      } else if(next_y < next_z){
	t=next_y;
	next_y+=dtdy;
	iy+=diy_dy;
	idx+=didx_dy;
	if(iy<0 || iy>=cellsize)
	  break;
      } else {
	t=next_z;
	next_z+=dtdz;
	iz+=diz_dz;
	idx+=didx_dz;
	if(iz<0 || iz >=cellsize)
	  break;
      }
      if(hit.min_t < t)
	break;
    }
  } else {
    MCell* mcells=macrocells[depth];
    for(;;){
      //cerr << "t=" << t << ": " << ix << ", " << iy << ", " << iz << ", " << next_x << ", " << next_y << ", " << next_z << ", " << idx << '\n';
      MCell& mcell=mcells[idx];
      if(mcell.nspheres!=0){
	int j;
	for(j=0;j<n;j++){
	  if(mcell.max[j] < dpy->range_begin[j])
	    break;
	  if(mcell.min[j] > dpy->range_end[j])
	    break;
	}
	if(j==n){
	  // Do this cell...
	  int new_ix=(int)((cellcorner.x()+t*celldir.x()-ix)*cellsize);
	  int new_iy=(int)((cellcorner.y()+t*celldir.y()-iy)*cellsize);
	  int new_iz=(int)((cellcorner.z()+t*celldir.z()-iz)*cellsize);
	  //cerr << "cc+t*cd=" << cellcorner+celldir*t << '\n';
	  //cerr << "new_ix=" << new_ix << ", new_iy=" << new_iy << ", new_iz=" << new_iz << '\n';
	  if(new_ix<0)
	    new_ix=0;
	  else if(new_ix>=cellsize)
	    new_ix=cellsize-1;
	  if(new_iy<0)
	    new_iy=0;
	  else if(new_iy>=cellsize)
	    new_iy=cellsize-1;
	  if(new_iz<0)
	    new_iz=0;
	  else if(new_iz>=cellsize)
	    new_iz=cellsize-1;
	  int new_idx=(((idx*cellsize+new_ix)*cellsize+new_iy)*cellsize)+new_iz;
	  double new_dtdx=dtdx*icellsize;
	  double new_dtdy=dtdy*icellsize;
	  double new_dtdz=dtdz*icellsize;
	  const Vector dir(ray.direction());
	  double new_next_x;
	  if(dir.x() > 0){
	    new_next_x=next_x-dtdx+new_dtdx*(new_ix+1);
	  } else {
	    new_next_x=next_x-new_ix*new_dtdx;
	  }
	  double new_next_y;
	  if(dir.y() > 0){
	    new_next_y=next_y-dtdy+new_dtdy*(new_iy+1);
	  } else {
	    new_next_y=next_y-new_iy*new_dtdy;
	  }
	  double new_next_z;
	  if(dir.z() > 0){
	    new_next_z=next_z-dtdz+new_dtdz*(new_iz+1);
	  } else {
	    new_next_z=next_z-new_iz*new_dtdz;
	  }
	  isect(depth-1, t,
		new_dtdx, new_dtdy, new_dtdz,
		new_next_x, new_next_y, new_next_z,
		new_idx, new_ix, new_iy, new_iz,
		dix_dx, diy_dy, diz_dz, didx_dx, didx_dy, didx_dz,
		(cellcorner-Vector(ix, iy, iz))*cellsize, celldir*cellsize,
		ray, hit, st, ppc);
	}
      }
      if(next_x < next_y && next_x < next_z){
	// Step in x...
	t=next_x;
	next_x+=dtdx;
	ix+=dix_dx;
	idx+=didx_dx;
	if(ix<0 || ix>=cellsize)
	  break;
      } else if(next_y < next_z){
	t=next_y;
	next_y+=dtdy;
	iy+=diy_dy;
	idx+=didx_dy;
	if(iy<0 || iy>=cellsize)
	  break;
      } else {
	t=next_z;
	next_z+=dtdz;
	iz+=diz_dz;
	idx+=didx_dz;
	if(iz<0 || iz >=cellsize)
	  break;
      }
      if(hit.min_t < t)
	break;
    }
  }
  //cerr << "Finished depth " << depth << '\n';
}

void GridSpheres::animate(double, bool& changed)
{
  dpy->animate(changed);
}

void GridSpheres::collect_prims(Array1<Object*>& prims)
{
  prims.add(this);
}

void GridSpheres::compute_bounds(BBox& bbox, double offset)
{
  bbox.extend(Point(min[0]-radius-offset, min[1]-radius-offset,
		    min[2]-radius-offset));
  bbox.extend(Point(max[0]+radius+offset, max[1]+radius+offset,
		    max[2]+radius+offset));
}

Vector GridSpheres::normal(const Point& hitpos, const HitInfo& hit)
{
  int cell=*(int*)hit.scratchpad;
  float* p=spheres+cell;
  Vector n=hitpos-Point(p[0], p[1], p[2]);
  n*=1./radius;
  return n;    
}

void GridSpheres::shade(Color& result, const Ray& ray,
			const HitInfo& hit, int depth,
			double atten, const Color& accumcolor,
			Context* cx)
{
  int cell=*(int*)hit.scratchpad;
  int colordata=dpy->colordata;
  float* p=spheres+cell+colordata;
  float min=dpy->color_begin[colordata];
  float scale=dpy->color_scales[colordata];
  float data=*p;
  float normalized=(data-min)*scale;
  int idx=(int)(normalized*(nmatls-1));
  if(idx<0)
    idx=0;
  if(idx>=nmatls)
    idx=nmatls-1;
  matls[idx]->shade(result, ray, hit, depth, atten, accumcolor, cx);
}

