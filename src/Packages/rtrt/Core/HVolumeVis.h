
#ifndef HVOLUMEVIS_H
#define HVOLUMEVIS_H 1

#include <Packages/rtrt/Core/BrickArray3.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/VolumeVis.h>
#include <Packages/rtrt/Core/VolumeVisDpy.h>

#include <Core/Geometry/Point.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/WorkQueue.h>

#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <unistd.h>
#include <iostream>
#include <fcntl.h>

namespace rtrt {

using SCIRun::Mutex;
using SCIRun::WorkQueue;

template<class DataT>
class VMCell {
public:
  // Need to make sure we have a 64 bit thing
  unsigned long long course_hash;
  // The max of unsigned long long is ULLONG_MAX
  VMCell(): course_hash(0) {}

  void turn_on_bits(DataT min, DataT max, DataT data_min, DataT data_max) {
    // We know that we have 64 bits, so figure out where min and max map
    // into [0..63].
    int min_index = (int)(min-data_min/(data_max-data_min)*63);
    int max_index = (int)(max-data_min/(data_max-data_min)*63);
    for (int i = min_index; i < max_index; i++)
      course_hash |= 1 << i;
  }
  inline VMCell<DataT>& operator |= (const VMCell<DataT>& v) {
    course_hash |= v.course_hash;
    return *this;
  }
};

template<class DataT, class MetaCT>
class HVolumeVis: public VolumeVis<DataT> {
protected:
public:
  Vector datadiag;
  Vector sdiag;
  Vector hierdiag;
  Vector ihierdiag;
  int depth;
  int* xsize;
  int* ysize;
  int* zsize;
  double* ixsize;
  double* iysize;
  double* izsize;
  BrickArray3<MetaCT>* macrocells;
  WorkQueue* work;

  void parallel_calc_mcell(int);
  void calc_mcell(int depth, int ix, int iy, int iz, MetaCT& mcell);
#if 0
  void isect(int depth, float isoval, double t,
	     double dtdx, double dtdy, double dtdz,
	     double next_x, double next_y, double next_z,
	     int ix, int iy, int iz,
	     int dix_dx, int diy_dy, int diz_dz,
	     int startx, int starty, int startz,
	     const Vector& cellcorner, const Vector& celldir,
	     const Ray& ray, HitInfo& hit,
	     DepthStats* st, PerProcessorContext* ppc);
#endif
  HVolumeVis(BrickArray3<DataT>& data, DataT data_min, DataT data_max,
	     int depth, Point min, Point max, VolumeVisDpy *dpy,
	     double spec_coeff, double ambient,
	     double diffuse, double specular, int np);
  virtual ~HVolumeVis();
  
  ///////////////////////////////////////
  // From Object
  virtual void print(ostream& out);

  ///////////////////////////////////////
  // From Material
  virtual void shade(Color& result, const Ray& ray,
		     const HitInfo& hit, int depth,
		     double atten, const Color& accumcolor,
		     Context* cx);

};
  

  /////////////////////////////////////////////////
  /////////////////////////////////////////////////
  // C code
  /////////////////////////////////////////////////
  /////////////////////////////////////////////////
  

extern Mutex io_lock_;
  
template<class DataT, class MetaCT>
HVolumeVis<DataT,MetaCT>::HVolumeVis(BrickArray3<DataT>& data,
				     DataT data_min, DataT data_max,
				     int depth, Point min, Point max,
				     VolumeVisDpy *dpy,
				     double spec_coeff, double ambient,
				     double diffuse, double specular, int np)
  : VolumeVis<DataT>(data, data_min, data_max,
		     data.dim1(), data.dim2(), data.dim3(),
		     min, max, spec_coeff, ambient, diffuse, specular, dpy),
    depth(depth)
{
  if(depth<=0)
    depth=1;
  
  datadiag=max-min;
  sdiag=datadiag/Vector(nx-1,ny-1,nz-1);
  
  // Compute all the grid stuff
  xsize=new int[depth];
  ysize=new int[depth];
  zsize=new int[depth];
  int tx=nx-1;
  int ty=ny-1;
  int tz=nz-1;
  for(int i=depth-1;i>=0;i--){
    int nx=(int)(pow(tx, 1./(i+1))+.9);
    tx=(tx+nx-1)/nx;
    xsize[depth-i-1]=nx;
    int ny=(int)(pow(ty, 1./(i+1))+.9);
    ty=(ty+ny-1)/ny;
    ysize[depth-i-1]=ny;
    int nz=(int)(pow(tz, 1./(i+1))+.9);
    tz=(tz+nz-1)/nz;
    zsize[depth-i-1]=nz;
  }
  ixsize=new double[depth];
  iysize=new double[depth];
  izsize=new double[depth];
  cerr << "Calculating depths...\n";
  for(int i=0;i<depth;i++){
    cerr << "xsize=" << xsize[i] << ", ysize=" << ysize[i] << ", zsize=" << zsize[i] << '\n';
    ixsize[i]=1./xsize[i];
    iysize[i]=1./ysize[i];
    izsize[i]=1./zsize[i];
  }
  cerr << "X: ";
  tx=1;
  for(int i=depth-1;i>=0;i--){
    cerr << xsize[i] << ' ';
    tx*=xsize[i];
  }
  cerr << "(" << tx << ")\n";
  if(tx<nx-1){
    cerr << "TX TOO SMALL!\n";
    exit(1);
  }
  cerr << "Y: ";
  ty=1;
  for(int i=depth-1;i>=0;i--){
    cerr << ysize[i] << ' ';
    ty*=ysize[i];
  }
  cerr << "(" << ty << ")\n";
  if(ty<ny-1){
    cerr << "TY TOO SMALL!\n";
    exit(1);
  }
  cerr << "Z: ";
  tz=1;
  for(int i=depth-1;i>=0;i--){
    cerr << zsize[i] << ' ';
    tz*=zsize[i];
  }
  cerr << "(" << tz << ")\n";
  if(tz<nz-1){
    cerr << "TZ TOO SMALL!\n";
    exit(1);
  }
  hierdiag=datadiag*Vector(tx,ty,tz)/Vector(nx-1,ny-1,nz-1);
  ihierdiag=Vector(1.,1.,1.)/hierdiag;
  
  if(depth==1){
    macrocells=0;
  } else {
    macrocells=new BrickArray3<MetaCT>[depth+1];
    int xs=1;
    int ys=1;
    int zs=1;
    for(int d=depth-1;d>=1;d--){
      xs*=xsize[d];
      ys*=ysize[d];
      zs*=zsize[d];
      macrocells[d].resize(xs, ys, zs);
      cerr << "Depth " << d << ": " << xs << "x" << ys << "x" << zs << '\n';
    }
    cerr << "Building hierarchy\n";
#if 0
    MetaCT top;
    calc_mcell(depth-1, 0, 0, 0, top);
    cerr << "Min: " << top.min << ", Max: " << top.max << '\n';
#else
    int nx=xsize[depth-1];
    int ny=ysize[depth-1];
    int nz=zsize[depth-1];
    int totaltop=nx*ny*nz;
    work=new WorkQueue("Building hierarchy");
    work->refill(totaltop, np, 5);
    SCIRun::Parallel<HVolumeVis<DataT,MetaCT> > phelper(this, &HVolumeVis<DataT,MetaCT>::parallel_calc_mcell);
    SCIRun::Thread::parallel(phelper, np, true);
    delete work;
#endif
    cerr << "done\n";
  }
}

template<class DataT, class MetaCT>
HVolumeVis<DataT,MetaCT>::~HVolumeVis()
{
}

template<class DataT, class MetaCT>
void HVolumeVis<DataT,MetaCT>::calc_mcell(int depth, int startx, int starty,
					  int startz, MetaCT& mcell)
{
  int endx=startx+xsize[depth];
  int endy=starty+ysize[depth];
  int endz=startz+zsize[depth];
  if(endx>nx-1)
    endx=nx-1;
  if(endy>ny-1)
    endy=ny-1;
  if(endz>nz-1)
    endz=nz-1;
  if(startx>=endx || starty>=endy || startz>=endz){
    /* This cell won't get used... */
    return;
  }
  if(depth==0){
    // We are at the data level.  Loop over each voxel and compute the
    // mcell for this group of voxels.
    for(int ix=startx;ix<endx;ix++){
      for(int iy=starty;iy<endy;iy++){
	for(int iz=startz;iz<endz;iz++){
	  DataT rhos[8];
	  rhos[0]=data(ix, iy, iz);
	  rhos[1]=data(ix, iy, iz+1);
	  rhos[2]=data(ix, iy+1, iz);
	  rhos[3]=data(ix, iy+1, iz+1);
	  rhos[4]=data(ix+1, iy, iz);
	  rhos[5]=data(ix+1, iy, iz+1);
	  rhos[6]=data(ix+1, iy+1, iz);
	  rhos[7]=data(ix+1, iy+1, iz+1);
	  DataT min=rhos[0];
	  DataT max=rhos[0];
	  for(int i=1;i<8;i++){
	    if(rhos[i]<min)
	      min=rhos[i];
	    if(rhos[i]>max)
	      max=rhos[i];
	  }
	  // Figure out what bits to turn on running from min to max.
	  mcell.turn_on_bits(min, max, data_min, data_max);
	}
      }
    }
  } else {
    int nx=xsize[depth-1];
    int ny=ysize[depth-1];
    int nz=zsize[depth-1];
    BrickArray3<MetaCT>& mcells=macrocells[depth];
    for(int x=startx;x<endx;x++){
      for(int y=starty;y<endy;y++){
	for(int z=startz;z<endz;z++){
	  // Compute the mcell for this block and store it in tmp
	  MetaCT tmp;
	  calc_mcell(depth-1, x*nx, y*ny, z*nz, tmp);
	  // Stash it away
	  mcells(x,y,z)=tmp;
	  // Now aggregate all the mcells created for this depth by
	  // doing a bitwise or.
	  mcell |= tmp;
	}
      }
    }
  }
}

// This function should not be called if depth is less than 2.
template<class DataT, class MetaCT>
void HVolumeVis<DataT,MetaCT>::parallel_calc_mcell(int)
{
  int ny=ysize[depth-1];
  int nz=zsize[depth-1];
  int nnx=xsize[depth-2];
  int nny=ysize[depth-2];
  int nnz=zsize[depth-2];
  BrickArray3<MetaCT>& mcells=macrocells[depth-1];
  int s, e;
  while(work->nextAssignment(s, e)){
    for(int block=s;block<e;block++){
      int z=block%nz;
      int y=(block%(nz*ny))/nz;
      int x=(block/(ny*nz));
      MetaCT tmp;
      calc_mcell(depth-2, x*nnx, y*nny, z*nnz, tmp);
      mcells(x,y,z)=tmp;
    }
  }
}

#if 0
template<class DataT, class MetaCT>
void HVolumeVis<DataT,MetaCT>::isect(int depth, float isoval, double t,
				  double dtdx, double dtdy, double dtdz,
				  double next_x, double next_y, double next_z,
				  int ix, int iy, int iz,
				  int dix_dx, int diy_dy, int diz_dz,
				  int startx, int starty, int startz,
				  const Vector& cellcorner,
				  const Vector& celldir,
				  const Ray& ray, HitInfo& hit,
				  DepthStats* st, PerProcessorContext* ppc)
{
  //cerr << "startx = " << startx << "\tix = " << ix << endl;
  //cerr << "starty = " << starty << "\tiy = " << iy << endl;
  //cerr << "startz = " << startx << "\tiz = " << iz << endl;
  //flush(cerr);
  //cerr << "start depth: " << depth << "\n";
  int cx=xsize[depth];
  int cy=ysize[depth];
  int cz=zsize[depth];
  if(depth==0){
    for(;;){
      int gx=startx+ix;
      int gy=starty+iy;
      int gz=startz+iz;
      if(gx<nx-1 && gy<ny-1 && gz<nz-1){
	//cerr << "Doing cell: " << gx << ", " << gy << ", " << gz
	//<< " (" << startx << "+" << ix << ", " << starty << "+" << iy << ", " << startz << "+" << iz << ")\n";
	DataT rhos[8];
	rhos[0]=data(gx, gy, gz);
	rhos[1]=data(gx, gy, gz+1);
	rhos[2]=data(gx, gy+1, gz);
	rhos[3]=data(gx, gy+1, gz+1);
	rhos[4]=data(gx+1, gy, gz);
	rhos[5]=data(gx+1, gy, gz+1);
	rhos[6]=data(gx+1, gy+1, gz);
	rhos[7]=data(gx+1, gy+1, gz+1);
	DataT min=rhos[0];
	DataT max=rhos[0];
	for(int i=1;i<8;i++){
	  if(rhos[i]<min)
	    min=rhos[i];
	  if(rhos[i]>max)
	    max=rhos[i];
	}
	if(min < isoval && max>isoval){
	  double hit_t;
	  Point p0(this->min+sdiag*Vector(gx,gy,gz));
	  Point p1(p0+sdiag);
	  double tmax=next_x;
	  if(next_y<tmax)
	    tmax=next_y;
	  if(next_z<tmax)
	    tmax=next_z;
	  float rho[2][2][2];
	  rho[0][0][0]=rhos[0];
	  rho[0][0][1]=rhos[1];
	  rho[0][1][0]=rhos[2];
	  rho[0][1][1]=rhos[3];
	  rho[1][0][0]=rhos[4];
	  rho[1][0][1]=rhos[5];
	  rho[1][1][0]=rhos[6];
	  rho[1][1][1]=rhos[7];
	  if(HitCell(ray, p0, p1, rho, isoval, t, tmax, hit_t)){
	    if(hit.hit(this, hit_t)){
#if 0
	      Point p(ray.origin()+ray.direction()*hit_t);
	      Vector sp((p-this->min)/datadiag*Vector(nx-1,ny-1,nz-1));
	      int x=(int)(sp.x()+.5);
	      int y=(int)(sp.y()+.5);
	      int z=(int)(sp.z()+.5);
	      if(x<1 || x>=nx-1 || y<1 || y>=ny-1 || z<1 || z>=nz-1){
#endif
		Vector* n=(Vector*)hit.scratchpad;
		*n=GradientCell(p0, p1, ray.origin()+ray.direction()*hit_t, rho);
		n->normalize();
		break;
#if 0
	      }
	      float fx=sp.x()-x;
	      float fy=sp.y()-y;
	      float fz=sp.z()-z;
	      float fx1=1-fx;
	      float fy1=1-fy;
	      float fz1=1-fz;
	      
	      Vector g[2][2][2];
	      for(int dx=0;dx<2;dx++){
		for(int dy=0;dy<2;dy++){
		  for(int dz=0;dz<2;dz++){
		    int x=gx+dx;
		    int y=gy+dy;
		    int z=gz+dz;
		    DataT ddx=data(x+1, y, z)-data(x-1, y, z);
		    DataT ddy=data(x, y+1, z)-data(x, y-1, z);
		    DataT ddz=data(x, y, z+1)-data(x, y, z-1);
		    
		    g[dx][dy][dz]=Vector(ddx,ddy,ddz)/sdiag;
		    g[dx][dy][dx].normalize();
		  }
		}
	      }
	      Vector* n=(Vector*)hit.scratchpad;
	      //*n=GradientCell(p0, p1, ray.origin()+ray.direction()*hit_t, rho);
	      Vector v00=g[0][0][0]*fz1+g[0][0][1]*fz;
	      Vector v01=g[0][1][0]*fz1+g[0][1][1]*fz;
	      Vector v10=g[1][0][0]*fz1+g[1][0][1]*fz;
	      Vector v11=g[1][1][0]*fz1+g[1][1][1]*fz;
	      
	      Vector v0=v00*fy1+v01*fy;
	      Vector v1=v10*fy1+v11*fy;
	      *n=v0*fx1+v1*fx;
	      n->normalize();
	      break;
#endif
	    }
	  }
	}
      }
      if(next_x < next_y && next_x < next_z){
	// Step in x...
	t=next_x;
	next_x+=dtdx;
	ix+=dix_dx;
	if(ix<0 || ix>=cx)
	  break;
      } else if(next_y < next_z){
	t=next_y;
	next_y+=dtdy;
	iy+=diy_dy;
	if(iy<0 || iy>=cy)
	  break;
      } else {
	t=next_z;
	next_z+=dtdz;
	iz+=diz_dz;
	if(iz<0 || iz>=cz)
	  break;
      }
    }
  } else {
    BrickArray3<MetaCT>& mcells=macrocells[depth];
    for(;;){
      int gx=startx+ix;
      int gy=starty+iy;
      int gz=startz+iz;
      //cerr << "startx = " << startx << "\tix = " << ix << endl;
      //cerr << "starty = " << starty << "\tiy = " << iy << endl;
      //cerr << "startz = " << startx << "\tiz = " << iz << endl;
      //flush(cerr);
      MetaCT& mcell=mcells(gx,gy,gz);
      //cerr << "doing macrocell: " << gx << ", " << gy << ", " << gz << ": " << mcell.min << ", " << mcell.max << '\n';
      if(mcell.max>isoval && mcell.min<isoval){
	// Do this cell...
	int new_cx=xsize[depth-1];
	int new_cy=ysize[depth-1];
	int new_cz=zsize[depth-1];
	int new_ix=(int)((cellcorner.x()+t*celldir.x()-ix)*new_cx);
	int new_iy=(int)((cellcorner.y()+t*celldir.y()-iy)*new_cy);
	int new_iz=(int)((cellcorner.z()+t*celldir.z()-iz)*new_cz);
	//cerr << "new: " << (cellcorner.x()+t*celldir.x()-ix)*new_cx
	//<< " " << (cellcorner.y()+t*celldir.y()-iy)*new_cy
	//<< " " << (cellcorner.z()+t*celldir.z()-iz)*new_cz
	//<< '\n';
	if(new_ix<0)
	  new_ix=0;
	else if(new_ix>=new_cx)
	  new_ix=new_cx-1;
	if(new_iy<0)
	  new_iy=0;
	else if(new_iy>=new_cy)
	  new_iy=new_cy-1;
	if(new_iz<0)
	  new_iz=0;
	else if(new_iz>=new_cz)
	  new_iz=new_cz-1;
	
	double new_dtdx=dtdx*ixsize[depth-1];
	double new_dtdy=dtdy*iysize[depth-1];
	double new_dtdz=dtdz*izsize[depth-1];
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
	int new_startx=gx*new_cx;
	int new_starty=gy*new_cy;
	int new_startz=gz*new_cz;
	//cerr << "startz=" << startz << '\n';
	//cerr << "iz=" << iz << '\n';
	//cerr << "new_cz=" << new_cz << '\n';
	Vector cellsize(new_cx, new_cy, new_cz);
	isect(depth-1, isoval, t,
	      new_dtdx, new_dtdy, new_dtdz,
	      new_next_x, new_next_y, new_next_z,
	      new_ix, new_iy, new_iz,
	      dix_dx, diy_dy, diz_dz,
	      new_startx, new_starty, new_startz,
	      (cellcorner-Vector(ix, iy, iz))*cellsize, celldir*cellsize,
	      ray, hit, st, ppc);
      }
      if(next_x < next_y && next_x < next_z){
	// Step in x...
	t=next_x;
	next_x+=dtdx;
	ix+=dix_dx;
	if(ix<0 || ix>=cx)
	  break;
      } else if(next_y < next_z){
	t=next_y;
	next_y+=dtdy;
	iy+=diy_dy;
	if(iy<0 || iy>=cy)
	  break;
      } else {
	t=next_z;
	next_z+=dtdz;
	iz+=diz_dz;
	if(iz<0 || iz>=cz)
	  break;
      }
      if(hit.min_t < t)
	break;
    }
  }
  //cerr << "end depth: " << depth << "\n";
}
#endif

template<class DataT, class MetaCT>
void HVolumeVis<DataT,MetaCT>::shade(Color& result, const Ray& ray,
				     const HitInfo& hit, int depth,
				     double atten, const Color& accumcolor,
				     Context* ctx)
{
  const Vector dir(ray.direction());
  const Point orig(ray.origin());
  int dix_dx;
  int ddx;
  if(dir.x() > 0){
    dix_dx=1;
    ddx=1;
  } else {
    dix_dx=-1;
    ddx=0;
  }	
  int diy_dy;
  int ddy;
  if(dir.y() > 0){
    diy_dy=1;
    ddy=1;
  } else if(dir.y() <-1.e-6){
    diy_dy=-1;
    ddy=0;
  }
  int diz_dz;
  int ddz;
  if(dir.z() > 0){
    diz_dz=1;
    ddz=1;
  } else {
    diz_dz=-1;
    ddz=0;
  }

  float t_min = hit.min_t;
  //  float* t_maxp = (float*)hit.scratchpad;
  //  float t_max = *t_maxp;
  
  Point start_p(orig+dir*t_min);
  Vector s((start_p-min)*ihierdiag);
  //cout << "s = " << s << "\tdepth = " << depth << endl;
  int cx=xsize[depth-1];
  int cy=ysize[depth-1];
  int cz=zsize[depth-1];
  int ix=(int)(s.x()*cx);
  int iy=(int)(s.y()*cy);
  int iz=(int)(s.z()*cz);
  //cerr << "ix = " << ix << endl;
  //cerr << "iy = " << iy << endl;
  //cerr << "iz = " << iz << endl;
  if(ix>=cx)
    ix--;
  if(iy>=cy)
    iy--;
  if(iz>=cz)
    iz--;
  if(ix<0)
    ix++;
  if(iy<0)
    iy++;
  if(iz<0)
    iz++;
  //cerr << "ix = " << ix << endl;
  //cerr << "iy = " << iy << endl;
  //cerr << "iz = " << iz << endl;
  
  double next_x, next_y, next_z;
  double dtdx, dtdy, dtdz;
  double icx=ixsize[depth-1];
  double x=min.x()+hierdiag.x()*double(ix+ddx)*icx;
  double xinv_dir=1./dir.x();
  next_x=(x-orig.x())*xinv_dir;
  dtdx=dix_dx*hierdiag.x()*icx*xinv_dir;
  double icy=iysize[depth-1];
  double y=min.y()+hierdiag.y()*double(iy+ddy)*icy;
  double yinv_dir=1./dir.y();
  next_y=(y-orig.y())*yinv_dir;
  dtdy=diy_dy*hierdiag.y()*icy*yinv_dir;
  double icz=izsize[depth-1];
  double z=min.z()+hierdiag.z()*double(iz+ddz)*icz;
  double zinv_dir=1./dir.z();
  next_z=(z-orig.z())*zinv_dir;
  dtdz=diz_dz*hierdiag.z()*icz*zinv_dir;
  
  Vector cellsize(cx,cy,cz);
  Vector cellcorner((orig-min)*ihierdiag*cellsize);
  Vector celldir(dir*ihierdiag*cellsize);

#if 0
  isect(depth-1, isoval, t, dtdx, dtdy, dtdz, next_x, next_y, next_z,
	ix, iy, iz, dix_dx, diy_dy, diz_dz,
	0, 0, 0,
	cellcorner, celldir,
	ray, hit, st, ppc);
#endif
}

template<class DataT, class MetaCT>
void HVolumeVis<DataT,MetaCT>::print(ostream& out) {
  //  out << "name_ = "<<get_name()<<endl;
  out << "min = "<<min<<", max = "<<max<<endl;
  out << "datadiag = "<<datadiag<<", hierdiag = "<<hierdiag<<", ihierdiag = "<<ihierdiag<<", sdiag = "<<sdiag<<endl;
  out << "dim = ("<<nx<<", "<<ny<<", "<<nz<<")\n";
  out << "data.get_datasize() = "<<data.get_datasize()<<endl;
  out << "data_min = "<<data_min<<", data_max = "<<data_max<<endl;
  out << "depth = "<<depth<<endl;
}

} // end namespace rtrt

#endif // HVOLUMEVIS_H
