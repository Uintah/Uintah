
#ifndef HVOLUME_H
#define HVOLUME_H 1

#include <Packages/rtrt/Core/VolumeBase.h>
#include <Core/Geometry/Point.h>
#include <Packages/rtrt/Core/Array3.h>
#include <stdlib.h>
#include <Core/Thread/WorkQueue.h>

#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Mutex.h>
#include <stdio.h>
#include <fstream>
#include <unistd.h>
#include <iostream>
#include <fcntl.h>

namespace rtrt {

using SCIRun::Mutex;
using SCIRun::WorkQueue;

template<class T>
struct VMCell {
  T max;
  T min;
};

template<class T>
class TypeInfo {
public:
  static T get_min() {
    ASSERTFAIL("TypeInfo::get_min():not implemented for this type");
    return 0;
  }
  static T get_max() {
    ASSERTFAIL("TypeInfo::get_max():not implemented for this type");
    return 0;
  }
};

class TypeInfo<unsigned char> {
public:
  static unsigned char get_min() { return 0; }
  static unsigned char get_max() { return 255; }
};

class TypeInfo<short> {
public:
  static short get_min() { return MAXSHORT+1; }
  static short get_max() { return MAXSHORT; }
};

class TypeInfo<int> {
public:
  static int get_min() { return -MAXINT-1; }
  static int get_max() { return MAXINT; }
};

class TypeInfo<float> {
public:
  static float get_min() { return -MAXFLOAT; }
  static float get_max() { return MAXFLOAT; }
};
  
class TypeInfo<double> {
public:
  static double get_min() { return -MAXDOUBLE; }
  static double get_max() { return MAXDOUBLE; }
};
  
template<class T, class A, class B>
class HVolume : public VolumeBase {
protected:
  inline int bound(const int val, const int min, const int max) {
    return (val>min?(val<max?val:max):min);
  }
public:
  Point min;
  Vector datadiag;
  Vector hierdiag;
  Vector ihierdiag;
  Vector sdiag;
  int nx,ny,nz;
  Array3<T> indata;
  A blockdata;
  T datamin, datamax;
  int depth;
  int* xsize;
  int* ysize;
  int* zsize;
  double* ixsize;
  double* iysize;
  double* izsize;
  B* macrocells;
  WorkQueue* work;
  void brickit(int);
  void parallel_calc_mcell(int);
  char* filebase;
  void calc_mcell(int depth, int ix, int iy, int iz, VMCell<T>& mcell);
  void isect(int depth, float isoval, double t,
	     double dtdx, double dtdy, double dtdz,
	     double next_x, double next_y, double next_z,
	     int ix, int iy, int iz,
	     int dix_dx, int diy_dy, int diz_dz,
	     int startx, int starty, int startz,
	     const Vector& cellcorner, const Vector& celldir,
	     const Ray& ray, HitInfo& hit,
	     DepthStats* st, PerProcessorContext* ppc);
  HVolume(Material* matl, VolumeDpy* dpy,
	  char* filebase, int depth, int np);
  HVolume(Material* matl, VolumeDpy* dpy,
	  int depth, int np,
	  int _nx, int _ny, int _nz,
	  Point min, Point max,
	  T _datamin, T _datamax, Array3<T> _indata);
  HVolume(Material* matl, VolumeDpy* dpy, HVolume<T,A,B>* share);
  virtual ~HVolume();
  virtual void intersect(Ray& ray, HitInfo& hit, DepthStats* st,
			 PerProcessorContext*);
  virtual Vector normal(const Point&, const HitInfo& hit);
  virtual void compute_bounds(BBox&, double offset);
  virtual void preprocess(double maxradius, int& pp_offset, int& scratchsize);
  virtual void compute_hist(int nhist, int* hist,
			    float datamin, float datamax);
  virtual void get_minmax(float& min, float& max);

  virtual void print(ostream& out);
  virtual bool interior_value( double& /*value*/, const Ray &/*ray*/,
			       const double /*t*/);
};
  

  /////////////////////////////////////////////////
  /////////////////////////////////////////////////
  // C code
  /////////////////////////////////////////////////
  /////////////////////////////////////////////////
  

extern Mutex io_lock_;
  
template<class T, class A, class B>
HVolume<T,A,B>::HVolume(Material* matl, VolumeDpy* dpy,
			char* filebase, int depth, int np)
  : VolumeBase(matl, dpy), depth(depth), filebase(filebase)
{
  this->filebase=strdup(filebase);
  if(depth<=0)
    depth=1;
  char buf[200];
  sprintf(buf, "%s.hdr", filebase);
  ifstream in(buf);
  if(!in){
    cerr << "Error opening header: " << buf << '\n';
    exit(1);
  }
  in >> nx >> ny >> nz;
  double x,y,z;
  in >> x >> y >> z;
  min=Point(x,y,z);
  in >> x >> y >> z;
  Point max(x,y,z);
  double dmin, dmax;
  in >> dmin >> dmax;
  datamin = (T)dmin;
  datamax = (T)dmax;
  if(!in){
    cerr << "Error reading header: " << buf << '\n';
    exit(1);
  }
  datadiag=max-min;
  sdiag=datadiag/Vector(nx-1,ny-1,nz-1);
  
  blockdata.resize(nx, ny, nz);
  sprintf(buf, "%s.brick", filebase);
  cout << "buf = " << buf << endl;
  //  ifstream bin(buf);
  int bin_fd = open(buf, O_RDONLY);
#if 0
  if(!bin){
    cerr << "Direct I/O failed, trying without\n";
    bin=open(buf, O_RDONLY);
  }
#endif
  
  if(bin_fd == -1){
    cerr << "Brick data not found, reading data file\n";
    int din_fd = open (filebase, O_RDONLY);
    if(din_fd == -1) {
      cerr << "Error opening data file: " << filebase << '\n';
      exit(1);
    }
    indata.resize(nx, ny, nz);
    
    double start=SCIRun::Time::currentSeconds();
    cerr << "Reading " << filebase << "...";
    cerr.flush();
    read(din_fd, indata.get_dataptr(), indata.get_datasize());
    double dt=SCIRun::Time::currentSeconds()-start;
    cerr << "done in " << dt << " seconds (" << (double)(sizeof(T)*nx*ny*nz)/dt/1024/1024 << " MB/sec)\n";
    int s = close (din_fd);
    if(s == -1 ) {
      cerr << "Error reading data file: " << filebase << '\n';
      exit(1);
    }
    cerr << "Done reading data\n";
    
    int bnp=np>8?8:np;
    cerr << "Bricking data with " << bnp << " processors\n";
    work=new WorkQueue("Bricking");
    work->refill(nx, bnp, 5);
    SCIRun::Parallel<HVolume<T,A,B> > phelper(this, &HVolume<T,A,B>::brickit);
    SCIRun::Thread::parallel(phelper, bnp, true);
    delete work;
    
    int bout_fd;
#ifdef __sgi
    ///////////////////////////////////////////////////////////////
    // write the bricked data to a file, so that we don't have to rebrick it
    //    ofstream bout(buf);
    //    if (!bout) {
    bout_fd = open (buf, O_WRONLY | O_CREAT | O_TRUNC,
		    S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
#else
    ASSERTFAIL("Can only write bricked data on SGI.");
#endif    

    if (bout_fd == -1 ) {
      cerr << "Error in opening file " << buf << " for writing.\n";
      exit(1);
    }
    cerr << "Writing " << buf << "...";
    start=SCIRun::Time::currentSeconds();	
    write(bout_fd, blockdata.get_dataptr(),blockdata.get_datasize());
    dt=SCIRun::Time::currentSeconds()-start;
    cerr << "done (" << (double)(blockdata.get_datasize())/dt/1024/1024 << " MB/sec)\n";
    indata.resize(0,0,0);
  } else {
#ifndef __sgi
    ASSERTFAIL("Can't do direct io on non-sgi machines.");
#else
#if 0
    struct dioattr s;
    if(fcntl(bin, F_DIOINFO, &s) == 0 && s.d_mem>0)
      fprintf(stderr, "direct io: d_mem=%d, d_miniosz=%d, d_maxiosz=%d\n", s.d_mem, s.d_miniosz, s.d_maxiosz);
    else {
      fprintf(stderr, "No direct io\n");
      s.d_maxiosz=16*1024*1024;
      s.d_mem=8;
    }
#endif
    cerr << "Reading " << buf << "...";
    cerr.flush();
    double start=SCIRun::Time::currentSeconds();
#if 1
    read(bin_fd, blockdata.get_dataptr(),blockdata.get_datasize());
#else
    cerr << "dataptr=" << blockdata.get_dataptr() << '\n';
    //    cerr << "bin=" << bin << '\n';
    cerr.flush();
    unsigned long ss=blockdata.get_datasize();
    ss=(ss+s.d_miniosz)/s.d_miniosz*s.d_miniosz;
    unsigned long total=0;
    while(total != ss){
      int t=ss-total;
      if(t>s.d_maxiosz)
	t=s.d_maxiosz;
      cerr << "reading: " << t << " bytes\n";
      int n=read(bin_fd, blockdata.get_dataptr(), t);
      cerr << "n=" << n << '\n';
      if(n != t){
	perror("read");
	cerr << "total=" << total << "\n";
	cerr << "ss=" << ss << "\n";
	cerr << "Error reading data file: " << filebase << '\n';
	exit(1);
      }
      total+=t;
    }
#endif // sgi
    double dt=SCIRun::Time::currentSeconds()-start;
    cerr << "done (" << (double)(blockdata.get_datasize())/dt/1024/1024 << " MB/sec)\n";
    close(bin_fd);
#endif
  }
  
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
    macrocells=new B[depth+1];
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
    VMCell<T> top;
    calc_mcell(depth-1, 0, 0, 0, top);
    cerr << "Min: " << top.min << ", Max: " << top.max << '\n';
#else
    int nx=xsize[depth-1];
    int ny=ysize[depth-1];
    int nz=zsize[depth-1];
    int totaltop=nx*ny*nz;
    work=new WorkQueue("Building hierarchy");
    work->refill(totaltop, np, 5);
    SCIRun::Parallel<HVolume<T,A,B> > phelper(this, &HVolume<T,A,B>::parallel_calc_mcell);
    SCIRun::Thread::parallel(phelper, np, true);
    delete work;
#endif
    cerr << "done\n";
  }
}

template<class T, class A, class B>
HVolume<T,A,B>::HVolume(Material* matl, VolumeDpy* dpy,
			int depth, int np,
			int _nx, int _ny, int _nz,
			Point min, Point max,
			T _datamin, T _datamax, Array3<T> _indata):
  VolumeBase(matl, dpy), depth(depth), work(NULL), filebase(NULL),
  nx(_nx), ny(_ny), nz(_nz), min(min), datadiag(max-min),
  datamin(_datamin), datamax(_datamax)
{
  indata.resize(nx,ny,nz);
  int nn=nx*ny*nz;
  T *orig=_indata.get_dataptr();
  T *cpy=indata.get_dataptr();
  for (int c=0; c<nn; c++) cpy[c]=orig[c];

  //cerr << "Dim of indata = (" << indata.dim1() << ", " << indata.dim2() <<
  //", " << indata.dim3() << ")\n";
#if 0
  {
    T my_datamin,my_datamax;
    int num_items = indata.dim1() * indata.dim2() * indata.dim3();
    T* data_ptr = indata.get_dataptr();
    my_datamin = my_datamax = data_ptr[0];
    for (int i = 0; i < num_items; i++) {
      my_datamin = Min(my_datamin, data_ptr[i]);
      my_datamax = Max(my_datamax, data_ptr[i]);
    }
    cerr << "my_datamin = " << (double)my_datamin << "\tmy_datamax = " << (double)my_datamax << endl;
  }
#endif
  
  if(depth<=0)
    depth=1;
  sdiag=datadiag/Vector(nx-1,ny-1,nz-1);

  blockdata.resize(nx, ny, nz);
  
  // brick the data
  double start=SCIRun::Time::currentSeconds();
  //cerr << "Bricking data...\n";
  //cerr.flush();
  int bnp=np>2?2:np;
  //cerr << "Bricking data with " << bnp << " processors\n";
  //cerr.flush();
  cerr << "Bricking data with " << bnp << " processors\n";
  work=new WorkQueue("Bricking");
  work->refill(nx, bnp, 5);
  SCIRun::Parallel<HVolume<T,A,B> > phelper(this, &HVolume<T,A,B>::brickit);
  SCIRun::Thread::parallel(phelper, bnp, true);
  delete work;

  double dt=SCIRun::Time::currentSeconds()-start;
  cerr << "Bricking data...done (" << dt << " sec)\n";
  cerr.flush();
  indata.resize(0,0,0);

#if 0
  {
    T my_datamin,my_datamax;
    my_datamin = my_datamax = blockdata(0,0,0);
    for (int x = 0; x < blockdata.dim1(); x++)
      for (int y = 0; y < blockdata.dim2(); y++)
	for (int z = 0; z < blockdata.dim3(); z++)
	  {
	    my_datamin = Min(my_datamin, blockdata(x,y,z));
	    my_datamax = Max(my_datamax, blockdata(x,y,z));
	    if (blockdata(x,y,z) != (x + y + z)) {
	      cerr << "blockdata(" << x << "," << y << "," << z << ") = " <<
		blockdata(x,y,z) << " instead of " << (x + y + z) << endl;
	    }
	  }
    cerr << "my_datamin = " << my_datamin << "\tmy_datamax = " << my_datamax << endl;
  }
#endif
  
  // process stuff
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
    macrocells=new B[depth+1];
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
#if 1
    VMCell<T> top;
    calc_mcell(depth-1, 0, 0, 0, top);
    cerr << "Min: " << top.min << ", Max: " << top.max << '\n';
#else
    int nx=xsize[depth-1];
    int ny=ysize[depth-1];
    int nz=zsize[depth-1];
    int totaltop=nx*ny*nz;
    work=new WorkQueue("Building hierarchy");
    work->refill(totaltop, np, 5);
    SCIRun::Parallel<HVolume<T,A,B> > phelper(this, &HVolume<T,A,B>::parallel_calc_mcell);
    Thread::parallel(phelper, np, true);
    delete work;
#endif
    cerr << "done\n";
  }
  
}

template<class T, class A, class B>
HVolume<T,A,B>::HVolume(Material* matl, VolumeDpy* dpy,
			       HVolume<T,A,B>* share)
  : VolumeBase(matl, dpy)
{
  min=share->min;
  datadiag=share->datadiag;
  hierdiag=share->hierdiag;
  ihierdiag=share->ihierdiag;
  sdiag=share->sdiag;
  nx=share->nx;
  ny=share->ny;
  nz=share->nz;
  indata.share(share->indata);
  blockdata.share(share->blockdata);
  datamin=share->datamin;
  datamax=share->datamax;
  depth=share->depth;
  xsize=share->xsize;
  ysize=share->ysize;
  zsize=share->zsize;
  ixsize=share->ixsize;
  iysize=share->iysize;
  izsize=share->izsize;
  macrocells=share->macrocells;
  filebase=share->filebase;
}

template<class T, class A, class B>
HVolume<T,A,B>::~HVolume()
{
}

template<class T, class A, class B>
void HVolume<T,A,B>::preprocess(double, int&, int&)
{
}

template<class T, class A, class B>
void HVolume<T,A,B>::calc_mcell(int depth, int startx, int starty, int startz,
				VMCell<T>& mcell)
{
  mcell.min=TypeInfo<T>::get_max();
  mcell.max=TypeInfo<T>::get_min();
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
    mcell.min=datamax;
    mcell.max=datamin;
    return;
  }
  if(depth==0){
    for(int ix=startx;ix<endx;ix++){
      for(int iy=starty;iy<endy;iy++){
	for(int iz=startz;iz<endz;iz++){
	  T rhos[8];
	  rhos[0]=blockdata(ix, iy, iz);
	  rhos[1]=blockdata(ix, iy, iz+1);
	  rhos[2]=blockdata(ix, iy+1, iz);
	  rhos[3]=blockdata(ix, iy+1, iz+1);
	  rhos[4]=blockdata(ix+1, iy, iz);
	  rhos[5]=blockdata(ix+1, iy, iz+1);
	  rhos[6]=blockdata(ix+1, iy+1, iz);
	  rhos[7]=blockdata(ix+1, iy+1, iz+1);
	  T min=rhos[0];
	  T max=rhos[0];
	  for(int i=1;i<8;i++){
	    if(rhos[i]<min)
	      min=rhos[i];
	    if(rhos[i]>max)
	      max=rhos[i];
	  }
	  if(min<mcell.min)
	    mcell.min=min;
	  if(max>mcell.max)
	    mcell.max=max;
	}
      }
    }
  } else {
    int nx=xsize[depth-1];
    int ny=ysize[depth-1];
    int nz=zsize[depth-1];
    B& mcells=macrocells[depth];
    for(int x=startx;x<endx;x++){
      for(int y=starty;y<endy;y++){
	for(int z=startz;z<endz;z++){
	  VMCell<T> tmp;
	  calc_mcell(depth-1, x*nx, y*ny, z*nz, tmp);
	  if(tmp.min < mcell.min)
	    mcell.min=tmp.min;
	  if(tmp.max > mcell.max)
	    mcell.max=tmp.max;
	  mcells(x,y,z)=tmp;
	}
      }
    }
  }
}

template<class T, class A, class B>
void HVolume<T,A,B>::parallel_calc_mcell(int)
{
  int ny=ysize[depth-1];
  int nz=zsize[depth-1];
  int nnx=xsize[depth-2];
  int nny=ysize[depth-2];
  int nnz=zsize[depth-2];
  B& mcells=macrocells[depth-1];
  int s, e;
  while(work->nextAssignment(s, e)){
    for(int block=s;block<e;block++){
      int z=block%nz;
      int y=(block%(nz*ny))/nz;
      int x=(block/(ny*nz));
      VMCell<T> tmp;
      calc_mcell(depth-2, x*nnx, y*nny, z*nnz, tmp);
      mcells(x,y,z)=tmp;
    }
  }
}

template<class T, class A, class B>
void HVolume<T,A,B>::compute_bounds(BBox& bbox, double offset)
{
  //  cout << "HVolume::compute_bounds::min = "<<min<<", datadiag = "<<datadiag<<"\n";
  bbox.extend(min-Vector(offset,offset,offset));
  bbox.extend(min+datadiag+Vector(offset,offset,offset));
}

extern int HitCell(const Ray& r, const Point& pmin, const Point& pmax, 
		   float rho[2][2][2], float iso, double tmin, double tmax, double& t);
extern Vector GradientCell(const Point& pmin, const Point& pmax,
			   const Point& p, float rho[2][2][2]);

template<class T, class A, class B>
void HVolume<T,A,B>::isect(int depth, float isoval, double t,
			   double dtdx, double dtdy, double dtdz,
			   double next_x, double next_y, double next_z,
			   int ix, int iy, int iz,
			   int dix_dx, int diy_dy, int diz_dz,
			   int startx, int starty, int startz,
			   const Vector& cellcorner, const Vector& celldir,
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
	T rhos[8];
	rhos[0]=blockdata(gx, gy, gz);
	rhos[1]=blockdata(gx, gy, gz+1);
	rhos[2]=blockdata(gx, gy+1, gz);
	rhos[3]=blockdata(gx, gy+1, gz+1);
	rhos[4]=blockdata(gx+1, gy, gz);
	rhos[5]=blockdata(gx+1, gy, gz+1);
	rhos[6]=blockdata(gx+1, gy+1, gz);
	rhos[7]=blockdata(gx+1, gy+1, gz+1);
	T min=rhos[0];
	T max=rhos[0];
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
		    T ddx=blockdata(x+1, y, z)
		      -blockdata(x-1, y, z);
		    T ddy=blockdata(x, y+1, z)
		      -blockdata(x, y-1, z);
		    T ddz=blockdata(x, y, z+1)
		      -blockdata(x, y, z-1);
		    
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
    B& mcells=macrocells[depth];
    for(;;){
      int gx=startx+ix;
      int gy=starty+iy;
      int gz=startz+iz;
      //cerr << "startx = " << startx << "\tix = " << ix << endl;
      //cerr << "starty = " << starty << "\tiy = " << iy << endl;
      //cerr << "startz = " << startx << "\tiz = " << iz << endl;
      //flush(cerr);
      VMCell<T>& mcell=mcells(gx,gy,gz);
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

template<class T, class A, class B>
void HVolume<T,A,B>::intersect(Ray& ray, HitInfo& hit,
			       DepthStats* st, PerProcessorContext* ppc)
{
  const Vector dir(ray.direction());
  const Point orig(ray.origin());
  Point max(min+hierdiag);
  double MIN, MAX;
  double xinv_dir=1./dir.x();
  int dix_dx;
  int ddx;
  if(dir.x() > 0){
    MIN=xinv_dir*(min.x()-orig.x());
    MAX=xinv_dir*(max.x()-orig.x());
    dix_dx=1;
    ddx=1;
    } else {
      MIN=xinv_dir*(max.x()-orig.x());
      MAX=xinv_dir*(min.x()-orig.x());
      dix_dx=-1;
      ddx=0;
    }	
  double y0, y1;
  int diy_dy;
  int ddy;
  double yinv_dir=1./dir.y();
  if(dir.y() > 0){
    y0=yinv_dir*(min.y()-orig.y());
    y1=yinv_dir*(max.y()-orig.y());
    diy_dy=1;
    ddy=1;
  } else if(dir.y() <-1.e-6){
    y0=yinv_dir*(max.y()-orig.y());
    y1=yinv_dir*(min.y()-orig.y());
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
  double zinv_dir=1./dir.z();
  if(dir.z() > 0){
    z0=zinv_dir*(min.z()-orig.z());
    z1=zinv_dir*(max.z()-orig.z());
    diz_dz=1;
    ddz=1;
  } else {
    z0=zinv_dir*(max.z()-orig.z());
    z1=zinv_dir*(min.z()-orig.z());
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
  Vector s((p-min)*ihierdiag);
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
  next_x=(x-orig.x())*xinv_dir;
  dtdx=dix_dx*hierdiag.x()*icx*xinv_dir;
  double icy=iysize[depth-1];
  double y=min.y()+hierdiag.y()*double(iy+ddy)*icy;
  next_y=(y-orig.y())*yinv_dir;
  dtdy=diy_dy*hierdiag.y()*icy*yinv_dir;
  double icz=izsize[depth-1];
  double z=min.z()+hierdiag.z()*double(iz+ddz)*icz;
  next_z=(z-orig.z())*zinv_dir;
  dtdz=diz_dz*hierdiag.z()*icz*zinv_dir;
  
  Vector cellsize(cx,cy,cz);
  Vector cellcorner((orig-min)*ihierdiag*cellsize);
  Vector celldir(dir*ihierdiag*cellsize);
  float isoval=dpy->isoval;
  //cerr << "isoval = "<<isoval<<" ";
  
  isect(depth-1, isoval, t, dtdx, dtdy, dtdz, next_x, next_y, next_z,
	ix, iy, iz, dix_dx, diy_dy, diz_dz,
	0, 0, 0,
	cellcorner, celldir,
	ray, hit, st, ppc);
  //  if (hit.was_hit && hit.hit_obj == this)
  //    cerr << name_ << " was hit."<<endl;
}

template<class T, class A, class B>
Vector HVolume<T,A,B>::normal(const Point&, const HitInfo& hit)
{
  // We computed the normal at intersect time and tucked it
  // away in the scratchpad...
  Vector* n=(Vector*)hit.scratchpad;
  return *n;
}

template<class T, class A, class B>
void HVolume<T,A,B>::compute_hist(int nhist, int* hist,
				  float datamin, float datamax)
{
  bool recompute_hist = true;
  char buf[200];
  if (filebase != NULL) {
    sprintf(buf, "%s.hist_%d", filebase, nhist);
    cerr << "Looking for histogram in " << buf << "\n";
    ifstream in(buf);
    if(in){
      for(int i=0;i<nhist;i++){
	in >> hist[i];
      }
      recompute_hist = false;
    }
  }
  if (recompute_hist) {
    cerr << "Recomputing Histgram\n";
    float scale=(nhist-1)/(datamax-datamin);
    int nx1=nx-1;
    int ny1=ny-1;
    int nz1=nz-1;
    int nynz=ny*nz;
    //cerr << "scale = " << scale;
    //cerr << "\tnx1 = " << nx1;
    //cerr << "\tny1 = " << ny1;
    //cerr << "\tnz1 = " << nz1;
    //cerr << "\tnynz = " << nynz << endl;
    for(int ix=0;ix<nx1;ix++){
      for(int iy=0;iy<ny1;iy++){
	int idx=ix*nynz+iy*nz;
	for(int iz=0;iz<nz1;iz++){
	  T p000=blockdata(ix,iy,iz);
	  T p001=blockdata(ix,iy,iz+1);
	  T p010=blockdata(ix,iy+1,iz);
	  T p011=blockdata(ix,iy+1,iz+1);
	  T p100=blockdata(ix+1,iy,iz);
	  T p101=blockdata(ix+1,iy,iz+1);
	  T p110=blockdata(ix+1,iy+1,iz);
	  T p111=blockdata(ix+1,iy+1,iz+1);
	  T min=Min(Min(Min(p000, p001), Min(p010, p011)), Min(Min(p100, p101), Min(p110, p111)));
	  T max=Max(Max(Max(p000, p001), Max(p010, p011)), Max(Max(p100, p101), Max(p110, p111)));
	  int nmin=(int)((min-datamin)*scale);
	  int nmax=(int)((max-datamin)*scale+.999999);
	  if(nmax>=nhist)
	    nmax=nhist-1;
	  if(nmin<0)
	    nmin=0;
	  if(nmax>nhist)
	    nmax=nhist;
	  //if ((nmin != 0) || (nmax != 0))
	  //  cerr << "nmin = " << nmin << "\tnmax = " << nmax << endl;
	  for(int i=nmin;i<nmax;i++){
	    hist[i]++;
	  }
	  idx++;
	}
      }
    }
    if (filebase != NULL) {
      ofstream out(buf);
      for(int i=0;i<nhist;i++){
	out << hist[i] << '\n';
      }
    }
  }
  cerr << "Done building histogram\n";
}    

template<class T, class A, class B>
void HVolume<T,A,B>::brickit(int /*proc*/)
{
  int sx, ex;
  while(work->nextAssignment(sx, ex)){
    for(int x=sx;x<ex;x++){
      io_lock_.lock();
      //      cerr << "processor " << proc << ": " << x << " of " << nx-1 << "\n";
      io_lock_.unlock();
      for(int y=0;y<ny;y++){
	for(int z=0;z<nz;z++){
	  T value=indata(x,y,z);
	  blockdata(x,y,z)=value;
	}
      }
    }
  }
}

template<class T, class A, class B>
void HVolume<T,A,B>::get_minmax(float& min, float& max)
{
  min=datamin;
  max=datamax;
}

template<class T, class A, class B>
void HVolume<T,A,B>::print(ostream& out) {
  //  out << "name_ = "<<get_name()<<endl;
  out << "min = "<<min<<endl;
  out << "datadiag = "<<datadiag<<", hierdiag = "<<hierdiag<<", ihierdiag = "<<ihierdiag<<", sdiag = "<<sdiag<<endl;
  out << "dim = ("<<nx<<", "<<ny<<", "<<nz<<")\n";
  out << "indata.get_datasize() = "<<indata.get_datasize()<<endl;
  out << "blockdata.get_datasize() = "<<blockdata.get_datasize()<<endl;
  out << "datamin = "<<datamin<<", datamax = "<<datamax<<endl;
  out << "depth = "<<depth<<endl;
}

template<class T, class A, class B>
bool HVolume<T,A,B>::interior_value( double& value, const Ray &ray,
				     const double t) {
  // Get the location of the point
  Point p = ray.eval(t);

  ////////////////////////////////////////////////////////////
  // Check bounds and return false if the point is outside

  if (p.x() < min.x() || p.y() < min.y() || p.z() < min.z())
    return false;
  Vector max(min+datadiag);
  if (p.x() > max.x() || p.y() > max.y() || p.z() > max.z())
    return false;
  
  ////////////////////////////////////////////////////////////
  // interpolate the point

  p = p - min.vector();
  // get the indices and weights for the indicies
  //  float norm = p.x() * inv_diag.x();
  //  float step = norm * (nx - 1);
  float step = p.x() / sdiag.x();
  int x_low = bound((int)step, 0, blockdata.dim1()-2);
  //  int x_low = (int)step;
  //  if (x_low > blockdata.dim1()-2) x_low = blockdata.dim1()-2;
  int x_high = x_low+1;
  float x_weight_low = x_high - step;
  
  //  norm = p.y() * inv_diag.y();
  //  step = norm * (ny - 1);
  step = p.y() / sdiag.y();
  int y_low = bound((int)step, 0, blockdata.dim2()-2);
  //  int y_low = (int)step;
  //  if (y_low > blockdata.dim1()-2) y_low = blockdata.dim2()-2;
  int y_high = y_low+1;
  float y_weight_low = y_high - step;
  
  //  norm = p.z() * inv_diag.z();
  //  step = norm * (nz - 1);
  step = p.z() / sdiag.z();
  int z_low = bound((int)step, 0, blockdata.dim3()-2);
  //  int z_low = (int)step;
  //  if (z_low > blockdata.dim3()-2) z_low = blockdata.dim3()-2;
  int z_high = z_low+1;
  float z_weight_low = z_high - step;
  
  ////////////////////////////////////////////////////////////
  // do the interpolation
  
  float a,b,c,d,e,f,g,h;
  a = blockdata(x_low,  y_low,  z_low);
  b = blockdata(x_low,  y_low,  z_high);
  c = blockdata(x_low,  y_high, z_low);
  d = blockdata(x_low,  y_high, z_high);
  e = blockdata(x_high, y_low,  z_low);
  f = blockdata(x_high, y_low,  z_high);
  g = blockdata(x_high, y_high, z_low);
  h = blockdata(x_high, y_high, z_high);
  
  float lz1, lz2, lz3, lz4, ly1, ly2;
  lz1 = a * z_weight_low + b * (1 - z_weight_low);
  lz2 = c * z_weight_low + d * (1 - z_weight_low);
  lz3 = e * z_weight_low + f * (1 - z_weight_low);
  lz4 = g * z_weight_low + h * (1 - z_weight_low);
  
  ly1 = lz1 * y_weight_low + lz2 * (1 - y_weight_low);
  ly2 = lz3 * y_weight_low + lz4 * (1 - y_weight_low);
  
  value = ly1 * x_weight_low + ly2 * (1 - x_weight_low);
  return true;
}

} // end namespace rtrt

#endif
