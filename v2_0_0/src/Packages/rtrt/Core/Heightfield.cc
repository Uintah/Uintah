#include <Packages/rtrt/Core/Heightfield.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Color.h>

#include <Core/Thread/Thread.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/WorkQueue.h>
#include <Core/Thread/Mutex.h>

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

#include <fstream>
#include <iostream>

#include <Packages/rtrt/Core/UV.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

using namespace rtrt;
using namespace std;
using SCIRun::Mutex;
using SCIRun::Parallel;
using SCIRun::Thread;

extern Mutex io_lock_;


template<class A, class B>
SCIRun::Persistent*
Heightfield<A,B>::maker() {
  return new Heightfield<A,B>;
}

template<class A, class B>
SCIRun::PersistentTypeID
Heightfield<A,B>::type_id(Heightfield<A,B>::type_name(), "Object", maker);


template<class A, class B>
const string
Heightfield<A,B>::type_name() 
{
  static const string name("Heightfield<A,B>");
  // hack type_name needs to be supported for more than one templated type!
  return name;
}

template<class A, class B>
Heightfield<A,B>::Heightfield(Material* matl, char* filebase,
			      int depth, int np)
  : Object(matl,this), depth(depth), filebase(filebase), np_(np)
{
    this->filebase=strdup(filebase);
    if(depth<1)
	depth=1;
    char buf[200];
    sprintf(buf, "%s.hdr", filebase);
    ifstream in(buf);
    if(!in){
	cerr << "Error opening header: " << buf << '\n';
	exit(1);
    }
    in >> nx >> ny;
    in >> x1 >> y1;
    in >> x2 >> y2;
    in >> datamin >> datamax;
    if(!in){
	cerr << "Error reading header: " << buf << '\n';
	exit(1);
    }
    min=Point(x1,y1,datamin);
    Point max(x2,y2,datamax);
    datadiag=max-min;
    sdiag=datadiag/Vector(nx-1,ny-1,1);

    blockdata.resize(nx, ny);
    sprintf(buf, "%s.brick", filebase);
    int bin=open(buf, O_RDONLY);//|O_DIRECT);
    if(bin == -1){
	cerr << "Direct I/O failed, trying without\n";
	bin=open(buf, O_RDONLY);
    }

    if(bin == -1){

      //ifstream din(filebase);

      int fd = open(filebase,O_RDONLY);

	//if(!din){
	//  cerr << "Error opening data file: " << filebase << '\n';
	//  exit(1);
	//}

	if( fd == -1 ){
	  cerr << "Error opening data file: " << filebase << '\n';
	  exit(1);
	}
	indata.resize(nx, ny);

	cerr << "Reading " << filebase << "...";
	cerr.flush();
	//read(din.rdbuf()->fd(), indata.get_dataptr(), indata.get_datasize());
	read(fd, indata.get_dataptr(), indata.get_datasize());

	//	if(!din){
	//	    cerr << "Error reading data file: " << filebase << '\n';
	//	    exit(1);
	//	}
	cerr << "Done reading data\n";

	int bnp=np>8?8:np;
	cerr << "Bricking data with " << bnp << " processors\n";
	work=new WorkQueue("Bricking"); // , nx, bnp, false, 5);
	work->refill(nx, bnp, 5);
	Parallel<Heightfield<A,B> > phelper(this, &Heightfield<A,B>::brickit);
	Thread::parallel(phelper, bnp, true);
	delete work;

	int boutfd = open(buf, O_TRUNC | O_WRONLY | O_CREAT, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH );

	if (boutfd == -1)
	  {
	    perror("Cannot write to %s.  ");
	  }
//  	ofstream bout(buf);
	cerr << "Writing " << buf << "...";
//  	write(bout.rdbuf()->fd(), blockdata.get_dataptr(),
//  	      blockdata.get_datasize());
	write(boutfd, blockdata.get_dataptr(),
	      blockdata.get_datasize());
//  	cerr << "done (" << (double)(blockdata.get_datasize())/dt/1024/1024 << " MB/sec)\n";
	indata.resize(0,0);
    } else {
#if __sgi
	struct dioattr s;
#if 0
	if(fcntl(bin, F_DIOINFO, &s) == 0 && s.d_mem>0)
	    fprintf(stderr, "direct io: d_mem=%d, d_miniosz=%d, d_maxiosz=%d\n", s.d_mem, s.d_miniosz, s.d_maxiosz);
	else {
#endif
	    fprintf(stderr, "No direct io\n");
	    s.d_miniosz=1;
	    s.d_maxiosz=16*1024*1024;
	    s.d_mem=8;
#if 0
	}
#endif
	cerr << "Reading " << buf << "...";
	cerr.flush();
//  	double start=Thread::currentSeconds();
//  	cerr << "dataptr=" << blockdata.get_dataptr() << '\n';
//  	cerr << "bin=" << bin << '\n';
//  	cerr.flush();
	unsigned long ss=blockdata.get_datasize();
	ss=(ss+s.d_miniosz-1)/s.d_miniosz*s.d_miniosz;
	unsigned long total=0;
	while(total != ss){
	    int t=ss-total;
	    if(t>s.d_maxiosz)
		t=s.d_maxiosz;
	    cerr << "reading: " << t << " bytes\n";
	    int n=read(bin, (char*)blockdata.get_dataptr()+total, t);
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
//  	double dt=Thread::currentSeconds()-start;
//  	cerr << "done (" << (double)(blockdata.get_datasize())/dt/1024/1024 << " MB/sec)\n";
	close(bin);
#else
	cerr << "Error - can only use direct IO on sgis.\n";
	exit(1);
#endif
    }

    xsize=new int[depth];
    ysize=new int[depth];
    int tx=nx-1;
    int ty=ny-1;
    for(int i=depth-1;i>=0;i--){
	int nx=(int)(pow(tx, 1./(i+1))+.9);
	tx=(tx+nx-1)/nx;
	xsize[depth-i-1]=nx;
	int ny=(int)(pow(ty, 1./(i+1))+.9);
	ty=(ty+ny-1)/ny;
	ysize[depth-i-1]=ny;
    }
    ixsize=new double[depth];
    iysize=new double[depth];
    cerr << "Calculating depths...\n";
    for(int i=0;i<depth;i++){
	cerr << "xsize=" << xsize[i] << ", ysize=" << ysize[i] << '\n';
	ixsize[i]=1./xsize[i];
	iysize[i]=1./ysize[i];
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
    hierdiag=datadiag*Vector(tx,ty,1)/Vector(nx-1,ny-1,1);
    ihierdiag=Vector(1.,1.,1.)/hierdiag;

    if(depth==1){
	macrocells=0;
    } else {
	macrocells=new B[depth+1];
	int xs=1;
	int ys=1;
	for(int d=depth-1;d>=1;d--){
	    xs*=xsize[d];
	    ys*=ysize[d];
	    macrocells[d].resize(xs, ys);
	    cerr << "Depth " << d << ": " << xs << "x" << ys << '\n';
	}
	cerr << "Building hierarchy\n";
	int nx=xsize[depth-1];
	int ny=ysize[depth-1];
	int totaltop=nx*ny;
	work=new WorkQueue("Building hierarchy" /*, totaltop, np, false, 5*/);
	work->refill(totaltop,np,5);
	Parallel<Heightfield<A,B> > phelper(this, &Heightfield<A,B>::parallel_calc_mcell);
	Thread::parallel(phelper, np, true);
	delete work;
	cerr << "done\n";
    }
}

template<class A, class B>
Heightfield<A,B>::Heightfield(Material* matl, Heightfield<A,B>* share) 
    : Object(matl,this)
{
    min=share->min;
    datadiag=share->datadiag;
    hierdiag=share->hierdiag;
    ihierdiag=share->ihierdiag;
    sdiag=share->sdiag;
    nx=share->nx;
    ny=share->ny;
    indata.share(share->indata);
    blockdata.share(share->blockdata);
    datamin=share->datamin;
    datamax=share->datamax;
    depth=share->depth;
    xsize=share->xsize;
    ysize=share->ysize;
    ixsize=share->ixsize;
    iysize=share->iysize;
    macrocells=share->macrocells;
    filebase=share->filebase;
}

template<class A, class B>
Heightfield<A,B>::~Heightfield()
{
}

template<class A, class B>
void Heightfield<A,B>::preprocess(double, int&, int&)
{
}

template<class A, class B>
void Heightfield<A,B>::calc_mcell(int depth, int startx, int starty,
				  HMCell<typename A::data_type>& mcell)
{
    mcell.min=datamax;
    mcell.max=datamin;
    int endx=startx+xsize[depth];
    int endy=starty+ysize[depth];
    if(endx>nx-1)
	endx=nx-1;
    if(endy>ny-1)
	endy=ny-1;
    if(startx>=endx || starty>=endy){
	/* This cell won't get used... */
	return;
    }
    if(depth==0){
	for(int ix=startx;ix<endx;ix++){
	    for(int iy=starty;iy<endy;iy++){
	       typename A::data_type rhos[4];
	       rhos[0]=blockdata(ix, iy);
	       rhos[1]=blockdata(ix+1, iy);
	       rhos[2]=blockdata(ix, iy+1);
	       rhos[3]=blockdata(ix+1, iy+1);
	       typename A::data_type min=rhos[0];
	       typename A::data_type max=rhos[0];
	       for(int i=1;i<4;i++){
		  if(rhos[i]<min)
		     min=rhos[i];
		  if(rhos[i]>max)
		     max=rhos[i];
	       }
#ifdef DEBUG
	       cerr << "ix=" << ix << ", iy=" << iy << ", min=" << min << ", max=" << max << '\n';
#endif
	       if(min<mcell.min)
		  mcell.min=min;
	       if(max>mcell.max)
		  mcell.max=max;
	    }
	}
    } else {
	int nx=xsize[depth-1];
	int ny=ysize[depth-1];
	B& mcells=macrocells[depth];
	for(int x=startx;x<endx;x++){
	    for(int y=starty;y<endy;y++){
	       typename B::data_type tmp;
	       calc_mcell(depth-1, x*nx, y*ny, tmp);
	       if(tmp.min < mcell.min)
		  mcell.min=tmp.min;
	       if(tmp.max > mcell.max)
		  mcell.max=tmp.max;
	       mcells(x,y)=tmp;
#ifdef DEBUG
	       cerr << "mcells(" << x << ", " << y << ")=" << mcells(x,y).min << ", " << mcells(x,y).max << '\n';
#endif
	    }
	}
    }
}

template<class A, class B>
void Heightfield<A,B>::parallel_calc_mcell(int)
{
    int ny=ysize[depth-1];
    int nnx=xsize[depth-2];
    int nny=ysize[depth-2];
    B& mcells=macrocells[depth-1];
    int s, e;
    while(work->nextAssignment(s, e)){
	for(int block=s;block<e;block++){
	    int y=block%ny;
	    int x=block/ny;
	    typename B::data_type tmp;
	    calc_mcell(depth-2, x*nnx, y*nny, tmp);
	    mcells(x,y)=tmp;
#ifdef DEBUG
	    cerr << "mcells(" << x << ", " << y << ")=" << mcells(x,y).min << ", " << mcells(x,y).max << '\n';
#endif
	}
    }
}

template<class A, class B>
void Heightfield<A,B>::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend(min-Vector(offset,offset,offset));
    bbox.extend(min+datadiag+Vector(offset,offset,offset));
}

namespace rtrt{

extern int HitCell(const Ray& r, const Point& pmin, const Point& pmax, 
		   float rho[2][2], double tmin, double tmax, double& t);
extern Vector GradientCell(const Point& pmin, const Point& pmax,
			   const Point& p, float rho[2][2]);
}

template<class A, class B>
void Heightfield<A,B>::isect_up(int depth, double t,
			     double dtdx, double dtdy,
			     double next_x, double next_y,
			     int ix, int iy,
			     int dix_dx, int diy_dy,
			     int startx, int starty,
			     const Vector& cellcorner, const Vector& celldir,
			     Ray& ray, HitInfo& hit,
			     DepthStats* st, PerProcessorContext* ppc)
{
#ifdef DEBUG
   cerr << "start depth: " << depth << "\n";
#endif
    int cx=xsize[depth];
    int cy=ysize[depth];
    if(depth==0){
       for(;;){
	    int gx=startx+ix;
	    int gy=starty+iy;
	    if(gx<nx-1 && gy<ny-1){
#ifdef DEBUG
	       cerr << "Doing cell: " << gx << ", " << gy
		    << " (" << startx << "+" << ix << ", " << starty << "+" << iy << ")\n";
#endif
	        typename A::data_type rhos[4];
		rhos[0]=blockdata(gx, gy);
		rhos[1]=blockdata(gx, gy+1);
		rhos[2]=blockdata(gx+1, gy);
		rhos[3]=blockdata(gx+1, gy+1);
#ifdef DEBUG
		cerr << "data: " << rhos[0] << ", " << rhos[1] << ", " << rhos[2] << ", " << rhos[3] << '\n';
#endif
		typename A::data_type min=rhos[0];
		typename A::data_type max=rhos[0];
		for(int i=1;i<4;i++){
		    if(rhos[i]<min)
			min=rhos[i];
		    if(rhos[i]>max)
			max=rhos[i];
		}
		double t1;
		if(next_x < next_y){
		   t1=next_x;
		} else {
		   t1=next_y;
		}
		double zmin = ray.origin().z()+ray.direction().z()*t;
		double zmax = ray.origin().z()+ray.direction().z()*t1;
		double mn = Min(zmax, (double)max);
		double mx = Max(zmin, (double)min);
#ifdef DEBUG
		cerr << "mn=" << mn << ", mx=" << mx << '\n';
		cerr << "zmin=" << zmin << ", zmax=" << zmax << '\n';
		cerr << "min=" << min << ", max=" << max << '\n';
#endif
		if(mn >= mx){
		    double hit_t;
		    Point p0(this->min+sdiag*Vector(gx,gy,0));
		    Point p1(p0+sdiag);
		    double tmax=next_x;
		    if(next_y<tmax)
			tmax=next_y;
		    float rho[2][2];
		    rho[0][0]=rhos[0];
		    rho[0][1]=rhos[1];
		    rho[1][0]=rhos[2];
		    rho[1][1]=rhos[3];
#ifdef DEBUG
		    cerr << "Calling hitCell\n";
#endif
		    if(HitCell(ray, p0, p1, rho, t, tmax, hit_t)){
			if(hit.hit(this, hit_t)){
#ifdef DEBUG
			   cerr << "Hit!\n";
#endif
			   Vector* n=(Vector*)hit.scratchpad;
			   *n=GradientCell(p0, p1, ray.origin()+ray.direction()*hit_t, rho);
			   n->normalize();
			   break;
			}
		    }
		}
	    }
	    if(next_x < next_y){
		// Step in x...
		t=next_x;
		next_x+=dtdx;
		ix+=dix_dx;
		if(ix<0 || ix>=cx)
		    break;
	    } else {
		t=next_y;
		next_y+=dtdy;
		iy+=diy_dy;
		if(iy<0 || iy>=cy)
		    break;
	    }
	}
    } else {
        B & mcells=macrocells[depth];
	for(;;){
	    int gx=startx+ix;
	    int gy=starty+iy;
	    typename B::data_type& mcell=mcells(gx,gy);
#ifdef DEBUG
	    cerr << "doing macrocell: " << gx << ", " << gy << ": " << mcell.min << ", " << mcell.max << '\n';
#endif
	    double t1;
	    if(next_x < next_y){
	       t1=next_x;
	    } else {
	       t1=next_y;
	    }
	    double zmin = ray.origin().z()+ray.direction().z()*t;
	    double zmax = ray.origin().z()+ray.direction().z()*t1;
	    double mn = Min(zmax, (double)mcell.max);
	    double mx = Max(zmin, (double)mcell.min);
#ifdef DEBUG
	    cerr << "zmin=" << zmin << ", zmax=" << zmax << '\n';
	    cerr << "t=" << t << ", t1=" << t1 << '\n';
#endif
	    if(mn >= mx){
		// Do this cell...
		int new_cx=xsize[depth-1];
		int new_cy=ysize[depth-1];
		int new_ix=(int)((cellcorner.x()+t*celldir.x()-ix)*new_cx);
		int new_iy=(int)((cellcorner.y()+t*celldir.y()-iy)*new_cy);
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

		double new_dtdx=dtdx*ixsize[depth-1];
		double new_dtdy=dtdy*iysize[depth-1];
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
		int new_startx=gx*new_cx;
		int new_starty=gy*new_cy;
		//cerr << "startz=" << startz << '\n';
		//cerr << "iz=" << iz << '\n';
		//cerr << "new_cz=" << new_cz << '\n';
		Vector cellsize(new_cx, new_cy, 1);
		isect_up(depth-1, t,
		      new_dtdx, new_dtdy,
		      new_next_x, new_next_y,
		      new_ix, new_iy,
		      dix_dx, diy_dy,
		      new_startx, new_starty,
		      (cellcorner-Vector(ix, iy, 0))*cellsize, celldir*cellsize,
		      ray, hit, st, ppc);
	    }
	    if(next_x < next_y){
		// Step in x...
		t=next_x;
		next_x+=dtdx;
		ix+=dix_dx;
		if(ix<0 || ix>=cx)
		    break;
	    } else {
		t=next_y;
		next_y+=dtdy;
		iy+=diy_dy;
		if(iy<0 || iy>=cy)
		    break;
	    }
	    if(hit.min_t < t)
		break;
	}
    }
#ifdef DEBUG
    cerr << "end depth: " << depth << "\n";
#endif
}

template<class A, class B>
void Heightfield<A,B>::isect_down(int depth, double t,
				  double dtdx, double dtdy,
				  double next_x, double next_y,
				  int ix, int iy,
				  int dix_dx, int diy_dy,
				  int startx, int starty,
				  const Vector& cellcorner, const Vector& celldir,
				  Ray& ray, HitInfo& hit,
				  DepthStats* st, PerProcessorContext* ppc)
{
#ifdef DEBUG
   cerr << "start depth: " << depth << "\n";
#endif
    int cx=xsize[depth];
    int cy=ysize[depth];
    if(depth==0){
       for(;;){
	    int gx=startx+ix;
	    int gy=starty+iy;
	    if(gx<nx-1 && gy<ny-1){
#ifdef DEBUG
	       cerr << "Doing cell: " << gx << ", " << gy
		    << " (" << startx << "+" << ix << ", " << starty << "+" << iy << ")\n";
#endif
		typename A::data_type rhos[4];
		rhos[0]=blockdata(gx, gy);
		rhos[1]=blockdata(gx, gy+1);
		rhos[2]=blockdata(gx+1, gy);
		rhos[3]=blockdata(gx+1, gy+1);
#ifdef DEBUG
		cerr << "data: " << rhos[0] << ", " << rhos[1] << ", " << rhos[2] << ", " << rhos[3] << '\n';
#endif
		typename A::data_type min=rhos[0];
		typename A::data_type max=rhos[0];
		for(int i=1;i<4;i++){
		    if(rhos[i]<min)
			min=rhos[i];
		    if(rhos[i]>max)
			max=rhos[i];
		}
		double t1;
		if(next_x < next_y){
		   t1=next_x;
		} else {
		   t1=next_y;
		}
		double zmin = ray.origin().z()+ray.direction().z()*t1;
		double zmax = ray.origin().z()+ray.direction().z()*t;
		double mn = Min(zmax, (double)max);
		double mx = Max(zmin, (double)min);
#ifdef DEBUG
		cerr << "mn=" << mn << ", mx=" << mx << '\n';
		cerr << "zmin=" << zmin << ", zmax=" << zmax << '\n';
		cerr << "min=" << min << ", max=" << max << '\n';
#endif
		if(mn >= mx){
		    double hit_t;
		    Point p0(this->min+sdiag*Vector(gx,gy,0));
		    Point p1(p0+sdiag);
		    double tmax=next_x;
		    if(next_y<tmax)
			tmax=next_y;
		    float rho[2][2];
		    rho[0][0]=rhos[0];
		    rho[0][1]=rhos[1];
		    rho[1][0]=rhos[2];
		    rho[1][1]=rhos[3];
#ifdef DEBUG
		    cerr << "Calling hitCell\n";
#endif
		    if(HitCell(ray, p0, p1, rho, t, tmax, hit_t)){
			if(hit.hit(this, hit_t)){
#ifdef DEBUG
			   cerr << "Hit!\n";
#endif
			   Vector* n=(Vector*)hit.scratchpad;
			   *n=GradientCell(p0, p1, ray.origin()+ray.direction()*hit_t, rho);
			   n->normalize();
			   break;
			}
		    }
		}
	    }
	    if(next_x < next_y){
		// Step in x...
		t=next_x;
		next_x+=dtdx;
		ix+=dix_dx;
		if(ix<0 || ix>=cx)
		    break;
	    } else {
		t=next_y;
		next_y+=dtdy;
		iy+=diy_dy;
		if(iy<0 || iy>=cy)
		    break;
	    }
	}
    } else {
	B& mcells=macrocells[depth];
	for(;;){
	    int gx=startx+ix;
	    int gy=starty+iy;
	    typename B::data_type& mcell=mcells(gx,gy);
#ifdef DEBUG
	    cerr << "doing macrocell: " << gx << ", " << gy << ": " << mcell.min << ", " << mcell.max << '\n';
#endif
	    double t1;
	    if(next_x < next_y){
	       t1=next_x;
	    } else {
	       t1=next_y;
	    }
	    double zmin = ray.origin().z()+ray.direction().z()*t1;
	    double zmax = ray.origin().z()+ray.direction().z()*t;
	    double mn = Min(zmax, (double)mcell.max);
	    double mx = Max(zmin, (double)mcell.min);
#ifdef DEBUG
	    cerr << "zmin=" << zmin << ", zmax=" << zmax << '\n';
	    cerr << "t=" << t << ", t1=" << t1 << '\n';
#endif
	    if(mn >= mx){
		// Do this cell...
		int new_cx=xsize[depth-1];
		int new_cy=ysize[depth-1];
		int new_ix=(int)((cellcorner.x()+t*celldir.x()-ix)*new_cx);
		int new_iy=(int)((cellcorner.y()+t*celldir.y()-iy)*new_cy);
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

		double new_dtdx=dtdx*ixsize[depth-1];
		double new_dtdy=dtdy*iysize[depth-1];
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
		int new_startx=gx*new_cx;
		int new_starty=gy*new_cy;
		//cerr << "startz=" << startz << '\n';
		//cerr << "iz=" << iz << '\n';
		//cerr << "new_cz=" << new_cz << '\n';
		Vector cellsize(new_cx, new_cy, 1);
		isect_down(depth-1, t,
		      new_dtdx, new_dtdy,
		      new_next_x, new_next_y,
		      new_ix, new_iy,
		      dix_dx, diy_dy,
		      new_startx, new_starty,
		      (cellcorner-Vector(ix, iy, 0))*cellsize, celldir*cellsize,
		      ray, hit, st, ppc);
	    }
	    if(next_x < next_y){
		// Step in x...
		t=next_x;
		next_x+=dtdx;
		ix+=dix_dx;
		if(ix<0 || ix>=cx)
		    break;
	    } else {
		t=next_y;
		next_y+=dtdy;
		iy+=diy_dy;
		if(iy<0 || iy>=cy)
		    break;
	    }
	    if(hit.min_t < t)
		break;
	}
    }
#ifdef DEBUG
    cerr << "end depth: " << depth << "\n";
#endif
}


template<class A, class B>
void Heightfield<A,B>::intersect(Ray& ray, HitInfo& hit,
				 DepthStats* st, PerProcessorContext* ppc)
{
  if (ray.already_tested[0] == this ||
      ray.already_tested[1] == this ||
      ray.already_tested[2] == this ||
      ray.already_tested[3] == this)
    return;
  else {
    ray.already_tested[3] = ray.already_tested[2];
    ray.already_tested[2] = ray.already_tested[1];
    ray.already_tested[1] = ray.already_tested[0];
    ray.already_tested[0] = this;
  }  
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
    } else {
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
    double zinv_dir=1./dir.z();
    if(dir.z() > 0){
	z0=zinv_dir*(min.z()-orig.z());
	z1=zinv_dir*(max.z()-orig.z());
    } else {
	z0=zinv_dir*(max.z()-orig.z());
	z1=zinv_dir*(min.z()-orig.z());
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
    Point start_p(orig+dir*t);
    Vector s((start_p-min)*ihierdiag);
    int cx=xsize[depth-1];
    int cy=ysize[depth-1];
    int ix=(int)(s.x()*cx);
    int iy=(int)(s.y()*cy);
    if(ix>=cx)
	ix--;
    if(iy>=cy)
	iy--;
    if(ix<0)
	ix++;
    if(iy<0)
	iy++;

    double next_x, next_y;
    double dtdx, dtdy;
    double icx=ixsize[depth-1];
    double x=min.x()+hierdiag.x()*double(ix+ddx)*icx;
    next_x=(x-start_p.x())*xinv_dir;
    dtdx=dix_dx*hierdiag.x()*icx*xinv_dir;
    double icy=iysize[depth-1];
    double y=min.y()+hierdiag.y()*double(iy+ddy)*icy;
    next_y=(y-start_p.y())*yinv_dir;
    dtdy=diy_dy*hierdiag.y()*icy*yinv_dir;

    Vector cellsize(cx,cy,1);
    Vector cellcorner((start_p-min)*ihierdiag*cellsize);
    Vector celldir(dir*ihierdiag*cellsize);

    
    // Make a new ray with the point start_p.  Be sure to offset t.
    Ray new_ray(start_p, dir);
    // Create a new HitInfo with the appropiate information
    HitInfo new_hit;
    if (hit.was_hit)
      // Offset min_t by the t from our new_ray
      new_hit.min_t = hit.min_t - t;
    if(dir.z() > 0)
       isect_up(depth-1, 0, dtdx, dtdy, next_x, next_y,
		ix, iy, dix_dx, diy_dy,
		0, 0,
		cellcorner, celldir,
		new_ray, new_hit, st, ppc);
    else
       isect_down(depth-1, 0, dtdx, dtdy, next_x, next_y,
		  ix, iy, dix_dx, diy_dy,
		  0, 0,
		  cellcorner, celldir,
		  new_ray, new_hit, st, ppc);    
    if (new_hit.was_hit) {
      // Since this would only be true if the intersection point was
      // closer than min_t, we can be safe to assume that the current
      // object is now the closest object and we should update hit.
      
      // We need to offset hit.min_t
      hit = new_hit;
      hit.min_t += t;
    }
}

template<class A, class B>
Vector Heightfield<A,B>::normal(const Point&, const HitInfo& hit)
{
    // We computed the normal at intersect time and tucked it
    // away in the scratchpad...
    Vector* n=(Vector*)hit.scratchpad;
    return *n;
}

template<class A, class B>
void Heightfield<A,B>::brickit(int proc)
{
    int sx, ex;
    while(work->nextAssignment(sx, ex)){
	for(int x=sx;x<ex;x++){
	    io_lock_.lock();
	    cerr << "processor " << proc << ": " << x << " of " << nx-1 << "\n";
	    io_lock_.unlock();
	    for(int y=0;y<ny;y++){
	       typename A::data_type value=indata(x,y);
	       blockdata(x,y)=value;
	    }
	}
    }
}

#if 0
template<class T, class A, class B>
void Heightfield<T,A,B>::get_minmax(float& min, float& max)
{
    min=datamin;
    max=datamax;
}
#endif


template<class A, class B>
void Heightfield<A,B>::uv(UV& uv, const Point&p, const HitInfo&)
{
    double scalex = 1./(x2-x1);
    double scaley = 1./(y2-y1);

    double u = (p.x()-x1)*scalex;
    double v = (p.y()-y1)*scaley;

    uv.set(v,u);

}

const int HEIGHTFIELD_VERSION = 1;

template<class A, class B>
void Heightfield<A,B>::io(SCIRun::Piostream &str) 
{
  str.begin_class("Heightfield<A,B>", HEIGHTFIELD_VERSION);
  Object::io(str);
  UVMapping::io(str);
  SCIRun::Pio(str, min);
  SCIRun::Pio(str, datadiag);
  SCIRun::Pio(str, hierdiag);
  SCIRun::Pio(str, ihierdiag);
  SCIRun::Pio(str, sdiag);
  SCIRun::Pio(str, nx);
  SCIRun::Pio(str, ny);
  SCIRun::Pio(str, x1);
  SCIRun::Pio(str, y1);
  SCIRun::Pio(str, x2);
  SCIRun::Pio(str, y2);
  SCIRun::Pio(str, maxx);
  SCIRun::Pio(str, minx);
  SCIRun::Pio(str, maxy);
  rtrt::Pio(str, indata);
  rtrt::Pio(str, blockdata);
  SCIRun::Pio(str, datamin);
  SCIRun::Pio(str, datamax);
  SCIRun::Pio(str, depth);

  if (str.reading()) {
    xsize = new int[depth];
    ysize = new int[depth];
    ixsize = new double[depth];
    iysize = new double[depth];
    if(depth==1){
      macrocells=0;
    } else {
      macrocells=new B[depth+1];
    }
  }
  for (int i = 0; i < depth; i++) {
    SCIRun::Pio(str, xsize[i]);
    SCIRun::Pio(str, ysize[i]);
    SCIRun::Pio(str, ixsize[i]);
    SCIRun::Pio(str, iysize[i]);
  }
  for (int i = 0; i < depth + 1; i++) {
    rtrt::Pio(str, macrocells[i]);
  }
  SCIRun::Pio(str, np_);

  if(str.reading()) {
    int bnp=np_>8?8:np_;
    work=new WorkQueue("Bricking"); // , nx, bnp, false, 5);
    work->refill(nx, bnp, 5);
    Parallel<Heightfield<A,B> > phelper(this, &Heightfield<A,B>::brickit);
    Thread::parallel(phelper, bnp, true);
    delete work;
  }
  str.end_class();
}

namespace SCIRun {
template<class T> void Pio(Piostream&str, rtrt::HMCell<T>&o)
{
  str.begin_cheap_delim();
  SCIRun::Pio(str, o.min);
  SCIRun::Pio(str, o.max);
  str.end_cheap_delim();
}
}
