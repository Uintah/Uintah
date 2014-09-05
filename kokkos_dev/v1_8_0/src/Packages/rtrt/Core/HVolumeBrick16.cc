
#include <Packages/rtrt/Core/HVolumeBrick16.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/WorkQueue.h>
#include <stdio.h>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

using namespace rtrt;
using namespace std;
using namespace SCIRun;


namespace rtrt {
  extern SCIRun::Mutex io_lock_;
  
  struct VMCell16 {
    float max;
    float min;
  };
} // end namespace rtrt

HVolumeBrick16::HVolumeBrick16(Material* matl, VolumeDpy* dpy,
			       char* filebase, int depth, int np)
    : VolumeBase(matl, dpy), depth(depth), filebase(filebase)
{
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
    in >> datamin >> datamax;
    if(!in){
	cerr << "Error reading header: " << buf << '\n';
	exit(1);
    }
    datadiag=max-min;
    sdiag=datadiag/Vector(nx-1,ny-1,nz-1);

#define L1 4
#define L2 5
    int totalx=(nx+L2*L1-1)/(L2*L1);
    int totaly=(ny+L2*L1-1)/(L2*L1);
    int totalz=(nz+L2*L1-1)/(L2*L1);

    xidx=new int[nx];

    for(int x=0;x<nx;x++){
	int m1x=x%L1;
	int xx=x/L1;
	int m2x=xx%L2;
	int m3x=xx/L2;
	xidx[x]=m3x*totaly*totalz*L2*L2*L2*L1*L1*L1+m2x*L2*L2*L1*L1*L1+m1x*L1*L1;
    }
    yidx=new int[ny];
    for(int y=0;y<ny;y++){
	int m1y=y%L1;
	int yy=y/L1;
	int m2y=yy%L2;
	int m3y=yy/L2;
	yidx[y]=m3y*totalz*L2*L2*L2*L1*L1*L1+m2y*L2*L1*L1*L1+m1y*L1;
    }
    zidx=new int[nz];
    for(int z=0;z<nz;z++){
	int m1z=z%L1;
	int zz=z/L1;
	int m2z=zz%L2;
	int m3z=zz/L2;
	zidx[z]=m3z*L2*L2*L2*L1*L1*L1+m2z*L1*L1*L1+m1z;
    }

    int totalsize=totalx*totaly*totalz*L2*L2*L2*L1*L1*L1;
#undef L1
#undef L2
    blockdata=new short[totalsize];
    if(!blockdata){
	cerr << "Error allocating data array, totalsize=" << totalsize << "\n";
	exit(1);
    }
    sprintf(buf, "%s.brick", filebase);
    //ifstream bin(buf);
    int bin_fd  = open (buf, O_RDONLY);
    if(bin_fd == -1){
      //ifstream din(filebase);
      int din_fd = open (filebase, O_RDONLY);
	if(din_fd == -1){
	    cerr << "Error opening data file: " << filebase << '\n';
	    exit(1);
	}
	indata=new short[nx*ny*nz];
	if(!indata){
	    cerr << "Error allocating data array\n";
	    exit(1);
	}
	double start=SCIRun::Time::currentSeconds();
	//din.read((char*)data, sizeof(short)*nx*ny*nz);
	cerr << "Reading " << filebase << "...";
	cerr.flush();
	//read(din.rdbuf()->fd(), indata, sizeof(short)*nx*ny*nz);
	read(din_fd, indata, sizeof(short)*nx*ny*nz);
	double dt=SCIRun::Time::currentSeconds()-start;
	cerr << "done in " << dt << " seconds (" << (double)(sizeof(short)*nx*ny*nz)/dt/1024/1024 << " MB/sec)\n";
	int s = close(din_fd);
	if(s == -1) {
	    cerr << "Error reading data file: " << filebase << '\n';
	    exit(1);
	}
	cerr << "Done reading data\n";

	int bnp=np>2?2:np;
	cerr << "Bricking data with " << bnp << " processors\n";
	work=new WorkQueue("Bricking");
	work->refill(nx, bnp, 5);
	Parallel<HVolumeBrick16> phelper(this, &HVolumeBrick16::brickit);
	Thread::parallel(phelper, bnp, true);
	delete work;

	//ofstream bout(buf);
	int bout_fd = open (buf, O_WRONLY | O_CREAT | O_TRUNC);
	if (bout_fd == -1) {
	  cerr << "Error in opening " << buf << " for writing.\n";
	  exit(1);
	}
	cerr << "Writing " << buf << "...";
	start=SCIRun::Time::currentSeconds();	
	//write(bout.rdbuf()->fd(), blockdata, sizeof(short)*totalsize);
	write(bout_fd, blockdata, sizeof(short)*totalsize);
	dt=SCIRun::Time::currentSeconds()-start;
	cerr << "done (" << (double)(sizeof(short)*totalsize)/dt/1024/1024 << " MB/sec)\n";
	delete[] indata;
    } else {
	//din.read((char*)data, sizeof(short)*nx*ny*nz);
	cerr << "Reading " << buf << "...";
	cerr.flush();
	double start=SCIRun::Time::currentSeconds();
	//read(bin.rdbuf()->fd(), blockdata, sizeof(short)*totalsize);
	read(bin_fd, blockdata, sizeof(short)*totalsize);
	double dt=SCIRun::Time::currentSeconds()-start;
	cerr << "done (" << (double)(sizeof(short)*totalsize)/dt/1024/1024 << " MB/sec)\n";
	int s = close(bin_fd);
	if(s == -1) {
	    cerr << "Error reading data file: " << filebase << '\n';
	    exit(1);
	}
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
	macrocells=new VMCell16*[depth+1];
	macrocells[0]=0;
	macrocell_xidx=new int*[depth+1];
	macrocell_yidx=new int*[depth+1];
	macrocell_zidx=new int*[depth+1];
	macrocell_xidx[0]=0;
	macrocell_yidx[0]=0;
	macrocell_zidx[0]=0;
#define L1 2
#define L2 6
	int xs=1;
	int ys=1;
	int zs=1;
	int total_macrocells=0;
	int total_indices=0;
	for(int d=depth-1;d>=1;d--){
	    xs*=xsize[d];
	    ys*=ysize[d];
	    zs*=zsize[d];
	    //cerr << d << ": " << xs << ", " << ys << ", " << zs << "\n";
	    total_indices+=xs+ys+zs;
	    int totalx=(xs+L2*L1-1)/(L2*L1);
	    int totaly=(ys+L2*L1-1)/(L2*L1);
	    int totalz=(zs+L2*L1-1)/(L2*L1);
	    //cerr << d << ": " << totalx << ", " << totaly << ", " << totalz << "\n";
	    total_macrocells+=totalx*totaly*totalz;
	}
	xs=1;
	ys=1;
	zs=1;
	total_macrocells*=L1*L1*L1*L2*L2*L2;
	VMCell16* p=new VMCell16[total_macrocells];
	int* indices=new int[total_indices];
	cerr << "Allocating " << total_macrocells << " macrocells and " << total_indices << " indices\n";
	for(int d=depth-1;d>=1;d--){
	    xs*=xsize[d];
	    ys*=ysize[d];
	    zs*=zsize[d];
	    int totalx=(xs+L2*L1-1)/(L2*L1);
	    int totaly=(ys+L2*L1-1)/(L2*L1);
	    int totalz=(zs+L2*L1-1)/(L2*L1);
	    macrocells[d]=p;
	    p+=totalx*totaly*totalz*L1*L1*L1*L2*L2*L2;
	    int* xidx=macrocell_xidx[d]=indices;
	    indices+=xs;
	    for(int x=0;x<xs;x++){
		int m1x=x%L1;
		int xx=x/L1;
		int m2x=xx%L2;
		int m3x=xx/L2;
		xidx[x]=m3x*totaly*totalz*L2*L2*L2*L1*L1*L1+m2x*L2*L2*L1*L1*L1+m1x*L1*L1;
	    }
	    int* yidx=macrocell_yidx[d]=indices;
	    indices+=ys;
	    for(int y=0;y<ys;y++){
		int m1y=y%L1;
		int yy=y/L1;
		int m2y=yy%L2;
		int m3y=yy/L2;
		yidx[y]=m3y*totalz*L2*L2*L2*L1*L1*L1+m2y*L2*L1*L1*L1+m1y*L1;
	    }
	    int* zidx=macrocell_zidx[d]=indices;
	    indices+=zs;
	    for(int z=0;z<zs;z++){
		int m1z=z%L1;
		int zz=z/L1;
		int m2z=zz%L2;
		int m3z=zz/L2;
		zidx[z]=m3z*L2*L2*L2*L1*L1*L1+m2z*L1*L1*L1+m1z;
	    }
	}
#undef L1
#undef L2
	cerr << "Building hierarchy\n";
#if 0
	VMCell16 top;
	calc_mcell(depth-1, 0, 0, 0, top);
	cerr << "Min: " << top.min << ", Max: " << top.max << '\n';
#else
	int nx=xsize[depth-1];
	int ny=ysize[depth-1];
	int nz=zsize[depth-1];
	int totaltop=nx*ny*nz;
	work=new WorkQueue("Building hierarchy");
	work->refill(totaltop, np, 5);
	Parallel<HVolumeBrick16> phelper(this, &HVolumeBrick16::parallel_calc_mcell);
	Thread::parallel(phelper, np, true);
	delete work;
#endif
	cerr << "done\n";
    }
}

HVolumeBrick16::HVolumeBrick16(Material* matl, VolumeDpy* dpy,
			       HVolumeBrick16* share)
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
    indata=share->indata;
    blockdata=share->blockdata;
    datamin=share->datamin;
    datamax=share->datamax;
    xidx=share->xidx;
    yidx=share->yidx;
    zidx=share->zidx;
    depth=share->depth;
    xsize=share->xsize;
    ysize=share->ysize;
    zsize=share->zsize;
    ixsize=share->ixsize;
    iysize=share->iysize;
    izsize=share->izsize;
    macrocells=share->macrocells;
    filebase=share->filebase;
    macrocell_xidx=share->macrocell_xidx;
    macrocell_yidx=share->macrocell_yidx;
    macrocell_zidx=share->macrocell_zidx;
}

HVolumeBrick16::~HVolumeBrick16()
{
    if(blockdata)
	delete[] blockdata;
}

void HVolumeBrick16::preprocess(double, int&, int&)
{
}

void HVolumeBrick16::calc_mcell(int depth, int startx, int starty, int startz,
				VMCell16& mcell)
{
    mcell.min=MAXSHORT;
    mcell.max=-MAXSHORT;
    int endx=startx+xsize[depth];
    int endy=starty+ysize[depth];
    int endz=startz+zsize[depth];
    if(depth==0){
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
	for(int ix=startx;ix<endx;ix++){
	    for(int iy=starty;iy<endy;iy++){
		for(int iz=startz;iz<endz;iz++){
		    short rhos[8];
		    int idx000=xidx[ix]+yidx[iy]+zidx[iz];
		    rhos[0]=blockdata[idx000];
		    int idx001=xidx[ix]+yidx[iy]+zidx[iz+1];
		    rhos[1]=blockdata[idx001];
		    int idx010=xidx[ix]+yidx[iy+1]+zidx[iz];
		    rhos[2]=blockdata[idx010];
		    int idx011=xidx[ix]+yidx[iy+1]+zidx[iz+1];
		    rhos[3]=blockdata[idx011];
		    int idx100=xidx[ix+1]+yidx[iy]+zidx[iz];
		    rhos[4]=blockdata[idx100];
		    int idx101=xidx[ix+1]+yidx[iy]+zidx[iz+1];
		    rhos[5]=blockdata[idx101];
		    int idx110=xidx[ix+1]+yidx[iy+1]+zidx[iz];
		    rhos[6]=blockdata[idx110];
		    int idx111=xidx[ix+1]+yidx[iy+1]+zidx[iz+1];
		    rhos[7]=blockdata[idx111];
		    short min=rhos[0];
		    short max=rhos[0];
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
	VMCell16* mcells=macrocells[depth];
	int* mxidx=macrocell_xidx[depth];
	int* myidx=macrocell_yidx[depth];
	int* mzidx=macrocell_zidx[depth];
	for(int x=startx;x<endx;x++){
	    for(int y=starty;y<endy;y++){
		for(int z=startz;z<endz;z++){
		    VMCell16 tmp;
		    calc_mcell(depth-1, x*nx, y*ny, z*nz, tmp);
		    if(tmp.min < mcell.min)
			mcell.min=tmp.min;
		    if(tmp.max > mcell.max)
			mcell.max=tmp.max;
		    int idx=mxidx[x]+myidx[y]+mzidx[z];
		    mcells[idx]=tmp;
		}
	    }
	}
    }
}

void HVolumeBrick16::parallel_calc_mcell(int)
{
    int ny=ysize[depth-1];
    int nz=zsize[depth-1];
    int nnx=xsize[depth-2];
    int nny=ysize[depth-2];
    int nnz=zsize[depth-2];
    VMCell16* mcells=macrocells[depth-1];
    int* mxidx=macrocell_xidx[depth-1];
    int* myidx=macrocell_yidx[depth-1];
    int* mzidx=macrocell_zidx[depth-1];
    int s, e;
    while(work->nextAssignment(s, e)){
	for(int block=s;block<e;block++){
	    int z=block%nz;
	    int y=(block%(nz*ny))/nz;
	    int x=(block/(ny*nz));
	    VMCell16 tmp;
	    calc_mcell(depth-2, x*nnx, y*nny, z*nnz, tmp);
	    int idx=mxidx[x]+myidx[y]+mzidx[z];
	    mcells[idx]=tmp;
	}
    }
}

void HVolumeBrick16::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend(min-Vector(offset,offset,offset));
    bbox.extend(min+datadiag+Vector(offset,offset,offset));
}

namespace rtrt {
extern int HitCell(const Ray& r, const Point& pmin, const Point& pmax, 
		   float rho[2][2][2], float iso, double tmin, double tmax, double& t);
extern Vector GradientCell(const Point& pmin, const Point& pmax,
			   const Point& p, float rho[2][2][2]);
} // end namespace rtrt

void HVolumeBrick16::isect(int depth, float isoval, double t,
			   double dtdx, double dtdy, double dtdz,
			   double next_x, double next_y, double next_z,
			   int ix, int iy, int iz,
			   int dix_dx, int diy_dy, int diz_dz,
			   int startx, int starty, int startz,
			   const Vector& cellcorner, const Vector& celldir,
			   const Ray& ray, HitInfo& hit,
			   DepthStats* st, PerProcessorContext* ppc)
{
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
		short rhos[8];
		int idx000=xidx[gx]+yidx[gy]+zidx[gz];
		rhos[0]=blockdata[idx000];
		int idx001=xidx[gx]+yidx[gy]+zidx[gz+1];
		rhos[1]=blockdata[idx001];
		int idx010=xidx[gx]+yidx[gy+1]+zidx[gz];
		rhos[2]=blockdata[idx010];
		int idx011=xidx[gx]+yidx[gy+1]+zidx[gz+1];
		rhos[3]=blockdata[idx011];
		int idx100=xidx[gx+1]+yidx[gy]+zidx[gz];
		rhos[4]=blockdata[idx100];
		int idx101=xidx[gx+1]+yidx[gy]+zidx[gz+1];
		rhos[5]=blockdata[idx101];
		int idx110=xidx[gx+1]+yidx[gy+1]+zidx[gz];
		rhos[6]=blockdata[idx110];
		int idx111=xidx[gx+1]+yidx[gy+1]+zidx[gz+1];
		rhos[7]=blockdata[idx111];
		short min=rhos[0];
		short max=rhos[0];
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
					int idx_px=xidx[gx+dx+1]+yidx[gy+dy]+zidx[gz+dz];
					int idx_mx=xidx[gx+dx-1]+yidx[gy+dy]+zidx[gz+dz];
					short ddx=blockdata[idx_px]-blockdata[idx_mx];

					int idx_py=xidx[gx+dx]+yidx[gy+dy+1]+zidx[gz+dz];
					int idx_my=xidx[gx+dx]+yidx[gy+dy-1]+zidx[gz+dz];
					short ddy=blockdata[idx_py]-blockdata[idx_my];

					int idx_pz=xidx[gx+dx]+yidx[gy+dy]+zidx[gz+dz+1];
					int idx_mz=xidx[gx+dx]+yidx[gy+dy]+zidx[gz+dz-1];
					short ddz=blockdata[idx_pz]-blockdata[idx_mz];

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
			    hit.min_t+=3/Dot(n, ray.direction());
				
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
	VMCell16* mcells=macrocells[depth];
	int* mxidx=macrocell_xidx[depth];
	int* myidx=macrocell_yidx[depth];
	int* mzidx=macrocell_zidx[depth];
	for(;;){
	    int gx=startx+ix;
	    int gy=starty+iy;
	    int gz=startz+iz;
	    int idx=mxidx[gx]+myidx[gy]+mzidx[gz];
	    VMCell16& mcell=mcells[idx];
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

void HVolumeBrick16::intersect(Ray& ray, HitInfo& hit,
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
    int cx=xsize[depth-1];
    int cy=ysize[depth-1];
    int cz=zsize[depth-1];
    int ix=(int)(s.x()*cx);
    int iy=(int)(s.y()*cy);
    int iz=(int)(s.z()*cz);
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

    /*
    if (drand48()<0.0001) {
      cerr << "I WHERE : " << orig << endl;
      cerr << "I MIN : " << min << endl;
      cerr << "I MAX : " << max << endl;
      cerr << "I cx,cy,cz : " << cx << "," << cy << "," << cz << endl;
      cerr << "I s : " << s << endl;
      cerr << "I ix,iy,iz : " << ix << "," << iy << "," << iz << endl;
      cerr << "I CELLCORNER : " << cellcorner << endl;
      cerr << "I CELLDIR : " << celldir << endl;
    }
    */

    isect(depth-1, isoval, t, dtdx, dtdy, dtdz, next_x, next_y, next_z,
	  ix, iy, iz, dix_dx, diy_dy, diz_dz,
	  0, 0, 0,
	  cellcorner, celldir,
	  ray, hit, st, ppc);    
}

Vector HVolumeBrick16::normal(const Point&, const HitInfo& hit)
{
    // We computed the normal at intersect time and tucked it
    // away in the scratchpad...
    Vector* n=(Vector*)hit.scratchpad;
    return *n;
}

void HVolumeBrick16::compute_hist(int nhist, int* hist,
				  float datamin, float datamax)
{
    char buf[200];
    sprintf(buf, "%s.hist_%d", filebase, nhist);
    ifstream in(buf);
    if(in){
	for(int i=0;i<nhist;i++){
	    in >> hist[i];
	}
    } else {
	float scale=(nhist-1)/(datamax-datamin);
	int nx1=nx-1;
	int ny1=ny-1;
	int nz1=nz-1;
	int nynz=ny*nz;
	for(int ix=0;ix<nx1;ix++){
	    for(int iy=0;iy<ny1;iy++){
		int idx=ix*nynz+iy*nz;
		for(int iz=0;iz<nz1;iz++){
		    int idx000=xidx[ix]+yidx[iy]+zidx[iz];
		    short p000=blockdata[idx000];
		    int idx001=xidx[ix]+yidx[iy]+zidx[iz+1];
		    short p001=blockdata[idx001];
		    int idx010=xidx[ix]+yidx[iy+1]+zidx[iz];
		    short p010=blockdata[idx010];
		    int idx011=xidx[ix]+yidx[iy+1]+zidx[iz+1];
		    short p011=blockdata[idx011];
		    int idx100=xidx[ix+1]+yidx[iy]+zidx[iz];
		    short p100=blockdata[idx100];
		    int idx101=xidx[ix+1]+yidx[iy]+zidx[iz+1];
		    short p101=blockdata[idx101];
		    int idx110=xidx[ix+1]+yidx[iy+1]+zidx[iz];
		    short p110=blockdata[idx110];
		    int idx111=xidx[ix+1]+yidx[iy+1]+zidx[iz+1];
		    short p111=blockdata[idx111];
		    short min=Min(Min(Min(p000, p001), Min(p010, p011)), Min(Min(p100, p101), Min(p110, p111)));
		    short max=Max(Max(Max(p000, p001), Max(p010, p011)), Max(Max(p100, p101), Max(p110, p111)));
		    int nmin=(int)((min-datamin)*scale);
		    int nmax=(int)((max-datamin)*scale+.999999);
		    if(nmax>=nhist)
			nmax=nhist-1;
		    if(nmin<0)
			nmin=0;
		    if(nmax>nhist)
			nmax=nhist;
		    for(int i=nmin;i<nmax;i++){
			hist[i]++;
		    }
		    idx++;
		}
	    }
	}
	ofstream out(buf);
	for(int i=0;i<nhist;i++){
	    out << hist[i] << '\n';
	}
    }
    cerr << "Done building histogram\n";
}    

void HVolumeBrick16::brickit(int proc)
{
    int nynz=ny*nz;
    int sx, ex;
    while(work->nextAssignment(sx, ex)){
	for(int x=sx;x<ex;x++){
	    io_lock_.lock();
	    cerr << "processor " << proc << ": " << x << " of " << nx-1 << "\n";
	    io_lock_.unlock();
	    for(int y=0;y<ny;y++){
		int idx=x*nynz+y*nz;
		for(int z=0;z<nz;z++){
		    short value=indata[idx];
		    int blockidx=xidx[x]+yidx[y]+zidx[z];
		    blockdata[blockidx]=value;
		    
		    idx++;
		}
	    }
	}
    }
}

void HVolumeBrick16::get_minmax(float& min, float& max)
{
    min=datamin;
    max=datamax;
}


#if 1
//returns a value from a point (ray+t) in a HVB16, or false if the point is not in it
bool HVolumeBrick16::interior_value(double& ret_val, const Ray &ray, const double t) {

  const Point where(ray.eval(t));
    
    //quick reject for out of bounds
    Point max(min+hierdiag);
    if ( 
	(where.x() < min.x()) ||
	(where.y() < min.y()) ||
	(where.z() < min.z()) ||
	(where.x() > max.x()) ||
	(where.y() > max.y()) ||
	(where.z() > max.z())
	) return false;

    //how many cells in the next depth down?
    int cx=xsize[depth-1];
    int cy=ysize[depth-1];
    int cz=zsize[depth-1];
    double icx=ixsize[depth-1];
    double icy=iysize[depth-1];
    double icz=izsize[depth-1];

    //use the ratio out of the current level's size to
    //find the cell for the point in the next level down 
    Vector s((where-min)*ihierdiag); 
    int ix=(int)(s.x()*cx); 
    int iy=(int)(s.y()*cy);
    int iz=(int)(s.z()*cz);

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

    //find the starting position of the containing cell in the next level down
    double ccx = min.x() + ((double)ix*hierdiag.x()*icx);
    double ccy = min.y() + ((double)iy*hierdiag.y()*icy);
    double ccz = min.z() + ((double)iz*hierdiag.z()*icz);
    Vector cellcorner(ccx, ccy, ccz);

    //how big is each cell, used to make the ratio in the next level down
    double ixs = ihierdiag.x()*cx;
    double iys = ihierdiag.y()*cy;
    double izs = ihierdiag.z()*cz;

    /*
    bool dbgprint = false;
    if (drand48()<0.001) {
      cerr << depth << " WHERE : " << where << endl;
      cerr << depth << " MIN : " << min << endl;
      cerr << depth << " MAX : " << max << endl;
      cerr << depth << " cx,cy,cz : " << cx << "," << cy << "," << cz << endl;
      cerr << depth << " s : " << s << endl;
      cerr << depth << " ix,iy,iz : " << ix << "," << iy << "," << iz << endl;
      cerr << depth << " ixs,iys,izs : " << ixs << "," << iys << "," << izs << endl;
      cerr << depth << " CELLCORNER : " << cellcorner << endl;
      dbgprint = true;
    }
    */

    //step down to the next level
    return interior_value_sublevels(ret_val,
				    where,
				    depth-1,                //level of the data hierachy tree start at
				    ix, iy, iz,             //starting cell indexes in hierarchy
				    cellcorner,
				    ixs, iys, izs
				    );    
    
}
#else
bool HVolumeBrick16::interior_value(double& ret_val, const Ray &ray, const double t) {

  const Point where(ray.eval(t));
	for(int x=sx;x<ex;x++){
	    io_lock_.lock();
	    cerr << "processor " << proc << ": " << x << " of " << nx-1 << "\n";
	    io_lock_.unlock();
	    for(int y=0;y<ny;y++){
		int idx=x*nynz+y*nz;
		for(int z=0;z<nz;z++){
		    short value=indata[idx];
		    int blockidx=xidx[x]+yidx[y]+zidx[z];
		    blockdata[blockidx]=value;
		    
		    idx++;
                }
            }
        }
}

#endif
//helper for interior_value returns a value from a point (ray+t) in a sub level of
//an HVB16, or false if the point is not in it
bool HVolumeBrick16::interior_value_sublevels(double &ret_val, //value to be returned
					      const Point & where, //what point to look for
					      int depth, //depth in H to look at
					      int ix, int iy, int iz, //cell to look in 
					      const Vector & cellcorner, //where is the cell in space
					      double ixs, double iys, double izs, //how big is the cell in space
					      bool dbgprint //turn on for prints
					      ) 
{
  if (depth==0) {
    //at the lowest level, just index into the blockdata and return the value

    /*
    if (dbgprint) {
      cerr << depth << " ix,iy,iz: " << ix << "," << iy << "," << iz << endl;
      cerr << depth << " nx,ny,nz: " << nx << "," << ny << "," << nz << endl;
      cerr << depth << " ixs,iys,izs: " << ixs << "," << iys << "," << izs << endl;
      cerr << depth << " cellcorner: " << cellcorner << endl;
      cerr << depth << " where: " << where << endl;
    }
    */

    if (ix<nx-1 && iy<ny-1 && iz<nz-1) {
      //everything kosher, return it

      //#define KLUGE
#ifdef KLUGE
	//slightly faster, and less acurate to return the avg of the voxel
#undef KLUGE
#define one_div_eight 0.125

      float avg = 0.0;

      int idx000=xidx[ix]+yidx[iy]+zidx[iz];
      avg+=blockdata[idx000];
      int idx001=xidx[ix]+yidx[iy]+zidx[iz+1];
      avg+=blockdata[idx001];
      int idx010=xidx[ix]+yidx[iy+1]+zidx[iz];
      avg+=blockdata[idx010];
      int idx011=xidx[ix]+yidx[iy+1]+zidx[iz+1];
      avg+=blockdata[idx011];
      int idx100=xidx[ix+1]+yidx[iy]+zidx[iz];
      avg+=blockdata[idx100];
      int idx101=xidx[ix+1]+yidx[iy]+zidx[iz+1];
      avg+=blockdata[idx101];
      int idx110=xidx[ix+1]+yidx[iy+1]+zidx[iz];
      avg+=blockdata[idx110];
      int idx111=xidx[ix+1]+yidx[iy+1]+zidx[iz+1];
      avg+=blockdata[idx111];

      avg *= one_div_eight;

#undef one_div_eight

      //scale and bias to 0..1
      avg -= datamin;
      avg /= (datamax - datamin);
      ret_val = avg;

#else
      
      //slightly more accurate to do trilinear interpolation
      //it's more subject to aliasing as well
      int idx000=xidx[ix]+yidx[iy]+zidx[iz];
      double b000=blockdata[idx000];
      int idx001=xidx[ix]+yidx[iy]+zidx[iz+1];
      double b001=blockdata[idx001];
      int idx010=xidx[ix]+yidx[iy+1]+zidx[iz];
      double b010=blockdata[idx010];
      int idx011=xidx[ix]+yidx[iy+1]+zidx[iz+1];
      double b011=blockdata[idx011];
      int idx100=xidx[ix+1]+yidx[iy]+zidx[iz];
      double b100=blockdata[idx100];
      int idx101=xidx[ix+1]+yidx[iy]+zidx[iz+1];
      double b101=blockdata[idx101];
      int idx110=xidx[ix+1]+yidx[iy+1]+zidx[iz];
      double b110=blockdata[idx110];
      int idx111=xidx[ix+1]+yidx[iy+1]+zidx[iz+1];
      double b111=blockdata[idx111];

      double u = (where.x()-cellcorner.x())/ixs;
      double v = (where.y()-cellcorner.y())/iys;
      double w = (where.z()-cellcorner.z())/izs;

      double u00 = (1-u)*b000 + u*b100;
      double u01 = (1-u)*b001 + u*b101;
      double u10 = (1-u)*b010 + u*b110;
      double u11 = (1-u)*b011 + u*b111;
      
      double v0 = (1-v)*u00 + v*u10;
      double v1 = (1-v)*u01 + v*u11;

      ret_val = (1-w)*v0+w*v1;
      ret_val -= datamin;
      ret_val /= (datamax - datamin);

#endif

      return true;

    } else {
      
      //boundary problem, (happens in overlapping vfem cells), just return
      return false;

    }

  } else {

    //there are more levels below, follow the same procedure as in interior_value
    //but this time, use the smaller grid cells at this level
    //how many cells at the next level, per one cell in this
    int new_cx=xsize[depth-1];
    int new_cy=ysize[depth-1];
    int new_cz=zsize[depth-1];
    
    //find the index of the next levels cell, based on where, the current cell's 
    //postition, and it's size
    int new_ix = (int) (((where.x()-cellcorner.x())*ixs)*(double)new_cx);
    int new_iy = (int) (((where.y()-cellcorner.y())*iys)*(double)new_cy);
    int new_iz = (int) (((where.z()-cellcorner.z())*izs)*(double)new_cz);
       
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

    //use the integer index to find where the next level's containing cell is
    double new_ccx = cellcorner.x() + ((double)new_ix/(ixs*(double)new_cx));
    double new_ccy = cellcorner.y() + ((double)new_iy/(iys*(double)new_cy));
    double new_ccz = cellcorner.z() + ((double)new_iz/(izs*(double)new_cz));
    Vector new_cellcorner(new_ccx, new_ccy, new_ccz);

    //the next level's cells are 1/xsize[depth-1] fraction of the size of these
    double next_ixs = ixs*new_cx;
    double next_iys = iys*new_cy;
    double next_izs = izs*new_cz;

    /*
    if (dbgprint) {
      cerr << depth << " WHERE : " << where << endl;
      cerr << depth << " CELLCORNER : " << cellcorner << endl;
      cerr << depth << " new cx,cy,cz : " << new_cx << "," << new_cy << "," << new_cz << endl;
      cerr << depth << " new ix,iy,iz : " << new_ix << "," << new_iy << "," << new_iz << endl;
      cerr << depth << " next ix,iy,iz : " << ix*new_cx+new_ix << "," << iy*new_cy+new_iy << "," << iz*new_cz+new_iz << endl;
      cerr << depth << " next ixs,iys,izs : " << next_ixs << "," << next_iys << "," << next_izs << endl;
      cerr << depth << " next cellcorner : " << new_cellcorner << endl;
    }
    */

    //the ix*new_cx+new_ix accounts for the fact that each cell at this level (ix)
    //is made up of new_cx cells, and then we want to add new_ix to get to the one
    //we just found
    int next_ix = ix*new_cx+new_ix;
    int next_iy = iy*new_cy+new_iy;
    int next_iz = iz*new_cz+new_iz;

    //recurse to process the next level down
    return interior_value_sublevels(ret_val,
				    where,
				    depth-1,
				    next_ix, next_iy, next_iz,
				    new_cellcorner,
				    next_ixs, next_iys, next_izs,
				    dbgprint
			     );
    
  }
}


