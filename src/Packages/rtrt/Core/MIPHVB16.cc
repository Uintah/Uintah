
#include <Packages/rtrt/Core/MIPHVB16.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/PriorityQ.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Mutex.h>
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
  extern SCIRun::Mutex xlock;
  
struct MIPVMCell {
    float max;
};

struct MIPCell {
    int depth;
    float maxval;
    double t;

    int gx, gy, gz;

    inline MIPCell() {
	depth=-1234;
    }
    inline MIPCell(float maxval, double t)
	: maxval(maxval), t(t) {
	    depth=-1;
	    /*gx=gy=gz=-123456; */
	    //mypri=maxval*1000;
    }
    inline MIPCell(float maxval, int depth,
		   double t,
		   int gx, int gy, int gz)
	: maxval(maxval), depth(depth), t(t),
	  gx(gx), gy(gy), gz(gz) {
	      //mypri=maxval*1000;
    }
    inline float pri() const {
	return maxval;
    }
};
} // end namespace rtrt

MIPHVB16::MIPHVB16(char* filebase, int depth, int np)
  : Object(this), depth(depth), filebase(filebase), work(0)
{
    if(depth<=0)
	this->depth=depth=1;
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
	// <<<<< bigler >>>>>
	//work=WorkQueue("Bricking", nx, bnp, false, 5);
	work.refill(nx, bnp, 5);
	Parallel<MIPHVB16> phelper(this, &MIPHVB16::brickit);
	Thread::parallel(phelper, bnp, true);

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
		   
    xsize=new int[depth+1];
    ysize=new int[depth+1];
    zsize=new int[depth+1];
    xsize[depth]=ysize[depth]=zsize[depth]=1;
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
    cellscale=new Vector[depth+1];
    cellscale[0]=datadiag/Vector(nx-1, ny-1, nz-1);
    for(int i=0;i<depth;i++){
	int nx=xsize[i];
	int ny=ysize[i];
	int nz=zsize[i];
	cellscale[i+1]=cellscale[i]*Vector(nx, ny, nz);
    }
    icellscale=new Vector[depth+1];
    for(int i=0;i<=depth;i++){
	icellscale[i]=Vector(1., 1., 1.)/cellscale[i];
    }
    isdatadiag=Vector(nx-1, ny-1, nz-1)/datadiag;
    cerr << "cellscale[depth]=" << cellscale[depth] << "\n";
    cerr << "hierdiag=" << hierdiag << "\n";

    if(depth==1){
	macrocells=0;
    } else {
	macrocells=new MIPVMCell*[depth+1];
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
	MIPVMCell* p=new MIPVMCell[total_macrocells];
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
	MIPVMCell top;
	calc_mcell(depth-1, 0, 0, 0, top);
	cerr << "Min: " << top.min << ", Max: " << top.max << '\n';
#else
	int nx=xsize[depth-1];
	int ny=ysize[depth-1];
	int nz=zsize[depth-1];
	int totaltop=nx*ny*nz;
	if(np<8)
	    np=8;
	// <<<<< bigler >>>>>
	//work=WorkQueue("Building hierarchy", totaltop, np, false, 5);
	work.refill(totaltop, np, 5);
	Parallel<MIPHVB16> phelper(this, &MIPHVB16::parallel_calc_mcell);
	Thread::parallel(phelper, np, true);
#endif
	cerr << "done\n";
    }
}

MIPHVB16::MIPHVB16(MIPHVB16* share)
  : Object(this), work(0)
{
    cerr << "WARNING: MIPHVB16 share not tested\n";
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

MIPHVB16::~MIPHVB16()
{
    if(blockdata)
	delete[] blockdata;
}

void MIPHVB16::preprocess(double, int& pp_offset, int&)
{
    offset=pp_offset;
    pp_offset+=sizeof(double);
}

void MIPHVB16::calc_mcell(int depth, int startx, int starty, int startz,
			  MIPVMCell& mcell)
{
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
		    if(max>mcell.max)
			mcell.max=max;
		}
	    }
	}
    } else {
	int nx=xsize[depth-1];
	int ny=ysize[depth-1];
	int nz=zsize[depth-1];
	MIPVMCell* mcells=macrocells[depth];
	int* mxidx=macrocell_xidx[depth];
	int* myidx=macrocell_yidx[depth];
	int* mzidx=macrocell_zidx[depth];
	for(int x=startx;x<endx;x++){
	    for(int y=starty;y<endy;y++){
		for(int z=startz;z<endz;z++){
		    MIPVMCell tmp;
		    calc_mcell(depth-1, x*nx, y*ny, z*nz, tmp);
		    if(tmp.max > mcell.max)
			mcell.max=tmp.max;
		    int idx=mxidx[x]+myidx[y]+mzidx[z];
		    mcells[idx]=tmp;
		}
	    }
	}
    }
}

void MIPHVB16::parallel_calc_mcell(int)
{
    int ny=ysize[depth-1];
    int nz=zsize[depth-1];
    int nnx=xsize[depth-2];
    int nny=ysize[depth-2];
    int nnz=zsize[depth-2];
    MIPVMCell* mcells=macrocells[depth-1];
    int* mxidx=macrocell_xidx[depth-1];
    int* myidx=macrocell_yidx[depth-1];
    int* mzidx=macrocell_zidx[depth-1];
    int s, e;
    while(work.nextAssignment(s, e)){
	for(int block=s;block<e;block++){
	    int z=block%nz;
	    int y=(block%(nz*ny))/nz;
	    int x=(block/(ny*nz));
	    MIPVMCell tmp;
	    calc_mcell(depth-2, x*nnx, y*nny, z*nnz, tmp);
	    int idx=mxidx[x]+myidx[y]+mzidx[z];
	    mcells[idx]=tmp;
	}
    }
}

void 
MIPHVB16::io(SCIRun::Piostream &str)
{
  ASSERTFAIL("Pio for MIPHVB16 not implemented");
}

void MIPHVB16::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend(min-Vector(offset,offset,offset));
    bbox.extend(min+datadiag+Vector(offset,offset,offset));
}

void MIPHVB16::intersect(Ray& ray, HitInfo& hit,
			 DepthStats*, PerProcessorContext* ppc)
{
    const Vector dir(ray.direction());
    const Point orig(ray.origin());
    PriorityQ<MIPCell, 200> curr;
    curr.insert(MIPCell(datamax, depth-1, 0,
			0, 0, 0));
    double xinv_dir=1./dir.x();
    double yinv_dir=1./dir.y();
    double zinv_dir=1./dir.z();

    float maxsofar=-MAXFLOAT;
    while(curr.length()>0){
	MIPCell cell(curr.pop());
	if(cell.depth==-1){
	    // done...
	    if(hit.hit(this, cell.t)){
#if 1
		double* tlast=(double*)ppc->get(offset, sizeof(double));
		*tlast=cell.t;
#endif
		double* val=(double*)hit.scratchpad;
		*val=cell.maxval;
	    }
	    break;
	}

	Point min(this->min+Vector(cell.gx, cell.gy, cell.gz)*cellscale[cell.depth+1]);
	Point max(min+cellscale[cell.depth+1]);
	double MIN, MAX;
	int ddx;
	int dix_dx;
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
	int ddy;
	int diy_dy;
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
	    continue;
	
	double z0, z1;
	int ddz;
	int diz_dz;
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
	    continue;
	double t;
	if(MIN > 1.e-6){
	    t=MIN;
	} else if(MAX > 1.e-6){
	    t=0;
	} else {
	    continue;
	}
#if 1
	if(cell.depth == depth-1){
	    double tlast=*(double*)ppc->get(offset, sizeof(double));
	    if(tlast < MIN || tlast > MAX){
		double tmid=(MAX+MIN)*0.5;
		tlast=tmid;
	    }
	    Vector pmid((orig+dir*tlast-min)*isdatadiag);
	    int gx=(int)pmid.x();
	    int gy=(int)pmid.y();
	    int gz=(int)pmid.z();
	    if(gx<nx-1 && gy<ny-1 && gz<nz-1 && gx>=0 && gy>=0 && gz>=0){
		int idx000=xidx[gx]+yidx[gy]+zidx[gz];
		short rho000=blockdata[idx000];
		int idx001=xidx[gx]+yidx[gy]+zidx[gz+1];
		short rho001=blockdata[idx001];
		int idx010=xidx[gx]+yidx[gy+1]+zidx[gz];
		short rho010=blockdata[idx010];
		int idx011=xidx[gx]+yidx[gy+1]+zidx[gz+1];
		short rho011=blockdata[idx011];
		int idx100=xidx[gx+1]+yidx[gy]+zidx[gz];
		short rho100=blockdata[idx100];
		int idx101=xidx[gx+1]+yidx[gy]+zidx[gz+1];
		short rho101=blockdata[idx101];
		int idx110=xidx[gx+1]+yidx[gy+1]+zidx[gz];
		short rho110=blockdata[idx110];
		int idx111=xidx[gx+1]+yidx[gy+1]+zidx[gz+1];
		short rho111=blockdata[idx111];
		float xf=pmid.x()-gx;
		float yf=pmid.y()-gy;
		float zf=pmid.z()-gz;
		float val00=rho000*(1-zf)+rho001*zf;
		float val01=rho010*(1-zf)+rho011*zf;
		float val10=rho100*(1-zf)+rho101*zf;
		float val11=rho110*(1-zf)+rho111*zf;
		float val0=val00*(1-yf)+val01*yf;
		float val1=val10*(1-yf)+val11*yf;
		maxsofar=val0*(1-xf)+val1*xf;
                if(hit.hit(this, MAX)){
                   double* val=(double*)hit.scratchpad;
                   *val=maxsofar;
		}
	    }
	}
#endif
	Point p(orig+dir*t);
	Vector s((p-min)*icellscale[cell.depth+1]);
	int cx=xsize[cell.depth];
	int cy=ysize[cell.depth];
	int cz=zsize[cell.depth];
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
	Vector& cellsize=cellscale[cell.depth];
	double x=min.x()+cellsize.x()*double(ix+ddx);
	next_x=(x-orig.x())*xinv_dir;
	dtdx=dix_dx*cellsize.x()*xinv_dir;
	double y=min.y()+cellsize.y()*double(iy+ddy);
	next_y=(y-orig.y())*yinv_dir;
	dtdy=diy_dy*cellsize.y()*yinv_dir;
	double z=min.z()+cellsize.z()*double(iz+ddz);
	next_z=(z-orig.z())*zinv_dir;
	dtdz=diz_dz*cellsize.z()*zinv_dir;

	if(cell.depth==0){
	    int startx=cell.gx*xsize[cell.depth];
	    int starty=cell.gy*ysize[cell.depth];
	    int startz=cell.gz*zsize[cell.depth];

	    int gx=startx+ix;
	    int gy=starty+iy;
	    int gz=startz+iz;

	    if(gx>=nx-1 || gy>=ny-1 || gz>=nz-1)
		continue;

	    int idx000=xidx[gx]+yidx[gy]+zidx[gz];
	    short rho000=blockdata[idx000];
	    int idx001=xidx[gx]+yidx[gy]+zidx[gz+1];
	    short rho001=blockdata[idx001];
	    int idx010=xidx[gx]+yidx[gy+1]+zidx[gz];
	    short rho010=blockdata[idx010];
	    int idx011=xidx[gx]+yidx[gy+1]+zidx[gz+1];
	    short rho011=blockdata[idx011];
	    int idx100=xidx[gx+1]+yidx[gy]+zidx[gz];
	    short rho100=blockdata[idx100];
	    int idx101=xidx[gx+1]+yidx[gy]+zidx[gz+1];
	    short rho101=blockdata[idx101];
	    int idx110=xidx[gx+1]+yidx[gy+1]+zidx[gz];
	    short rho110=blockdata[idx110];
	    int idx111=xidx[gx+1]+yidx[gy+1]+zidx[gz+1];
	    short rho111=blockdata[idx111];
	    Point cellp(orig+dir*t);
	    Vector p((cellp-this->min)*isdatadiag);
	    float xf=p.x()-gx;
	    float yf=p.y()-gy;
	    float zf=p.z()-gz;
	    float val00=rho000*(1-zf)+rho001*zf;
	    float val01=rho010*(1-zf)+rho011*zf;
	    float val10=rho100*(1-zf)+rho101*zf;
	    float val11=rho110*(1-zf)+rho111*zf;
	    float val0=val00*(1-yf)+val01*yf;
	    float val1=val10*(1-yf)+val11*yf;
	    float maxval=val0*(1-xf)+val1*xf;
	    double maxt=t;
	    for(;;){
		if(next_x < next_y && next_x < next_z){
		    // Step in x...
		    t=next_x;
		    next_x+=dtdx;
		    ix+=dix_dx;
		    if(ix<0 || ix>=cx || startx+ix >= nx-1)
			break;
		    int gx=startx+ix;
		    int gy=starty+iy;
		    int gz=startz+iz;
		    // Interpolate in the YZ plane
		    Point cellp(orig+dir*t);
		    Vector p((cellp-this->min)*isdatadiag);
		    float yf=p.y()-gy;
		    float zf=p.z()-gz;
		    int idx000=xidx[gx]+yidx[gy]+zidx[gz];
		    short rho000=blockdata[idx000];
		    int idx001=xidx[gx]+yidx[gy]+zidx[gz+1];
		    short rho001=blockdata[idx001];
		    int idx010=xidx[gx]+yidx[gy+1]+zidx[gz];
		    short rho010=blockdata[idx010];
		    int idx011=xidx[gx]+yidx[gy+1]+zidx[gz+1];
		    short rho011=blockdata[idx011];
		    float val0=rho000*(1-yf)+rho010*yf;
		    float val1=rho001*(1-yf)+rho011*yf;
		    float val=val0*(1-zf)+val1*zf;
		    if(val>maxval){
			maxval=val;
			maxt=t;
		    }
		} else if(next_y < next_z){
		    t=next_y;
		    next_y+=dtdy;
		    iy+=diy_dy;
		    if(iy<0 || iy>=cy || starty+iy >= ny-1)
			break;
		    int gx=startx+ix;
		    int gy=starty+iy;
		    int gz=startz+iz;
		    // Interpolate in the XZ plane
		    Point cellp(orig+dir*t);
		    Vector p((cellp-this->min)*isdatadiag);
		    float xf=p.x()-gx;
		    float zf=p.z()-gz;
		    int idx000=xidx[gx]+yidx[gy]+zidx[gz];
		    short rho000=blockdata[idx000];
		    int idx001=xidx[gx]+yidx[gy]+zidx[gz+1];
		    short rho001=blockdata[idx001];
		    int idx100=xidx[gx+1]+yidx[gy]+zidx[gz];
		    short rho100=blockdata[idx100];
		    int idx101=xidx[gx+1]+yidx[gy]+zidx[gz+1];
		    short rho101=blockdata[idx101];
		    float val0=rho000*(1-xf)+rho100*xf;
		    float val1=rho001*(1-xf)+rho101*xf;
		    float val=val0*(1-zf)+val1*zf;
		    if(val>maxval){
			maxval=val;
			maxt=t;
		    }
		} else {
		    t=next_z;
		    next_z+=dtdz;
		    iz+=diz_dz;
		    if(iz<0 || iz>=cz || startz+iz >= nz-1)
			break;
		    int gx=startx+ix;
		    int gy=starty+iy;
		    int gz=startz+iz;
		    // Interpolate in the XY plane
		    Point cellp(orig+dir*t);
		    Vector p((cellp-this->min)*isdatadiag);
		    float xf=p.x()-gx;
		    float yf=p.y()-gy;
		    int idx000=xidx[gx]+yidx[gy]+zidx[gz];
		    short rho000=blockdata[idx000];
		    int idx010=xidx[gx]+yidx[gy+1]+zidx[gz];
		    short rho010=blockdata[idx010];
		    int idx100=xidx[gx+1]+yidx[gy]+zidx[gz];
		    short rho100=blockdata[idx100];
		    int idx110=xidx[gx+1]+yidx[gy+1]+zidx[gz];
		    short rho110=blockdata[idx110];
		    float val0=rho000*(1-yf)+rho010*yf;
		    float val1=rho100*(1-yf)+rho110*yf;
		    float val=val0*(1-xf)+val1*xf;
		    if(val>maxval){
			maxval=val;
			maxt=t;
		    }
		}
	    }
	    if(maxval > maxsofar){
		maxsofar=maxval;
		curr.insert(MIPCell(maxval, maxt));
	    }
	} else {
	    int startx=cell.gx*xsize[cell.depth];
	    int starty=cell.gy*ysize[cell.depth];
	    int startz=cell.gz*zsize[cell.depth];
	    double t=cell.t;
	    MIPVMCell* mcells=macrocells[cell.depth];
	    int* mxidx=macrocell_xidx[cell.depth];
	    int* myidx=macrocell_yidx[cell.depth];
	    int* mzidx=macrocell_zidx[cell.depth];
	    for(;;){
		int gx=startx+ix;
		int gy=starty+iy;
		int gz=startz+iz;
		int idx=mxidx[gx]+myidx[gy]+mzidx[gz];
		MIPVMCell& mcell=mcells[idx];
		if(mcell.max > maxsofar){
		    curr.insert(MIPCell(mcell.max, cell.depth-1, t,
					gx, gy, gz));
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
	}
    }
}

Vector MIPHVB16::normal(const Point&, const HitInfo& hit)
{
    // We computed the normal at intersect time and tucked it
    // away in the scratchpad...
    Vector* n=(Vector*)hit.scratchpad;
    return *n;
}

void MIPHVB16::brickit(int proc)
{
    int nynz=ny*nz;
    int sx, ex;
    while(work.nextAssignment(sx, ex)){
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

void MIPHVB16::shade(Color& result, const Ray&,
		     const HitInfo& hit, int,
		     double, const Color&,
		     Context*)
{
    double* val=(double*)hit.scratchpad;
    double nval=*val/datamax;
    result=Color(nval, nval, nval);
}
