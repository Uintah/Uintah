
#include <Packages/rtrt/Core/VolumeBrick16.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <stdio.h>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

using namespace rtrt;
using namespace std;

VolumeBrick16::VolumeBrick16(Material* matl, VolumeDpy* dpy, char* filebase)
    : VolumeBase(matl, dpy)
{
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
    diag=max-min;
    sdiag=diag/Vector(nx-1,ny-1,nz-1);

    //    ifstream din(filebase);
    int din_fd = open (filebase, O_RDONLY);
    if(din_fd == -1) {
	cerr << "Error opening data file: " << filebase << '\n';
	exit(1);
    }
    indata=new short[nx*ny*nz];
    if(!indata){
	cerr << "Error allocating data array\n";
	exit(1);
    }
    //din.read((char*)data, sizeof(short)*nx*ny*nz);
    //    read(din.rdbuf()->fd(), indata, sizeof(short)*nx*ny*nz);
    read(din_fd, indata, sizeof(short)*nx*ny*nz);
    int s = close (din_fd);
    if(s == -1) {
	cerr << "Error reading data file: " << filebase << '\n';
	exit(1);
    }
}

VolumeBrick16::~VolumeBrick16()
{
    if(indata)
	delete[] indata;
    if(blockdata)
	delete[] blockdata;
}

void VolumeBrick16::preprocess(double, int&, int&)
{
    int nx1=nx-1;
    int ny1=ny-1;
    int nz1=nz-1;
    int nynz=ny*nz;
    int count=0;
    xidx=new int[nx];
#define L1 4
#define L2 5
    int totalx=(nx+L2*L1-1)/(L2*L1);
    int totaly=(ny+L2*L1-1)/(L2*L1);
    int totalz=(nz+L2*L1-1)/(L2*L1);
    blockdata=new short[totalx*totaly*totalz*L2*L2*L2*L1*L1*L1];
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

    float isoval=dpy->isoval;
    for(int x=0;x<nx1;x++){
	cerr << x << " of " << nx1 << "\n";
	for(int y=0;y<ny1;y++){
	    int idx=x*nynz+y*nz;
	    for(int z=0;z<nz1;z++){
		short p000=indata[idx];
		short p001=indata[idx+1];
		short p010=indata[idx+nz];
		short p011=indata[idx+nz+1];
		short p100=indata[idx+nynz];
		short p101=indata[idx+nynz+1];
		short p110=indata[idx+nynz+nz];
		short p111=indata[idx+nynz+nz+1];
		int idx000=xidx[x]+yidx[y]+zidx[z];
		blockdata[idx000]=p000;
		int idx001=xidx[x]+yidx[y]+zidx[z+1];
		blockdata[idx001]=p001;
		int idx010=xidx[x]+yidx[y+1]+zidx[z];
		blockdata[idx010]=p010;
		int idx011=xidx[x]+yidx[y+1]+zidx[z+1];
		blockdata[idx011]=p011;
		int idx100=xidx[x+1]+yidx[y]+zidx[z];
		blockdata[idx100]=p100;
		int idx101=xidx[x+1]+yidx[y]+zidx[z+1];
		blockdata[idx101]=p101;
		int idx110=xidx[x+1]+yidx[y+1]+zidx[z];
		blockdata[idx110]=p110;
		int idx111=xidx[x+1]+yidx[y+1]+zidx[z+1];
		blockdata[idx111]=p111;
		
		float min=Min(Min(Min(p000, p001), Min(p010, p011)), Min(Min(p100, p101), Min(p110, p111)));
		float max=Max(Max(Max(p000, p001), Max(p010, p011)), Max(Max(p100, p101), Max(p110, p111)));
		if(min < isoval && max>isoval){
		    count++;
		}
		idx++;
	    }
	}
    }
    cerr << "Found " << count << " cells (isoval=" << isoval << ")\n";
}

void VolumeBrick16::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend(min-Vector(offset,offset,offset));
    bbox.extend(min+diag+Vector(offset,offset,offset));
}

Vector VolumeBrick16::normal(const Point&, const HitInfo& hit)
{
    // We computed the normal at intersect time and tucked it
    // away in the scratchpad...
    Vector* n=(Vector*)hit.scratchpad;
    return *n;
}

namespace rtrt {
extern int HitCell(const Ray& r, const Point& pmin, const Point& pmax, 
		   float rho[2][2][2], float iso, double tmin, double tmax, double& t);
extern Vector GradientCell(const Point& pmin, const Point& pmax,
			   const Point& p, float rho[2][2][2]);
} // end namespace rtrt

void VolumeBrick16::intersect(Ray& ray, HitInfo& hit,
			    DepthStats*, PerProcessorContext*)
{
    float isoval=dpy->isoval;
    Point max(min+diag);
    const Vector dir(ray.direction());
    const Point orig(ray.origin());
    double MIN, MAX;
    double inv_dir=1./dir.x();
    //int didx_dx, dix_dx;
    int dix_dx;
    int ddx;
    //int nynz=ny*nz;
    if(dir.x() > 0){
	MIN=inv_dir*(min.x()-orig.x());
	MAX=inv_dir*(max.x()-orig.x());
	//didx_dx=nynz;
	dix_dx=1;
	ddx=1;
    } else {
	MIN=inv_dir*(max.x()-orig.x());
	MAX=inv_dir*(min.x()-orig.x());
	//didx_dx=-nynz;
	dix_dx=-1;
	ddx=0;
    }	
    double y0, y1;
    //int didx_dy, diy_dy;
    int diy_dy;
    int ddy;
    if(dir.y() > 0){
	double inv_dir=1./dir.y();
	y0=inv_dir*(min.y()-orig.y());
	y1=inv_dir*(max.y()-orig.y());
	//didx_dy=nz;
	diy_dy=1;
	ddy=1;
    } else {
	double inv_dir=1./dir.y();
	y0=inv_dir*(max.y()-orig.y());
	y1=inv_dir*(min.y()-orig.y());
	//didx_dy=-nz;
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
    //int didx_dz, diz_dz;
    int diz_dz;
    int ddz;
    if(dir.z() > 0){
	double inv_dir=1./dir.z();
	z0=inv_dir*(min.z()-orig.z());
	z1=inv_dir*(max.z()-orig.z());
	//didx_dz=1;
	diz_dz=1;
	ddz=1;
    } else {
	double inv_dir=1./dir.z();
	z0=inv_dir*(max.z()-orig.z());
	z1=inv_dir*(min.z()-orig.z());
	//didx_dz=-1;
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
    int ix=(int)(s.x()*(nx-1));
    int iy=(int)(s.y()*(ny-1));
    int iz=(int)(s.z()*(nz-1));
    if(ix>=nx-1)
	ix--;
    if(iy>=ny-1)
	iy--;
    if(iz>=nz-1)
	iz--;
    if(ix<0)
	ix++;
    if(iy<0)
	iy++;
    if(iz<0)
	iz++;

    //int idx=ix*nynz+iy*nz+iz;

    double next_x, next_y, next_z;
    double dtdx, dtdy, dtdz;
    double x=min.x()+diag.x()*double(ix+ddx)/(nx-1);
    next_x=(x-orig.x())/dir.x();
    dtdx=dix_dx*diag.x()/(nx-1)/dir.x();
    double y=min.y()+diag.y()*double(iy+ddy)/(ny-1);
    next_y=(y-orig.y())/dir.y();
    dtdy=diy_dy*diag.y()/(ny-1)/dir.y();
    double z=min.z()+diag.z()*double(iz+ddz)/(nz-1);
    next_z=(z-orig.z())/dir.z();
    dtdz=diz_dz*diag.z()/(nz-1)/dir.z();
    for(;;){
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
	if(min < isoval && max>isoval){
#if 1
	    double hit_t;
	    Point p0(this->min+sdiag*Vector(ix,iy,iz));
	    Point p1(p0+sdiag);
	    double tmax=next_x;
	    if(next_y<tmax)
		tmax=next_y;
	    if(next_z<tmax)
		tmax=next_z;
	    float rho[2][2][2];
	    rho[0][0][0]=p000;
	    rho[0][0][1]=p001;
	    rho[0][1][0]=p010;
	    rho[0][1][1]=p011;
	    rho[1][0][0]=p100;
	    rho[1][0][1]=p101;
	    rho[1][1][0]=p110;
	    rho[1][1][1]=p111;
#if 0 
	    cerr << "p0=" << p0 << ", p1=" << p1 << ", ix=" << ix << ", iy=" << iy << ", iz=" << iz << '\n';
#endif
	    if(HitCell(ray, p0, p1, rho, isoval, t, tmax, hit_t)){
#else
		
	    double hit_t=t;
	    {
#endif
		if(hit.hit(this, hit_t)){
		    Vector* n=(Vector*)hit.scratchpad;
		    *n=GradientCell(p0, p1, orig+dir*hit_t, rho);
		    n->normalize();
		}
	    }
	}
	if(next_x < next_y && next_x < next_z){
	    // Step in x...
	    t=next_x;
	    next_x+=dtdx;
	    ix+=dix_dx;
	    //idx+=didx_dx;
	    if(ix<0 || ix>=nx-1)
		return;
	} else if(next_y < next_z){
	    t=next_y;
	    next_y+=dtdy;
	    iy+=diy_dy;
	    //idx+=didx_dy;
	    if(iy<0 || iy>=ny-1)
		return;
	} else {
	    t=next_z;
	    next_z+=dtdz;
	    iz+=diz_dz;
	    //idx+=didx_dz;
	    if(iz<0 || iz >=nz-1)
		return;
	}
	if(hit.min_t < t)
	    break;
    }
}

void VolumeBrick16::compute_hist(int nhist, int* hist,
			       float datamin, float datamax)
{
    float scale=(nhist-1)/(datamax-datamin);
    int nx1=nx-1;
    int ny1=ny-1;
    int nz1=nz-1;
    int nynz=ny*nz;
    for(int x=0;x<nx1;x++){
	for(int y=0;y<ny1;y++){
	    int idx=x*nynz+y*nz;
	    for(int z=0;z<nz1;z++){
		short p000=indata[idx];
		short p001=indata[idx+1];
		short p010=indata[idx+nz];
		short p011=indata[idx+nz+1];
		short p100=indata[idx+nynz];
		short p101=indata[idx+nynz+1];
		short p110=indata[idx+nynz+nz];
		short p111=indata[idx+nynz+nz+1];
		short min=Min(Min(Min(p000, p001), Min(p010, p011)), Min(Min(p100, p101), Min(p110, p111)));
		short max=Max(Max(Max(p000, p001), Max(p010, p011)), Max(Max(p100, p101), Max(p110, p111)));
		int nmin=(int)((min-datamin)*scale);
		int nmax=(int)((max-datamin)*scale+.999999);
		if(nmax>=nhist)
		    nmax=nhist-1;
		if(nmin<0)
		    nmax=0;
		if(nmax>nhist)
		    nmax=nhist;
		for(int i=nmin;i<nmax;i++){
		    hist[i]++;
		}
		idx++;
	    }
	}
    }
    cerr << "Done building histogram\n";
}

void VolumeBrick16::get_minmax(float& min, float& max)
{
    min=datamin;
    max=datamax;
}
