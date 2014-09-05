
#include <Packages/rtrt/Core/Volume16.h>
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

Volume16::Volume16(Material* matl, VolumeDpy* dpy, char* filebase)
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
    data=new short[nx*ny*nz];
    if(!data){
	cerr << "Error allocating data array\n";
	exit(1);
    }
    //din.read((char*)data, sizeof(short)*nx*ny*nz);
    //    read(din.rdbuf()->fd(), data, sizeof(short)*nx*ny*nz);
    read(din_fd, data, sizeof(short)*nx*ny*nz);
    int s = close(din_fd);
    if(s == -1) {
	cerr << "Error reading data file: " << filebase << '\n';
	exit(1);
    }
}

Volume16::~Volume16()
{
    if(data)
	delete[] data;
}

void Volume16::preprocess(double, int&, int&)
{
    int nx1=nx-1;
    int ny1=ny-1;
    int nz1=nz-1;
    int nynz=ny*nz;
    int count=0;
    float isoval=dpy->isoval;
    for(int x=0;x<nx1;x++){
	for(int y=0;y<ny1;y++){
	    int idx=x*nynz+y*nz;
	    for(int z=0;z<nz1;z++){
		float p000=data[idx];
		float p001=data[idx+1];
		float p010=data[idx+nz];
		float p011=data[idx+nz+1];
		float p100=data[idx+nynz];
		float p101=data[idx+nynz+1];
		float p110=data[idx+nynz+nz];
		float p111=data[idx+nynz+nz+1];
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

void Volume16::compute_bounds(BBox& bbox, double offset)
{
    bbox.extend(min-Vector(offset,offset,offset));
    bbox.extend(min+diag+Vector(offset,offset,offset));
}

Vector Volume16::normal(const Point&, const HitInfo& hit)
{
    // We computed the normal at intersect time and tucked it
    // away in the scratchpad...
    Vector* n=(Vector*)hit.scratchpad;
    return *n;
}

namespace rtrt {
extern Vector GradientCell(const Point& pmin, const Point& pmax,
			   const Point& p, float rho[2][2][2]);
extern int HitCell(const Ray& r, const Point& pmin, const Point& pmax, 
		   float rho[2][2][2], float iso, double tmin, double tmax, double& t);
} // end namespace rtrt

void Volume16::intersect(Ray& ray, HitInfo& hit,
		       DepthStats*, PerProcessorContext*)
{
    Point max(min+diag);
    const Vector dir(ray.direction());
    const Point orig(ray.origin());
    float isoval=dpy->isoval;

    double MIN, MAX;
    double inv_dir=1./dir.x();
    int didx_dx, dix_dx;
    int ddx;
    int nynz=ny*nz;
    if(dir.x() >= 0){
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

    int idx=ix*nynz+iy*nz+iz;


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
	float p000=data[idx];
	float p001=data[idx+1];
	float p010=data[idx+nz];
	float p011=data[idx+nz+1];
	float p100=data[idx+nynz];
	float p101=data[idx+nynz+1];
	float p110=data[idx+nynz+nz];
	float p111=data[idx+nynz+nz+1];
	float min=Min(Min(Min(p000, p001), Min(p010, p011)), Min(Min(p100, p101), Min(p110, p111)));
	float max=Max(Max(Max(p000, p001), Max(p010, p011)), Max(Max(p100, p101), Max(p110, p111)));
	if(min <= isoval && max>isoval){
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
	    if(HitCell(ray, p0, p1, rho, isoval, t, tmax, hit_t)){
		if(hit.hit(this, hit_t)){
		    Vector* n=(Vector*)hit.scratchpad;
		    *n=GradientCell(p0, p1, orig+dir*hit_t, rho);
		    n->normalize();
		}
	    }
	}
	if(next_x <= next_y && next_x <= next_z){
	    // Step in x...

	    t=next_x;
	    next_x+=dtdx;
	    ix+=dix_dx;
	    idx+=didx_dx;
	    if(ix<0 || ix>=nx-1)
		return;
	} else if(next_y <= next_z){
	    t=next_y;
	    next_y+=dtdy;
	    iy+=diy_dy;
	    idx+=didx_dy;
	    if(iy<0 || iy>=ny-1)
		return;
	} else {
	    t=next_z;
	    next_z+=dtdz;
	    iz+=diz_dz;
	    idx+=didx_dz;
	    if(iz<0 || iz >=nz-1)
		return;
	}
	if(hit.min_t < t)
	    break;
    }
}


void Volume16::compute_hist(int nhist, int* hist, float datamin, float datamax)
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
		float p000=data[idx];
		float p001=data[idx+1];
		float p010=data[idx+nz];
		float p011=data[idx+nz+1];
		float p100=data[idx+nynz];
		float p101=data[idx+nynz+1];
		float p110=data[idx+nynz+nz];
		float p111=data[idx+nynz+nz+1];
		float min=Min(Min(Min(p000, p001), Min(p010, p011)),
			      Min(Min(p100, p101), Min(p110, p111)));
		float max=Max(Max(Max(p000, p001), Max(p010, p011)),
			      Max(Max(p100, p101), Max(p110, p111)));
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
    cerr << "Done building histogram\n";
}

void Volume16::get_minmax(float& min, float& max)
{
    min=datamin;
    max=datamax;
}

