
/*
 *  VectorFieldOcean.h: Vector Fields defined on a Regular grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/VectorFieldOcean.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <Geom/Grid.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <iostream.h>
#include <fcntl.h>

#ifdef __sun
#define MMAP_TYPE char*
#else
#define MMAP_TYPE void*
#endif
#include <stdio.h>

static Persistent* maker()
{
    return scinew VectorFieldOcean("", "");
}

PersistentTypeID VectorFieldOcean::type_id("VectorFieldOcean", "VectorField", maker);

VectorFieldOcean::VectorFieldOcean(const clString& filename, const clString& depthfile)
: VectorField(OceanFile), filename(filename)
{
  nx=1280;
  ny=896;
  nz=20;
  int fd=open(filename(), O_RDONLY);
  if(fd == -1){
    perror("open");
    data=0;
    return;
  }
  struct stat buf;
  if(fstat(fd, &buf) == -1){
    perror("stat");
    data=0;
    return;
  }
  data=(float*)mmap(0, buf.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if(data == (float*)-1){
    cerr << "Error mapping file...\n";
    data=0;
  }
  close(fd);
  double depthscale=1./50;
  bmin=Point(0,0,-5200*depthscale);
  bmax=Point(1280, 896, 0.);
  cerr << "depthfile is " << depthfile << endl;
  fd=open(depthfile(), O_RDONLY);
  if(fd == -1){
    perror("open");
    data=0;
    return;
  }
  if(fstat(fd, &buf) == -1){
    perror("stat");
    data=0;
    return;
  }
  depth=(int*)mmap(0, buf.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if(depth == (int*)-1){
    cerr << "Error mapping file...\n";
    data=0;
  }
  close(fd);

  depthval.add(0.0);
  depthval.add(-25.0);
  depthval.add(-50.0);
  depthval.add(-75.0);
  depthval.add(-100.0);
  depthval.add(-135.0);
  depthval.add(-185.0);
  depthval.add(-260.0);
  depthval.add(-360.0);
  depthval.add(-510.0);
  depthval.add(-710.0);
  depthval.add(-985.0);
  depthval.add(-1335.0);
  depthval.add(-1750.0);
  depthval.add(-2200.0);
  depthval.add(-2750.0);
  depthval.add(-3300.0);
  depthval.add(-3850.0);
  depthval.add(-4400.0);
  depthval.add(-4800.0);
  depthval.add(-5200.0);
  for(int i=0;i<depthval.size();i++)
    depthval[i]*=depthscale;
}

VectorFieldOcean::~VectorFieldOcean()
{
  munmap((MMAP_TYPE)data, 1280*896*20*2*sizeof(float));
  munmap((MMAP_TYPE)depth, 1280*896*sizeof(int));
}

void VectorFieldOcean::locate(const Point& p, int& ix, int& iy, int& iz)
{
  cerr << "Locate called...\n";
    Vector pn=p-bmin;
    double dx=diagonal.x();
    double dy=diagonal.y();
    double dz=diagonal.z();
    double x=pn.x()*(nx-1)/dx;
    double y=pn.y()*(ny-1)/dy;
    double z=pn.z()*(nz-1)/dz;
    ix=(int)x;
    iy=(int)y;
    iz=(int)z;
}

#define VectorFIELDOcean_VERSION 1

void VectorFieldOcean::io(Piostream&)
{
    NOT_FINISHED("VectorFieldOcean::io");
}	

void VectorFieldOcean::compute_bounds()
{
    // Nothing to do - we store the bounds in the base class...
}

int VectorFieldOcean::interpolate(const Point& p, Vector& value, int&)
{
    return interpolate(p, value);
}

int VectorFieldOcean::interpolate(const Point& p, Vector& value)
{
    if(p.z() > depthval[0])return 0;
    if(p.z() < depthval[depthval.size()-1])return 0;
    int l=0;
    int h=depthval.size()-1;
    while(l<h-1){
      int m=(l+h)/2;
      if(p.z() < depthval[m]){
	// High road...
	l=m;
      } else {
	// Low road...
	h=m;
      }
    }
    // It is now between level l and level h...
    double z1=depthval[l];
    double z2=depthval[h];
    ASSERT(p.z() <= z1 && p.z() >= z2);
    
    Vector pn=p-bmin;
    double dx=diagonal.x();
    double dy=diagonal.y();
    double x=pn.x()*(nx-1)/dx;
    double y=pn.y()*(ny-1)/dy;
    int ix=(int)x;
    int iy=(int)y;
    int iz=l;
    if(ix<0 || ix+1>=nx)return 0;
    if(iy<0 || iy+1>=ny)return 0;
    int idx=iz*(ny*nx)+iy*nx+ix;
    double fx=x-ix;
    double fy=y-iy;
    double fz=(p.z()-z1)/(z2-z1);
    //cerr << "z=" << p.z() << ", iz=" << iz << ", fz=" << fz << endl;
    //cerr << "fz=" << fz << endl;
    double ux00=Interpolate(data[idx], data[idx+1], fx);
    double ux01=Interpolate(data[idx+ny*nx], data[idx+ny*nx+1], fx);
    double ux10=Interpolate(data[idx+nx], data[idx+nx+1], fx);
    double ux11=Interpolate(data[idx+ny*nx+nx], data[idx+ny*nx+nx+1], fx);
    idx+=nx*ny*nz;
    double vx00=Interpolate(data[idx], data[idx+1], fx);
    double vx01=Interpolate(data[idx+ny*nx], data[idx+ny*nx+1], fx);
    double vx10=Interpolate(data[idx+nx], data[idx+nx+1], fx);
    double vx11=Interpolate(data[idx+ny*nx+nx], data[idx+ny*nx+nx+1], fx);
    Vector y0=Interpolate(Vector(ux00, vx00, 0), Vector(ux10, vx10, 0), fy);
    Vector y1=Interpolate(Vector(ux01, vx01, 0), Vector(ux11, vx11, 0), fy);
    value=Interpolate(y0, y1, fz);
    return 1;
}

VectorField* VectorFieldOcean::clone()
{
    return scinew VectorFieldOcean(*this);
}
#if 0
static MaterialHandle black=scinew Material(Color(0,0,0), Color(0,0,0),
					    Color(0,0,0), 0);
static MaterialHandle white=scinew Material(Color(0,0,0), Color(.6, .6, .6),
					    Color(.6, .6, .6), 20);
#endif

GeomObj* VectorFieldOcean::makesurf(int /*downsample*/)
{
#if 0
  GeomGrid* grid=0;
new GeomGrid(1280/downsample, 896/downsample,
			      Point(0,0,0), Vector(1280,0,0), Vector(0,896,0),
			      GeomGrid::WithNormAndMatl);
  for(int j=0;j<896;j+=downsample){
    for(int i=0;i<1280;i+=downsample){
      int deep=depth[j*nx+i];
      int xoff=i+downsample>=1280?-downsample:downsample;
      int yoff=j+downsample>=896?-downsample:downsample;
      Vector u(xoff, 0, depthval[depth[j*nx+i+xoff]]-depthval[deep]);
      if(j+downsample>=1280)
	u=-u;
      Vector v(0, yoff, depthval[depth[j*nx+yoff*nx+i]]-depthval[deep]);
      if(j+downsample>=896)
	v=-v;
      Vector normal(Cross(u, v));
      normal.normalize();
      grid->set(i/downsample, j/downsample, depthval[deep], normal, deep==0?black:white);
    }
  }
  return grid;
#else
  return 0;
#endif
}

void VectorFieldOcean::get_boundary_lines(Array1<Point>&)
{
    NOT_FINISHED("VectorFieldOcean::get_boundary_lines");
}
