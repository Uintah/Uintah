
/*
 *  ScalarFieldRGTYPE.cc: TYPE Scalar Field defined on a Regular grid
 *
 *  Written by:
 *   Steven G. Parker (& David Weinstein)
 *   Department of Computer Science
 *   University of Utah
 *   March 1994 (& January 1996)
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/ScalarFieldRGTYPE.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

static Persistent* maker()
{
    return scinew ScalarFieldRGTYPE;
}

PersistentTypeID ScalarFieldRGTYPE::type_id("ScalarFieldRGTYPE", "ScalarField", maker);

ScalarFieldRGTYPE::ScalarFieldRGTYPE()
: ScalarFieldRGBase("TYPE")
{
}

ScalarFieldRGTYPE::ScalarFieldRGTYPE(const ScalarFieldRGTYPE& copy)
: ScalarFieldRGBase(copy), grid(copy.grid)
{
}

ScalarFieldRGTYPE::~ScalarFieldRGTYPE()
{
}

void ScalarFieldRGTYPE::resize(int x, int y, int z) {
    nx=x; ny=y; nz=z;
    grid.newsize(x,y,z);
}

#define ScalarFieldRGTYPE_VERSION 2

void ScalarFieldRGTYPE::io(Piostream& stream)
{
    int version=stream.begin_class("ScalarFieldRGTYPE", ScalarFieldRGTYPE_VERSION);
    if(version == 1){
	// From before, when the ScalarFieldRGBase didn't exist...
	ScalarField::io(stream);

	// Save these since the ScalarField doesn't
	Pio(stream, bmin);
	Pio(stream, bmax);
	if(stream.reading()){
	    have_bounds=1;
	    diagonal=bmax-bmin;
	}

	// Save the rest..
	Pio(stream, nx);
	Pio(stream, ny);
	Pio(stream, nz);
    } else {
	// Do the TYPE class first...
	ScalarFieldRGBase::io(stream);
    }
    Pio(stream, grid);
    stream.end_class();
}

void ScalarFieldRGTYPE::compute_minmax()
{
    if(nx==0 || ny==0 || nz==0)return;
    double min=grid(0,0,0);
    double max=grid(0,0,0);
    for(int i=0;i<nx;i++){
	for(int j=0;j<ny;j++){
	    for(int k=0;k<nz;k++){
		double val=grid(i,j,k);
		min=Min(min, val);
		max=Max(max, val);
	    }
	}
    }
    data_min=min;
    data_max=max;
}

Vector ScalarFieldRGTYPE::gradient(const Point& p)
{
    Vector pn=p-bmin;
    double diagx=diagonal.x();
    double diagy=diagonal.y();
    double diagz=diagonal.z();
    double x=pn.x()*(nx-1)/diagx;
    double y=pn.y()*(ny-1)/diagy;
    double z=pn.z()*(nz-1)/diagz;
    int ix=(int)x;
    int iy=(int)y;
    int iz=(int)z;
    int ix1=ix+1;
    int iy1=iy+1;
    int iz1=iz+1;
    if(ix<0 || ix1>=nx)return Vector(0,0,0);
    if(iy<0 || iy1>=ny)return Vector(0,0,0);
    if(iz<0 || iz1>=nz)return Vector(0,0,0);
    double fx=x-ix;
    double fy=y-iy;
    double fz=z-iz;
    double z00=Interpolate(grid(ix, iy, iz), grid(ix, iy, iz1), fz);
    double z01=Interpolate(grid(ix, iy1, iz), grid(ix, iy1, iz1), fz);
    double z10=Interpolate(grid(ix1, iy, iz), grid(ix1, iy, iz1), fz);
    double z11=Interpolate(grid(ix1, iy1, iz), grid(ix1, iy1, iz1), fz);
    double yy0=Interpolate(z00, z01, fy);
    double yy1=Interpolate(z10, z11, fy);
    double dx=(yy1-yy0)*(nx-1)/diagonal.x();
    double x00=Interpolate(grid(ix, iy, iz), grid(ix1, iy, iz), fx);
    double x01=Interpolate(grid(ix, iy, iz1), grid(ix1, iy, iz1), fx);
    double x10=Interpolate(grid(ix, iy1, iz), grid(ix1, iy1, iz), fx);
    double x11=Interpolate(grid(ix, iy1, iz1), grid(ix1, iy1, iz1), fx);
    double y0=Interpolate(x00, x10, fy);
    double y1=Interpolate(x01, x11, fy);
    double dz=(y1-y0)*(nz-1)/diagonal.z();
    double z0=Interpolate(x00, x01, fz);
    double z1=Interpolate(x10, x11, fz);
    double dy=(z1-z0)*(ny-1)/diagonal.y();
    return Vector(dx, dy, dz);
}

int ScalarFieldRGTYPE::interpolate(const Point& p, double& value, int&,
				   double epsilon1, double epsilon2) {
    return interpolate(p, value, epsilon1, epsilon2);
}

int ScalarFieldRGTYPE::interpolate(const Point& p, double& value, double,
				   double)
{
    Vector pn=p-bmin;
    double dx=diagonal.x();
    double dy=diagonal.y();
    double dz=diagonal.z();
    double x=pn.x()*(nx-1)/dx;
    double y=pn.y()*(ny-1)/dy;
    double z=pn.z()*(nz-1)/dz;
    int ix=(int)x;
    int iy=(int)y;
    int iz=(int)z;
    int ix1=ix+1;
    int iy1=iy+1;
    int iz1=iz+1;
    if(ix<0 || ix1>=nx)return 0;
    if(iy<0 || iy1>=ny)return 0;
    if(iz<0 || iz1>=nz)return 0;
    double fx=x-ix;
    double fy=y-iy;
    double fz=z-iz;
    double x00=Interpolate(grid(ix, iy, iz), grid(ix1, iy, iz), fx);
    double x01=Interpolate(grid(ix, iy, iz1), grid(ix1, iy, iz1), fx);
    double x10=Interpolate(grid(ix, iy1, iz), grid(ix1, iy1, iz), fx);
    double x11=Interpolate(grid(ix, iy1, iz1), grid(ix1, iy1, iz1), fx);
    double y0=Interpolate(x00, x10, fy);
    double y1=Interpolate(x01, x11, fy);
    value=Interpolate(y0, y1, fz);
    return 1;
}

ScalarField* ScalarFieldRGTYPE::clone()
{
    return scinew ScalarFieldRGTYPE(*this);
}

#ifdef __GNUG__

#include <Classlib/Array3.cc>
template class Array3<TYPE>;
template void Pio(Piostream&, Array3<TYPE>&);

#endif


#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/Array3.cc>

static void _dummy_(Piostream& p1, Array3<TYPE>& p2)
{
    Pio(p1, p2);
}

#endif
#endif

