
/*
 *  ScalarFieldOcean.cc: float Scalar Field defined on a Regular grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <Datatypes/ScalarFieldOcean.h>
#include <Classlib/String.h>
#include <Malloc/Allocator.h>
#include <iostream.h>

static Persistent* maker()
{
    return scinew ScalarFieldOcean("");
}

PersistentTypeID ScalarFieldOcean::type_id("ScalarFieldOcean", "ScalarField", maker);

ScalarFieldOcean::ScalarFieldOcean(const clString filename)
: ScalarFieldRGBase()
{
}

ScalarFieldOcean::ScalarFieldOcean(const ScalarFieldOcean& copy)
: ScalarFieldRGBase(copy), grid(copy.grid)
{
}

ScalarFieldOcean::~ScalarFieldOcean()
{
}

void ScalarFieldOcean::io(Piostream& stream)
{
    NOT_FINISHED("ScalarFieldOcean::io");
}

void ScalarFieldOcean::compute_minmax()
{
    if(nx==0 || ny==0 || nz==0)return;
    int n=nx*ny*nz;
    double min=data[0];
    double max=data[0];
    for(int i=1;i<n;i++){
      min=Min(min, data[i]);
      max=Max(max, data[i]);
    }
    data_min=min;
    data_max=max;
}

Vector ScalarFieldOcean::gradient(const Point& p)
{
#if 0
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
#endif
}

int ScalarFieldOcean::interpolate(const Point& p, double& value, int&,
				   double epsilon1, double epsilon2) {
    return interpolate(p, value, epsilon1, epsilon2);
}

int ScalarFieldOcean::interpolate(const Point& p, double& value, double,
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

ScalarField* ScalarFieldOcean::clone()
{
    return scinew ScalarFieldOcean(*this);
}

