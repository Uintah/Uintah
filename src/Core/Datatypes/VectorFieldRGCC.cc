
/*
 *  VectorFieldRGCC.cc: Vector Fields defined on a Regular grid
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Datatypes/VectorFieldRGCC.h>
#include <Core/Containers/String.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {

static Persistent* maker()
{
    return scinew VectorFieldRGCC;
}

PersistentTypeID VectorFieldRGCC::type_id("VectorFieldRGCC", "VectorField", maker);

VectorFieldRGCC::VectorFieldRGCC()
: VectorFieldRG(0, 0, 0)
{
}

VectorFieldRGCC::~VectorFieldRGCC()
{
}

Point VectorFieldRGCC::get_point(int i, int j, int k)
{
    double x=bmin.x()+diagonal.x()*double(i)/double(nx);
    double y=bmin.y()+diagonal.y()*double(j)/double(ny);
    double z=bmin.z()+diagonal.z()*double(k)/double(nz);
    return Point(x,y,z);
}

bool VectorFieldRGCC::locate(int *loc, const Point& p)
{
    Vector pn=p-bmin;
    loc[0] = (int)(pn.x()*nx/diagonal.x());
    loc[1] = (int)(pn.y()*ny/diagonal.y());
    loc[2] = (int)(pn.z()*nz/diagonal.z());

    if (loc[0] < 0 || loc[0] >= nx ||
	loc[1] < 0 || loc[1] >= ny ||
	loc[2] < 0 || loc[2] >= nz)
    {
      return false;
    }
    else
    {
      return true;
    }
}

#define VectorFIELDRGCC_VERSION 1

void VectorFieldRGCC::io(Piostream& stream)
{

    /*int version=*/
    stream.begin_class("VectorFieldRGCC", VectorFIELDRGCC_VERSION);

    // Do the base class first...
    VectorField::io(stream);

    // Save these since the VectorField doesn't
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
    Pio(stream, grid);
    stream.end_class();
}	

int VectorFieldRGCC::interpolate(const Point& p, Vector& value, int&, int)
{
    return interpolate(p, value);
}

int VectorFieldRGCC::interpolate(const Point& p, Vector& value)
{
    Vector pn=p-bmin;
    int ix=(int)(pn.x()*nx/diagonal.x());
    int iy=(int)(pn.y()*ny/diagonal.y());
    int iz=(int)(pn.z()*nz/diagonal.z());
    if(ix<0 || ix>=nx)return 0;
    if(iy<0 || iy>=ny)return 0;
    if(iz<0 || iz>=nz)return 0;
    value=grid(ix, iy, iz);
    return 1;
}


VectorField* VectorFieldRGCC::clone()
{
    return scinew VectorFieldRGCC(*this);
}

} // End namespace SCIRun



