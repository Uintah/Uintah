
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

#include <SCICore/Datatypes/VectorFieldRGCC.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Malloc/Allocator.h>

namespace SCICore {
namespace Datatypes {

static Persistent* maker()
{
    return scinew VectorFieldRGCC;
}

PersistentTypeID VectorFieldRGCC::type_id("VectorFieldRGCC", "VectorField", maker);

VectorFieldRGCC::VectorFieldRGCC()
: VectorFieldRG()
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

void VectorFieldRGCC::locate(const Point& p, int& ix, int& iy, int& iz)
{
    Vector pn=p-bmin;
    ix=(int)(pn.x()*nx/diagonal.x());
    iy=(int)(pn.y()*ny/diagonal.y());
    iz=(int)(pn.z()*nz/diagonal.z());

}

#define VectorFIELDRGCC_VERSION 1

void VectorFieldRGCC::io(Piostream& stream)
{
    using SCICore::PersistentSpace::Pio;
    using SCICore::Containers::Pio;
    using SCICore::Geometry::Pio;

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

} // End namespace Datatypes
} // End namespace SCICore



