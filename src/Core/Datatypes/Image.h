
/*
 *  Image.h
 *
 *  Written by:
 *   Author: ?
 *   Sourced from MeshPort.cc by David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef SCI_DATATYPES_IMAGE_H
#define SCI_DATATYPES_IMAGE_H 1

#include <SCICore/CoreDatatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/Array2.h>
#include <SCICore/Geom/Color.h>

namespace SCICore {
namespace CoreDatatypes {

using SCICore::Containers::LockingHandle;
using SCICore::Containers::Array2;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::GeomSpace::Color;

class Image;
typedef LockingHandle<Image> ImageHandle;

class SCICORESHARE Image : public Datatype {
    /* Complex... */
public:
    float** rows;
    int xr, yr;
    Image(int xres, int yres);
    Image(const Image&);
    virtual ~Image();
    int xres() const;
    int yres() const;

    inline float getr(int x, int y) {
	return rows[y][x*2];
    }
    inline void set(int x, int y, float r, float i) {
	rows[y][x*2]=r;
	rows[y][x*2+1]=i;
    }
    float max_abs();

    virtual Image* clone();

    // Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};

class SCICORESHARE ColorImage {
public:
    ColorImage(int xres, int yres);
    ~ColorImage();
    Array2<Color> imagedata;
    inline Color& get_pixel(int x, int y) {
	return imagedata(y,x);
    }
    inline void put_pixel(int x, int y, const Color& pixel) {
	imagedata(y,x)=pixel;
    }
    int xres() const;
    int yres() const;
};

class SCICORESHARE DepthImage {
public:
    DepthImage(int xres, int yres);
    ~DepthImage();
    Array2<double> depthdata;
    double get_depth(int x, int y) {
	return depthdata(y,x);
    }
    inline void put_pixel(int x, int y, double depth) {
	depthdata(y,x)=depth;
    }
    int xres() const;
    int yres() const;
};

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:46  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:22  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:47  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:38  dav
// added SCICore .h files to /include directories
//
// Revision 1.1  1999/04/25 04:07:07  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:47  dav
// Import sources
//
//

#endif
