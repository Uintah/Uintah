/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Array2.h>
#include <Core/Datatypes/Color.h>

namespace SCIRun {


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

} // End namespace SCIRun


#endif
