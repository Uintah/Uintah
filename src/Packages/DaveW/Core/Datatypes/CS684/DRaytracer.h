/*
 *  DRaytracer.cc:  Project parallel rays at a sphere and see where they go
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_DaveW_Datatypes_DRaytracer_h
#define SCI_DaveW_Datatypes_DRaytracer_h 1

#include <DaveW/Datatypes/CS684/ImageR.h>
#include <DaveW/Datatypes/CS684/RTPrims.h>
#include <DaveW/Datatypes/CS684/Sample2D.h>
#include <DaveW/Datatypes/CS684/Scene.h>
#include <SCICore/Containers/Array1.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/String.h>
#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Datatypes/VoidStar.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Math/Trig.h>

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace DaveW {
namespace Datatypes {

using SCICore::PersistentSpace::Piostream;

class DRaytracer : public VoidStar {
    double stepX;
    double stepY;
    Point midPt;
    Point topLeft;

    Array1<double> pixX, pixY, lensX, lensY, w;
    Array1<int> shuffle;
    Sample2D s;

    void buildTempXYZSpectra();
    void lensPixelSamples(Sample2D &, Array1<double>&, Array1<double>&,
			  Array1<double>&, Array1<double>&, Array1<double>&,
			  Array1<int>&);
public:
    Array1<double> tempXSpectra;
    Array1<double> tempYSpectra;
    Array1<double> tempZSpectra;
    Color singleTrace(const Point& curr, Point &xyz, Pixel* p);
    void preRayTrace();
    void rayTrace(int minx, int maxx, int miny, int maxy, double *image,
		  unsigned char *rawImage, double *);
    void rayTrace(double *image, unsigned char *rawImage, double *);
    Color spectrumToClr(Array1<double> & s);
    Point spectrumToXYZ(Array1<double> & s);
    ImageRM* irm;
    Scene scene;
    RTCamera camera;
    int nx,ny;
    int ns;
    double specMin, specMax;
    int specNum;

    DRaytracer();
    DRaytracer(const DRaytracer& copy);
    virtual ~DRaytracer();
    virtual VoidStar* clone();
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
};
} // End namespace Datatypes
} // End namespace DaveW

#endif
