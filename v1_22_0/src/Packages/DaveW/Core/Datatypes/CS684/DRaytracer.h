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

#ifndef SCI_Packages_DaveW_Datatypes_DRaytracer_h
#define SCI_Packages_DaveW_Datatypes_DRaytracer_h 1

#include <Packages/DaveW/Core/Datatypes/CS684/ImageR.h>
#include <Packages/DaveW/Core/Datatypes/CS684/RTPrims.h>
#include <Packages/DaveW/Core/Datatypes/CS684/Sample2D.h>
#include <Packages/DaveW/Core/Datatypes/CS684/Scene.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/String.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/VoidStar.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/Material.h>
#include <Core/Math/Expon.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Trig.h>

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

namespace DaveW {
using namespace SCIRun;

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

} // End namespace DaveW

#endif
