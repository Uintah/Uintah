
/*
 *  GeomColormapInterface - interface to colormap class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCI_Geom_GeomColormapInterface_h
#define SCI_Geom_GeomColormapInterface_h 1

#include <Geom/Material.h>

class GeomColormapInterface {
public:
    virtual MaterialHandle& lookup2(double value)=0;
    virtual double getMin()=0;
    virtual double getMax()=0;
};

#endif
