
/*
 *  GeomColormapInterface.h - interface to colormap class
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

#include <Core/Geom/Material.h>

namespace SCIRun {

class SCICORESHARE GeomColormapInterface {
public:
    virtual MaterialHandle& lookup2(double value)=0;
    virtual double getMin()=0;
    virtual double getMax()=0;
};

} // End namespace SCIRun



#endif
