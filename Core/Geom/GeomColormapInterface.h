
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

#include <Geom/Material.h>

namespace SCICore {
namespace GeomSpace {

class GeomColormapInterface {
public:
    virtual MaterialHandle& lookup2(double value)=0;
    virtual double getMin()=0;
    virtual double getMax()=0;
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:37  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:03  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:55  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//


#endif
