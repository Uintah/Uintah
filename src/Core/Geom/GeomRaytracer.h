
/*
 *  GeomRaytracer.h: Information for Ray Tracing
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_RayTracer_h
#define SCI_Geom_RayTracer_h 1

#include <Geom/Material.h>

namespace PSECommon {
  namespace Modules {
    class Raytracer;
  }
}

namespace SCICore {
namespace GeomSpace {

using PSECommon::Modules::Raytracer;

class GeomObj;
class View;

class Hit {
    double _t;
    GeomObj* _prim;
    Material* _matl;
    void* _data;
public:
    Hit();
    ~Hit();
    double t() const;
    int hit() const;
    GeomObj* prim() const;
    MaterialHandle matl() const;

    void hit(double t, GeomObj*, Material*, void* data=0);
};

struct OcclusionData {
    GeomObj* hit_prim;
    Raytracer* raytracer;
    int level;
    View* view;
    OcclusionData(GeomObj*, Raytracer*, int level, View* view);
};

} // End namespace GeomSpace
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:56:43  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:56:06  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:00  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:22  dav
// Import sources
//
//

#endif /* SCI_Geom_RayTracer_h */

