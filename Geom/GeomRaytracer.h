
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
class GeomObj;
class Raytracer;
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


#endif /* SCI_Geom_GeomOpenGL_h */

