
/*
 *  IsoSurface.h: The 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_IsoSurface_h
#define SCI_project_module_IsoSurface_h

#include <UserModule.h>
#include <Geometry/Point.h>
#include <Field3D.h>
class Field3DIPort;
class GeometryOPort;
class ObjGroup;

class IsoSurface : public UserModule {
    Field3DIPort* infield;
    GeometryOPort* ogeom;
    int abort_flag;

    int have_seedpoint;
    Point seed_point;
    double isoval;

    void iso_cube(int, int, int, double, ObjGroup*);
    void iso_tetra(int, double, ObjGroup*);

    void iso_reg_grid(const Field3DHandle&, const Point&, ObjGroup*);
    void iso_reg_grid(const Field3DHandle&, double, ObjGroup*);
    void iso_tetrahedra(const Field3DHandle&, const Point&, ObjGroup*);
    void iso_tetrahedra(const Field3DHandle&, double, ObjGroup*);
public:
    IsoSurface();
    IsoSurface(const IsoSurface&, int deep);
    virtual ~IsoSurface();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void mui_callback(void*, int);
};

#endif
