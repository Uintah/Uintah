
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
class ColormapPort;
class Field3DIPort;
class GeometryOPort;
class MUI_slider_real;
class ObjGroup;


class IsoSurface : public UserModule {
    Field3DIPort* infield;
    ColormapPort* incolormap;
    Field3DIPort* incolorfield;
    GeometryOPort* ogeom;
    int abort_flag;

    int have_seedpoint;
    Point seed_point;
    int need_seed;
    double isoval;
    int make_normals;
    int do_3dwidget;
    double scalar_val;

    double old_min;
    double old_max;
    MUI_slider_real* value_slider;

    int widget_id;

    int iso_cube(int, int, int, double, ObjGroup*, const Field3DHandle&);
    void iso_tetra(int, double, ObjGroup*, const Field3DHandle&);

    void iso_reg_grid(const Field3DHandle&, const Point&, ObjGroup*);
    void iso_reg_grid(const Field3DHandle&, double, ObjGroup*);
    void iso_tetrahedra(const Field3DHandle&, const Point&, ObjGroup*);
    void iso_tetrahedra(const Field3DHandle&, double, ObjGroup*);

    void find_seed_from_value(const Field3DHandle&);
public:
    IsoSurface();
    IsoSurface(const IsoSurface&, int deep);
    virtual ~IsoSurface();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void mui_callback(void*, int);
};

#endif
