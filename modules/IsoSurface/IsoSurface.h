
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
#include <ScalarField.h>
#include <ScalarFieldPort.h>
class ColormapPort;
class GeomCone;
class GeomCylinder;
class GeomDisc;
class GeomSphere;
class GeometryOPort;
class MaterialProp;
class MUI_slider_real;
class ObjGroup;


class IsoSurface : public UserModule {
    ScalarFieldIPort* infield;
    ColormapPort* incolormap;
    ScalarFieldIPort* incolorfield;
    GeometryOPort* ogeom;
    int abort_flag;

    int have_seedpoint;
    Point seed_point;
    int need_seed;
    double isoval;
    int make_normals;
    int do_3dwidget;
    double scalar_val;
    ObjGroup* widget;
    GeomSphere* widget_sphere;
    GeomCylinder* widget_cylinder;
    GeomCone* widget_cone;
    GeomDisc* widget_disc;
    double widget_scale;

    int widget_id;
    int isosurface_id;

    double old_min;
    double old_max;
    MUI_slider_real* value_slider;

    MaterialProp* widget_matl;
    MaterialProp* widget_highlight_matl;

    int iso_cube(int, int, int, double, ObjGroup*, ScalarFieldRG*);
    void iso_tetra(int, double, ObjGroup*, ScalarFieldUG*);

    void iso_reg_grid(ScalarFieldRG*, const Point&, ObjGroup*);
    void iso_reg_grid(ScalarFieldRG*, double, ObjGroup*);
    void iso_tetrahedra(ScalarFieldUG*, const Point&, ObjGroup*);
    void iso_tetrahedra(ScalarFieldUG*, double, ObjGroup*);

    void find_seed_from_value(const ScalarFieldHandle&);

    virtual void geom_moved(int, double, const Vector&, void*);
    Point ov[9];
    Point v[9];
public:
    IsoSurface();
    IsoSurface(const IsoSurface&, int deep);
    virtual ~IsoSurface();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void mui_callback(void*, int);
};

#endif
