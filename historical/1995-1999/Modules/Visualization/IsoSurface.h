
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

#include <Dataflow/Module.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Geometry/Point.h>
#include <Geom/Geom.h>
class ColormapPort;
class Element;
class GeomCone;
class GeomCylinder;
class GeomDisc;
class GeomGroup;
class GeomSphere;
class GeometryOPort;
class Mesh;


class IsoSurface : public Module {
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
    GeomGroup* widget;
    GeomSphere* widget_sphere;
    GeomCylinder* widget_cylinder;
    GeomCone* widget_cone;
    GeomDisc* widget_disc;
    double widget_scale;

    int widget_id;
    int isosurface_id;

    double old_min;
    double old_max;

    MaterialHandle widget_matl;
    MaterialHandle widget_highlight_matl;
    MaterialHandle matl;

    int iso_cube(int, int, int, double, GeomGroup*, ScalarFieldRG*);
    int iso_tetra(Element*, Mesh*, ScalarFieldUG*, double, GeomGroup*);

    void iso_reg_grid(ScalarFieldRG*, const Point&, GeomGroup*);
    void iso_reg_grid(ScalarFieldRG*, double, GeomGroup*);
    void iso_tetrahedra(ScalarFieldUG*, const Point&, GeomGroup*);
    void iso_tetrahedra(ScalarFieldUG*, double, GeomGroup*);

    void find_seed_from_value(const ScalarFieldHandle&);

    virtual void geom_moved(int, double, const Vector&, void*);
    Point ov[9];
    Point v[9];
public:
    IsoSurface(const clString& id);
    IsoSurface(const IsoSurface&, int deep);
    virtual ~IsoSurface();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
