
/*
 *  Streamsurf.h: Visualization module
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_Streamsurf_h
#define SCI_project_module_Streamsurf_h

#include <UserModule.h>
#include <Geometry/Point.h>
#include <ScalarField.h>
#include <ScalarFieldPort.h>
#include <VectorField.h>
#include <VectorFieldPort.h>
class ColormapPort;
class GeomCylinder;
class GeomDisc;
class GeomPick;
class GeomSphere;
class GeometryOPort;
class MaterialProp;
class MUI_slider_real;
class ObjGroup;


class Streamsurf : public UserModule {
    VectorFieldIPort* infield;
    ColormapPort* incolormap;
    ScalarFieldIPort* incolorfield;
    GeometryOPort* ogeom;
    int abort_flag;

public:
    enum Algorithm {
	AEuler,
	ARK4,
	AStreamFunction,
    };
private:
    int alg;

    double stepsize;
    int maxsteps;
    double maxangle;

    int need_p1;
    int need_widget;
    Point p1;
    Point p2;
    double slider1_dist;

    ObjGroup* widget;
    GeomSphere* widget_p1;
    GeomSphere* widget_p2;
    GeomCylinder* widget_edge1;
    ObjGroup* widget_slider1;
    GeomCylinder* widget_slider1body;
    GeomDisc* widget_slider1cap1;
    GeomDisc* widget_slider1cap2;
    GeomPick* widget_p1_pick;
    GeomPick* widget_p2_pick;
    GeomPick* widget_edge1_pick;
    GeomPick* widget_slider1_pick;
    int widget_id;
    double widget_scale;

    int streamsurf_id;

    MaterialProp* widget_point_matl;
    MaterialProp* widget_edge_matl;
    MaterialProp* widget_slider_matl;
    MaterialProp* widget_highlight_matl;

    virtual void geom_moved(int, double, const Vector&, void*);
public:
    Streamsurf();
    Streamsurf(const Streamsurf&, int deep);
    virtual ~Streamsurf();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void mui_callback(void*, int);
};

#endif
