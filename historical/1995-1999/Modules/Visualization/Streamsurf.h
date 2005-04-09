
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

#include <Dataflow/Module.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/VectorField.h>
#include <Datatypes/VectorFieldPort.h>
#include <Geom/Geom.h>
#include <Geometry/Point.h>
class ColormapPort;
class GeomCylinder;
class GeomDisc;
class GeomGroup;
class GeomPick;
class GeomSphere;
class GeometryOPort;


class Streamsurf : public Module {
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

    int maxsteps;
    double maxangle;

    int need_p1;
    int need_widget;
    Point p1;
    Point p2;
    double slider1_dist;

    GeomGroup* widget;
    GeomSphere* widget_p1;
    GeomSphere* widget_p2;
    GeomCylinder* widget_edge1;
    GeomGroup* widget_slider1;
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

    MaterialHandle widget_point_matl;
    MaterialHandle widget_edge_matl;
    MaterialHandle widget_slider_matl;
    MaterialHandle widget_highlight_matl;
    MaterialHandle matl;

    virtual void geom_moved(int, double, const Vector&, void*);
public:
    Streamsurf(const clString& id);
    Streamsurf(const Streamsurf&, int deep);
    virtual ~Streamsurf();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
