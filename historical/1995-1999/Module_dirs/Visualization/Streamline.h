
/*
 *  Streamline.h: Visualization module
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_Streamline_h
#define SCI_project_module_Streamline_h

#include <Dataflow/Module.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/VectorField.h>
#include <Datatypes/VectorFieldPort.h>
#include <Geom/Geom.h>
#include <Geometry/Point.h>
#include <TCL/TCLvar.h>
class ColormapPort;
class GeomCylinder;
class GeomDisc;
class GeomGroup;
class GeomPick;
class GeomSphere;
class GeometryOPort;

class Streamline : public Module {
    VectorFieldIPort* infield;
    ColormapPort* incolormap;
    ScalarFieldIPort* incolorfield;
    GeometryOPort* ogeom;
    int abort_flag;

    TCLstring widgettype;
    clString oldwidgettype;
    TCLstring markertype;
    TCLdouble lineradius;
    TCLstring algorithm;

    TCLvardouble stepsize;
    TCLvarint maxsteps;

    int need_p1;
    Point p1;
    Point p2;
    Point p3;
    Point p4;
    double slider1_dist;
    double slider2_dist;

    GeomGroup* widget;
    GeomSphere* widget_p1;
    GeomSphere* widget_p2;
    GeomSphere* widget_p3;
    GeomSphere* widget_p4;
    GeomCylinder* widget_edge1;
    GeomCylinder* widget_edge2;
    GeomCylinder* widget_edge3;
    GeomCylinder* widget_edge4;
    GeomGroup* widget_slider1;
    GeomGroup* widget_slider2;
    GeomCylinder* widget_slider1body;
    GeomCylinder* widget_slider2body;
    GeomDisc* widget_slider1cap1;
    GeomDisc* widget_slider1cap2;
    GeomDisc* widget_slider2cap1;
    GeomDisc* widget_slider2cap2;
    GeomPick* widget_p1_pick;
    GeomPick* widget_p2_pick;
    GeomPick* widget_p3_pick;
    GeomPick* widget_p4_pick;
    GeomPick* widget_edge1_pick;
    GeomPick* widget_edge2_pick;
    GeomPick* widget_edge3_pick;
    GeomPick* widget_edge4_pick;
    GeomPick* widget_slider1_pick;
    GeomPick* widget_slider2_pick;
    int widget_id;
    double widget_scale;

    int streamline_id;

    MaterialHandle widget_point_matl;
    MaterialHandle widget_edge_matl;
    MaterialHandle widget_slider_matl;
    MaterialHandle widget_highlight_matl;
    MaterialHandle matl;

    virtual void geom_moved(int, double, const Vector&, void*);
public:
    Streamline(const clString& id);
    Streamline(const Streamline&, int deep);
    virtual ~Streamline();
    virtual Module* clone(int deep);
    virtual void execute();
};

#endif
