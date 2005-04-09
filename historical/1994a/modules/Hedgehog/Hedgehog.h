
/*
 *  Hedgehog.h: Visualization module
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_Hedgehog_h
#define SCI_project_module_Hedgehog_h

#include <UserModule.h>
#include <Geometry/Point.h>
#include <ScalarField.h>
#include <ScalarFieldPort.h>
#include <VectorField.h>
#include <VectorFieldPort.h>
class GeometryOPort;
class MaterialProp;
class MUI_point;
class MUI_slider_real;


class Hedgehog : public UserModule {
    VectorFieldIPort* infield;
    GeometryOPort* ogeom;
    int abort_flag;

    Point min;
    Point max;
    double space_x;
    double space_y;
    double space_z;
    double length_scale;
    double radius;

    MUI_point* ui_min;
    MUI_point* ui_max;
    MUI_slider_real* ui_space_x;
    MUI_slider_real* ui_space_y;
    MUI_slider_real* ui_space_z;
    MUI_slider_real* ui_length_scale;
    MUI_slider_real* ui_radius;

    int need_minmax;

    int hedgehog_id;

    MaterialProp* front_matl;
    MaterialProp* back_matl;
    virtual void geom_moved(int, double, const Vector&, void*);
public:
    Hedgehog();
    Hedgehog(const Hedgehog&, int deep);
    virtual ~Hedgehog();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void mui_callback(void*, int);
};

#endif
