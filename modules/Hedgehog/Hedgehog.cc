
/*
 *  Hedgehog.cc:  Generate Hedgehogs from a field...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Hedgehog/Hedgehog.h>
#include <Geom.h>
#include <GeometryPort.h>
#include <ModuleList.h>
#include <MUI.h>
#include <NotFinished.h>
#include <Geometry/Point.h>
#include <iostream.h>
#include <fstream.h>

static Module* make_Hedgehog()
{
    return new Hedgehog;
}

static RegisterModule db1("Fields", "Hedgehog", make_Hedgehog);
static RegisterModule db2("Visualization", "Hedgehog", make_Hedgehog);

static clString hedgehog_name("Hedgehog");

Hedgehog::Hedgehog()
: UserModule("Hedgehog", Filter)
{
    // Create the input ports
    infield=new VectorFieldIPort(this, "Vector Field", ScalarFieldIPort::Atomic);
    add_iport(infield);

    // Create the output port
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);

    min=Point(0,0,0);
    max=Point(1,1,1);
    ui_min=new MUI_point("Min: ", &min, MUI_widget::Immediate, 0);
    add_ui(ui_min);
    ui_max=new MUI_point("Max: ", &max, MUI_widget::Immediate, 0);
    add_ui(ui_max);
    space_x=space_y=space_z=1;
    ui_space_x=new MUI_slider_real("X spacing", &space_x,
				   MUI_widget::Immediate, 0);
    add_ui(ui_space_x);
    ui_space_y=new MUI_slider_real("Y spacing", &space_y,
				   MUI_widget::Immediate, 0);
    add_ui(ui_space_y);
    ui_space_z=new MUI_slider_real("Z spacing", &space_z,
				   MUI_widget::Immediate, 0);
    add_ui(ui_space_z);
    length_scale=1;
    ui_length_scale=new MUI_slider_real("Length Scale", &length_scale,
					MUI_widget::Immediate, 0);
    add_ui(ui_length_scale);
    radius=0;
    ui_radius=new MUI_slider_real("Length Scale", &radius,
				  MUI_widget::Immediate, 0);
    add_ui(ui_radius);

    need_minmax=1;

    front_matl=new MaterialProp(Color(0,0,0), Color(.6, 0, 0),
				Color(.5,0,0), 20);
    back_matl=new MaterialProp(Color(0,0,0), Color(0, 0, .6),
			       Color(0,0,.5), 20);
    hedgehog_id=0;
}

Hedgehog::Hedgehog(const Hedgehog& copy, int deep)
: UserModule(copy, deep)
{
    NOT_FINISHED("Hedgehog::Hedgehog");
}

Hedgehog::~Hedgehog()
{
}

Module* Hedgehog::clone(int deep)
{
    return new Hedgehog(*this, deep);
}

void Hedgehog::execute()
{
    abort_flag=0;
    if(hedgehog_id)
	ogeom->delObj(hedgehog_id);
    VectorFieldHandle field;
    if(!infield->get(field))
	return;
    if(need_minmax){
	field->get_bounds(min, max);
	Vector diagonal(max-min);
	space_x=diagonal.x()/10;
	ui_space_x->set_value(space_x);
	space_y=diagonal.y()/10;
	ui_space_y->set_value(space_y);
	space_z=diagonal.z()/10;
	ui_space_z->set_value(space_z);
	length_scale=Min(space_x, space_y, space_z)*.75;
    }
    ObjGroup* group=new ObjGroup;
    for(double x=min.x();x<=max.x();x+=space_x){
	for(double y=min.y();y<=max.y();y+=space_y){
	    for(double z=min.z();z<=max.z();z+=space_z){
		Point p(x,y,z);
		Vector v;
		if(field->interpolate(p, v)){
		    GeomLine* line=new GeomLine(p, p+(v*length_scale));
		}
	    }
	}
    }

    if(group->size() == 0){
	delete group;
	hedgehog_id=0;
    } else {
	hedgehog_id=ogeom->addObj(group, hedgehog_name);
    }
}

void Hedgehog::mui_callback(void*, int)
{
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}

void Hedgehog::geom_moved(int, double, const Vector& delta, void*)
{
}
