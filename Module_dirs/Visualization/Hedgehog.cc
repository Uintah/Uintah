
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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/VectorFieldPort.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>
#include <Geom/Line.h>
#include <Geometry/Point.h>
#include <iostream.h>

class Hedgehog : public Module {
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

    int need_minmax;

    int hedgehog_id;

    MaterialHandle front_matl;
    MaterialHandle back_matl;
    virtual void geom_moved(int, double, const Vector&, void*);
public:
    Hedgehog(const clString& id);
    Hedgehog(const Hedgehog&, int deep);
    virtual ~Hedgehog();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_Hedgehog(const clString& id)
{
    return new Hedgehog(id);
}

static RegisterModule db1("Fields", "Hedgehog", make_Hedgehog);
static RegisterModule db2("Visualization", "Hedgehog", make_Hedgehog);

static clString hedgehog_name("Hedgehog");

Hedgehog::Hedgehog(const clString& id)
: Module("Hedgehog", id, Filter)
{
    // Create the input ports
    infield=new VectorFieldIPort(this, "Vector Field",
				 VectorFieldIPort::Atomic);
    add_iport(infield);

    // Create the output port
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);

    min=Point(0,0,0);
    max=Point(1,1,1);
#ifdef OLDUI
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
    ui_length_scale->set_minmax(0, 10);
    add_ui(ui_length_scale);
    radius=0;
    ui_radius=new MUI_slider_real("Radius", &radius,
				  MUI_widget::Immediate, 0);
    add_ui(ui_radius);

    need_minmax=1;

    front_matl=new MaterialProp(Color(0,0,0), Color(.6, 0, 0),
				Color(.5,0,0), 20);
    back_matl=new MaterialProp(Color(0,0,0), Color(0, 0, .6),
			       Color(0,0,.5), 20);
    hedgehog_id=0;
#endif
}

Hedgehog::Hedgehog(const Hedgehog& copy, int deep)
: Module(copy, deep)
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
#ifdef OLDUI
	ui_space_x->set_value(space_x);
	ui_space_x->set_minmax(0, diagonal.x());
	space_y=diagonal.y()/10;
	ui_space_y->set_value(space_y);
	ui_space_y->set_minmax(0, diagonal.y());
	space_z=diagonal.z()/10;
	ui_space_z->set_value(space_z);
	ui_space_z->set_minmax(0, diagonal.z());
#endif
	length_scale=Min(space_x, space_y, space_z)*.75;
	need_minmax=0;
    }
    GeomGroup* group=new GeomGroup;
    cerr << "length_scale=" << length_scale << endl;
    for(double x=min.x();x<=max.x();x+=space_x){
	for(double y=min.y();y<=max.y();y+=space_y){
	    for(double z=min.z();z<=max.z();z+=space_z){
		Point p(x,y,z);
		Vector v;
		if(field->interpolate(p, v)){
		    GeomLine* line=new GeomLine(p, p+(v*length_scale));
		    group->add(line);
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

void Hedgehog::geom_moved(int, double, const Vector& delta, void*)
{
}
