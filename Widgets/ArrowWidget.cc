
/*
 *  ArrowWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Aug. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Widgets/ArrowWidget.h>
#include <Constraints/DistanceConstraint.h>
#include <Constraints/HypotenousConstraint.h>
#include <Geom/Cone.h>
#include <Geom/Cylinder.h>
#include <Geom/Group.h>
#include <Geom/Pick.h>
#include <Geom/Sphere.h>

const Index NumCons = 0;
const Index NumVars = 1;
const Index NumGeoms = 3;
const Index NumMatls = 3;
const Index NumSchemes = 1;

enum { ArrowW_Sphere, ArrowW_Cylinder, ArrowW_Cone };
enum { ArrowW_PointMatl, ArrowW_EdgeMatl, ArrowW_HighMatl };

ArrowWidget::ArrowWidget( Module* module )
: BaseWidget(module, NumVars, NumCons, NumGeoms, NumMatls), direction(0, 0, 1)
{
   variables[ArrowW_Point] = new Variable("Point", Scheme1, Point(0, 0, 0));

   materials[ArrowW_PointMatl] = new Material(Color(0,0,0), Color(.54, .60, 1),
						  Color(.5,.5,.5), 20);
   materials[ArrowW_EdgeMatl] = new Material(Color(0,0,0), Color(.54, .60, .66),
						 Color(.5,.5,.5), 20);
   materials[ArrowW_HighMatl] = new Material(Color(0,0,0), Color(.7,.7,.7),
						 Color(0,0,.6), 20);

   geometries[ArrowW_Sphere] = new GeomSphere;
   GeomPick* p=new GeomPick(module);
   p->set_highlight(materials[ArrowW_HighMatl]);
   p->set_cbdata((void*)ArrowW_Sphere);
   geometries[ArrowW_Sphere]->set_pick(p);
   geometries[ArrowW_Sphere]->set_matl(materials[ArrowW_PointMatl]);
   geometries[ArrowW_Cylinder] = new GeomCylinder;
   p=new GeomPick(module);
   p->set_highlight(materials[ArrowW_HighMatl]);
   p->set_cbdata((void*)ArrowW_Cylinder);
   geometries[ArrowW_Cylinder]->set_pick(p);
   geometries[ArrowW_Cylinder]->set_matl(materials[ArrowW_EdgeMatl]);
   geometries[ArrowW_Cone] = new GeomCone;
   p = new GeomPick(module);
   p->set_highlight(materials[ArrowW_HighMatl]);
   p->set_cbdata((void*)ArrowW_Cone);
   geometries[ArrowW_Cone]->set_pick(p);
   geometries[ArrowW_Cone]->set_matl(materials[ArrowW_EdgeMatl]);

   widget = new GeomGroup;
   for (Index geom = 0; geom <= NumGeoms; geom++) {
      widget->add(geometries[geom]);
   }
   widget->set_pick(new GeomPick(module));

   // Init variables.
   for (Index vindex=0; vindex<NumVariables; vindex++)
      variables[vindex]->Order();
   
   for (vindex=0; vindex<NumVariables; vindex++)
      variables[vindex]->Resolve();
}


ArrowWidget::~ArrowWidget()
{
}


void
ArrowWidget::execute()
{
   ((GeomSphere*)geometries[ArrowW_Sphere])->move(variables[ArrowW_Point]->Get(),
						  1*widget_scale);
   ((GeomCylinder*)geometries[ArrowW_Cylinder])->move(variables[ArrowW_Point]->Get(),
						      variables[ArrowW_Point]->Get()
						      + direction * widget_scale,
						      0.5*widget_scale);
   ((GeomCone*)geometries[ArrowW_Cone])->move(variables[ArrowW_Point]->Get()
					      + direction * widget_scale,
					      variables[ArrowW_Point]->Get()
					      + direction * widget_scale * 1.5,
					      1*widget_scale,
					      0);

   for (Index geom = 0; geom <= NumGeoms; geom++) {
      geometries[geom]->get_pick()->set_principal(direction);
   }
}

void
ArrowWidget::geom_moved( int axis, double dist, const Vector& delta,
			 void* cbdata )
{
   cerr << "Moved called..." << endl;
   switch((int)cbdata){
   case ArrowW_Sphere:
   case ArrowW_Cylinder:
   case ArrowW_Cone:
      variables[ArrowW_Point]->MoveDelta(delta);
      break;
   }
}

