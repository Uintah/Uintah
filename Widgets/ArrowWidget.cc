
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
#include <Geom/Cone.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>

const Index NumCons = 0;
const Index NumVars = 1;
const Index NumGeoms = 3;
const Index NumMatls = 3;
const Index NumPcks = 1;
// const Index NumSchemes = 1;

enum { ArrowW_Sphere, ArrowW_Cylinder, ArrowW_Cone };
enum { ArrowW_PointMatl, ArrowW_EdgeMatl, ArrowW_HighMatl };
enum { ArrowW_Pick };

ArrowWidget::ArrowWidget( Module* module, CrowdMonitor* lock,

			 double widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale),
  direction(0, 0, 1.0)
{
   variables[ArrowW_Point] = new Variable("Point", Scheme1, Point(0, 0, 0));

   materials[ArrowW_PointMatl] = PointWidgetMaterial;
   materials[ArrowW_EdgeMatl] = EdgeWidgetMaterial;
   materials[ArrowW_HighMatl] = HighlightWidgetMaterial;

   GeomGroup* arr = new GeomGroup;
   geometries[ArrowW_Sphere] = new GeomSphere;
   GeomMaterial* sphm = new GeomMaterial(geometries[ArrowW_Sphere], materials[ArrowW_PointMatl]);
   arr->add(sphm);
   geometries[ArrowW_Cylinder] = new GeomCylinder;
   GeomMaterial* cylm = new GeomMaterial(geometries[ArrowW_Cylinder], materials[ArrowW_EdgeMatl]);
   arr->add(cylm);
   geometries[ArrowW_Cone] = new GeomCone;
   GeomMaterial* conem = new GeomMaterial(geometries[ArrowW_Cone], materials[ArrowW_EdgeMatl]);
   arr->add(conem);
   picks[ArrowW_Pick] = new GeomPick(arr, module);
   picks[ArrowW_Pick]->set_highlight(materials[ArrowW_HighMatl]);
   picks[ArrowW_Pick]->set_cbdata((void*)ArrowW_Pick);

   FinishWidget(picks[ArrowW_Pick]);
}


ArrowWidget::~ArrowWidget()
{
}


void
ArrowWidget::widget_execute()
{
   ((GeomSphere*)geometries[ArrowW_Sphere])->move(variables[ArrowW_Point]->Get(),
						  1*widget_scale);
   ((GeomCylinder*)geometries[ArrowW_Cylinder])->move(variables[ArrowW_Point]->Get(),
						      variables[ArrowW_Point]->Get()
						      + direction * widget_scale * 3.0,
						      0.5*widget_scale);
   ((GeomCone*)geometries[ArrowW_Cone])->move(variables[ArrowW_Point]->Get()
					      + direction * widget_scale * 3.0,
					      variables[ArrowW_Point]->Get()
					      + direction * widget_scale * 5.0,
					      widget_scale,
					      0);

   Vector v1, v2;
   direction.find_orthogonal(v1, v2);
   for (Index geom = 0; geom < NumPcks; geom++) {
      picks[geom]->set_principal(direction, v1, v2);
   }
}

void
ArrowWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			 void* cbdata )
{
   switch((int)cbdata){
   case ArrowW_Pick:
      variables[ArrowW_Point]->MoveDelta(delta);
      break;
   }
}

