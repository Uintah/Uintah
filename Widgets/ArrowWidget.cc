
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

enum { GeomPoint, GeomShaft, GeomHead };
enum { Pick };

ArrowWidget::ArrowWidget( Module* module, CrowdMonitor* lock, double widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale),
  direction(0, 0, 1.0)
{
   variables[PointVar] = new PointVariable("Point", solve, Scheme1, Point(0, 0, 0));

   materials[PointMatl] = PointWidgetMaterial;
   materials[EdgeMatl] = EdgeWidgetMaterial;
   materials[HighMatl] = HighlightWidgetMaterial;

   GeomGroup* arr = new GeomGroup;
   geometries[GeomPoint] = new GeomSphere;
   GeomMaterial* sphm = new GeomMaterial(geometries[GeomPoint], materials[PointMatl]);
   arr->add(sphm);
   geometries[GeomShaft] = new GeomCylinder;
   GeomMaterial* cylm = new GeomMaterial(geometries[GeomShaft], materials[EdgeMatl]);
   arr->add(cylm);
   geometries[GeomHead] = new GeomCappedCone;
   GeomMaterial* conem = new GeomMaterial(geometries[GeomHead], materials[EdgeMatl]);
   arr->add(conem);
   picks[Pick] = new GeomPick(arr, module);
   picks[Pick]->set_highlight(materials[HighMatl]);
   picks[Pick]->set_cbdata((void*)Pick);

   FinishWidget(picks[Pick]);
}


ArrowWidget::~ArrowWidget()
{
}


void
ArrowWidget::widget_execute()
{
   ((GeomSphere*)geometries[GeomPoint])->move(variables[PointVar]->point(),
					      1*widget_scale);
   ((GeomCylinder*)geometries[GeomShaft])->move(variables[PointVar]->point(),
						variables[PointVar]->point()
						+ direction * widget_scale * 3.0,
						0.5*widget_scale);
   ((GeomCappedCone*)geometries[GeomHead])->move(variables[PointVar]->point()
						 + direction * widget_scale * 3.0,
						 variables[PointVar]->point()
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
   case Pick:
      MoveDelta(delta);
      break;
   }
}


void
ArrowWidget::MoveDelta( const Vector& delta )
{
   variables[PointVar]->MoveDelta(delta);
}


Point
ArrowWidget::ReferencePoint() const
{
   return variables[PointVar]->point();
}


void
ArrowWidget::SetPosition( const Point& p )
{
   variables[PointVar]->Move(p);
   execute();
}


const Point&
ArrowWidget::GetPosition() const
{
   return variables[PointVar]->point();
}


void
ArrowWidget::SetDirect( const Vector& v )
{
   direction = v;
   execute();
}


const Vector&
ArrowWidget::GetDirect() const
{
   return direction;
}


