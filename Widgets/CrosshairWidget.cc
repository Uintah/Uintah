
/*
 *  CrosshairWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Widgets/CrosshairWidget.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>

const Index NumCons = 0;
const Index NumVars = 1;
const Index NumGeoms = 4;
const Index NumMatls = 3;
const Index NumPcks = 1;
// const Index NumSchemes = 1;

enum { GeomCenter, GeomAxis1, GeomAxis2, GeomAxis3 };
enum { Pick };

CrosshairWidget::CrosshairWidget( Module* module, CrowdMonitor* lock, double widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale),
  axis1(1, 0, 0), axis2(0, 1, 0), axis3(0, 0, 1)
{
   variables[CenterVar] = new PointVariable("Crosshair", solve, Scheme1, Point(0, 0, 0));

   materials[CenterMatl] = PointWidgetMaterial;
   materials[AxesMatl] = EdgeWidgetMaterial;
   materials[HighMatl] = HighlightWidgetMaterial;

   GeomGroup* axes = new GeomGroup;
   geometries[GeomCenter] = new GeomSphere;
   GeomMaterial* centerm = new GeomMaterial(geometries[GeomCenter], materials[CenterMatl]);
   axes->add(centerm);
   geometries[GeomAxis1] = new GeomCappedCylinder;
   axes->add(geometries[GeomAxis1]);
   geometries[GeomAxis2] = new GeomCappedCylinder;
   axes->add(geometries[GeomAxis2]);
   geometries[GeomAxis3] = new GeomCappedCylinder;
   axes->add(geometries[GeomAxis3]);
   GeomMaterial* axesm = new GeomMaterial(axes, materials[AxesMatl]);
   picks[Pick] = new GeomPick(axesm, module);
   picks[Pick]->set_highlight(materials[HighMatl]);
   picks[Pick]->set_cbdata((void*)Pick);

   FinishWidget(picks[Pick]);
}


CrosshairWidget::~CrosshairWidget()
{
}


void
CrosshairWidget::widget_execute()
{
   ((GeomSphere*)geometries[GeomCenter])->move(variables[CenterVar]->point(),
					       1*widget_scale);
   ((GeomCappedCylinder*)geometries[GeomAxis1])->move(variables[CenterVar]->point()
						      - (axis1 * 100.0 * widget_scale),
						      variables[CenterVar]->point()
						      + (axis1 * 100.0 * widget_scale),
						      0.5*widget_scale);
   ((GeomCappedCylinder*)geometries[GeomAxis2])->move(variables[CenterVar]->point()
						      - (axis2 * 100.0 * widget_scale),
						      variables[CenterVar]->point()
						      + (axis2 * 100.0 * widget_scale),
						      0.5*widget_scale);
   ((GeomCappedCylinder*)geometries[GeomAxis3])->move(variables[CenterVar]->point()
						      - (axis3 * 100.0 * widget_scale),
						      variables[CenterVar]->point()
						      + (axis3 * 100.0 * widget_scale),
						      0.5*widget_scale);
}


void
CrosshairWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			     void* cbdata )
{
   switch((int)cbdata){
   case Pick:
      MoveDelta(delta);
      break;
   }
}


void
CrosshairWidget::MoveDelta( const Vector& delta )
{
   variables[CenterVar]->MoveDelta(delta);
}


Point
CrosshairWidget::ReferencePoint() const
{
   return variables[CenterVar]->point();
}


void
CrosshairWidget::SetPosition( const Point& p )
{
   variables[CenterVar]->Move(p);
   execute();
}


const Point&
CrosshairWidget::GetPosition() const
{
   return variables[CenterVar]->point();
}


void
CrosshairWidget::SetAxes( const Vector& v1, const Vector& v2, const Vector& v3 )
{
   if ((v1.length2() > 1e-6)
       && (v2.length2() > 1e-6)
       && (v3.length2() > 1e-6)) {
      axis1 = v1.normal();
      axis2 = v2.normal();
      axis3 = v3.normal();
      execute();
   }
}


void
CrosshairWidget::GetAxes( Vector& v1, Vector& v2, Vector& v3 ) const
{
   v1 = axis1;
   v2 = axis2;
   v3 = axis3;
}


