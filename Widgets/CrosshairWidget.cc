
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

enum { CrosshairW_GeomCenter, CrosshairW_GeomAxis1, CrosshairW_GeomAxis2, CrosshairW_GeomAxis3 };
enum { CrosshairW_Pick };

CrosshairWidget::CrosshairWidget( Module* module, CrowdMonitor* lock, double widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale),
  axis1(1, 0, 0), axis2(0, 1, 0), axis3(0, 0, 1)
{
   variables[CrosshairW_Center] = new PointVariable("Crosshair", Scheme1, Point(0, 0, 0));

   materials[CrosshairW_CenterMatl] = PointWidgetMaterial;
   materials[CrosshairW_AxesMatl] = EdgeWidgetMaterial;
   materials[CrosshairW_HighMatl] = HighlightWidgetMaterial;

   GeomGroup* axes = new GeomGroup;
   geometries[CrosshairW_GeomCenter] = new GeomSphere;
   GeomMaterial* centerm = new GeomMaterial(geometries[CrosshairW_GeomCenter], materials[CrosshairW_CenterMatl]);
   axes->add(centerm);
   geometries[CrosshairW_GeomAxis1] = new GeomCappedCylinder;
   axes->add(geometries[CrosshairW_GeomAxis1]);
   geometries[CrosshairW_GeomAxis2] = new GeomCappedCylinder;
   axes->add(geometries[CrosshairW_GeomAxis2]);
   geometries[CrosshairW_GeomAxis3] = new GeomCappedCylinder;
   axes->add(geometries[CrosshairW_GeomAxis3]);
   GeomMaterial* axesm = new GeomMaterial(axes, materials[CrosshairW_AxesMatl]);
   picks[CrosshairW_Pick] = new GeomPick(axesm, module);
   picks[CrosshairW_Pick]->set_highlight(materials[CrosshairW_HighMatl]);
   picks[CrosshairW_Pick]->set_cbdata((void*)CrosshairW_Pick);

   FinishWidget(picks[CrosshairW_Pick]);
}


CrosshairWidget::~CrosshairWidget()
{
}


void
CrosshairWidget::widget_execute()
{
   ((GeomSphere*)geometries[CrosshairW_GeomCenter])->move(variables[CrosshairW_Center]->GetPoint(),
							  1*widget_scale);
   ((GeomCappedCylinder*)geometries[CrosshairW_GeomAxis1])->move(variables[CrosshairW_Center]->GetPoint()
								 - (axis1 * 100.0 * widget_scale),
								 variables[CrosshairW_Center]->GetPoint()
								 + (axis1 * 100.0 * widget_scale),
								 0.5*widget_scale);
   ((GeomCappedCylinder*)geometries[CrosshairW_GeomAxis2])->move(variables[CrosshairW_Center]->GetPoint()
								 - (axis2 * 100.0 * widget_scale),
								 variables[CrosshairW_Center]->GetPoint()
								 + (axis2 * 100.0 * widget_scale),
								 0.5*widget_scale);
   ((GeomCappedCylinder*)geometries[CrosshairW_GeomAxis3])->move(variables[CrosshairW_Center]->GetPoint()
								 - (axis3 * 100.0 * widget_scale),
								 variables[CrosshairW_Center]->GetPoint()
								 + (axis3 * 100.0 * widget_scale),
								 0.5*widget_scale);
}


void
CrosshairWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			 void* cbdata )
{
   switch((int)cbdata){
   case CrosshairW_Pick:
      variables[CrosshairW_Center]->MoveDelta(delta);
      break;
   }
}


void
CrosshairWidget::SetPosition( const Point& p )
{
    variables[CrosshairW_Center]->Move(p);
    execute();
}


const Point&
CrosshairWidget::GetPosition() const
{
   return variables[CrosshairW_Center]->GetPoint();
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


