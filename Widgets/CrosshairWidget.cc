
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
const Index NumPcks = 2;
const Index NumMatls = 2;
const Index NumMdes = 2;
const Index NumSwtchs = 2;
// const Index NumSchemes = 1;

enum { GeomCenter, GeomAxis1, GeomAxis2, GeomAxis3 };
enum { Pick, PickAxes };

CrosshairWidget::CrosshairWidget( Module* module, CrowdMonitor* lock, double widget_scale )
: BaseWidget(module, lock, "CrosshairWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale),
  axis1(1, 0, 0), axis2(0, 1, 0), axis3(0, 0, 1)
{
   variables[CenterVar] = new PointVariable("Crosshair", solve, Scheme1, Point(0, 0, 0));

   geometries[GeomCenter] = new GeomSphere;
   materials[PointMatl] = new GeomMaterial(geometries[GeomCenter], DefaultPointMaterial);
   picks[PickAxes] = new GeomPick(materials[PointMatl], module, this, PickAxes);
   picks[PickAxes]->set_highlight(DefaultHighlightMaterial);
   CreateModeSwitch(0, picks[PickAxes]);

   GeomGroup* axes = new GeomGroup;
   geometries[GeomAxis1] = new GeomCappedCylinder;
   axes->add(geometries[GeomAxis1]);
   geometries[GeomAxis2] = new GeomCappedCylinder;
   axes->add(geometries[GeomAxis2]);
   geometries[GeomAxis3] = new GeomCappedCylinder;
   axes->add(geometries[GeomAxis3]);
   materials[AxesMatl] = new GeomMaterial(axes, DefaultEdgeMaterial);
   picks[Pick] = new GeomPick(materials[AxesMatl], module, this, Pick);
   picks[Pick]->set_highlight(DefaultHighlightMaterial);
   CreateModeSwitch(1, picks[Pick]);

   SetMode(Mode0, Switch0|Switch1);
   SetMode(Mode1, Switch0);

   FinishWidget();
}


CrosshairWidget::~CrosshairWidget()
{
}


void
CrosshairWidget::redraw()
{
   Point center(variables[CenterVar]->point());
   
   if (mode_switches[0]->get_state())
      ((GeomSphere*)geometries[GeomCenter])->move(center, widget_scale);

   if (mode_switches[1]->get_state()) {
      Real axislen(100.0*widget_scale), axisrad(0.5*widget_scale);
      ((GeomCappedCylinder*)geometries[GeomAxis1])->move(center - (axis1 * axislen),
							 center + (axis1 * axislen),
							 axisrad);
      ((GeomCappedCylinder*)geometries[GeomAxis2])->move(center - (axis2 * axislen),
							 center + (axis2 * axislen),
							 axisrad);
      ((GeomCappedCylinder*)geometries[GeomAxis3])->move(center - (axis3 * axislen),
							 center + (axis3 * axislen),
							 axisrad);
   }

   for (Index geom = 0; geom < NumPcks; geom++) {
      picks[geom]->set_principal(axis1, axis2, axis3);
   }
}


void
CrosshairWidget::geom_moved( GeomPick*, int /* axis */, double /* dist */,
			     const Vector& delta, int pick, const BState& )
{
   switch(pick){
   case Pick:
   case PickAxes:
      MoveDelta(delta);
      break;
   }
   execute();
}


void
CrosshairWidget::MoveDelta( const Vector& delta )
{
   variables[CenterVar]->MoveDelta(delta);

   execute();
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


Point
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


clString
CrosshairWidget::GetMaterialName( const Index mindex ) const
{
   ASSERT(mindex<NumMaterials);
   
   switch(mindex){
   case 0:
      return "Point";
   case 1:
      return "Axes";
   default:
      return "UnknownMaterial";
   }
}


void
CrosshairWidget::widget_tcl( TCLArgs& args )
{
   if (args[1] == "translate"){
      if (args.count() != 4) {
	 args.error("crosshair widget needs axis translation");
	 return;
      }
      Real trans;
      if (!args[3].get_double(trans)) {
	 args.error("crosshair widget can't parse translation `"+args[3]+"'");
	 return;
      }
      Point p(GetPosition());
      switch (args[2](0)) {
      case 'x':
	 p.x(trans);
	 break;
      case 'y':
	 p.y(trans);
	 break;
      case 'z':
	 p.z(trans);
	 break;
      default:
	 args.error("crosshair widget unknown axis `"+args[2]+"'");
	 break;
      }
      SetPosition(p);
   }
}

