
/*
 *  PointWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Widgets/PointWidget.h>
#include <Geom/Sphere.h>
#include <Malloc/Allocator.h>

const Index NumCons = 0;
const Index NumVars = 1;
const Index NumGeoms = 1;
const Index NumPcks = 1;
const Index NumMatls = 1;
const Index NumMdes = 1;
const Index NumSwtchs = 1;
// const Index NumSchemes = 1;

enum { GeomPoint };
enum { Pick };

PointWidget::PointWidget( Module* module, CrowdMonitor* lock, double widget_scale )
: BaseWidget(module, lock, "PointWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale)
{
   variables[PointVar] = scinew PointVariable("Point", solve, Scheme1, Point(0, 0, 0));

   geometries[GeomPoint] = scinew GeomSphere;
   materials[PointMatl] = scinew GeomMaterial(geometries[GeomPoint], DefaultPointMaterial);
   picks[Pick] = scinew GeomPick(materials[PointMatl], module, this, Pick);
   picks[Pick]->set_highlight(DefaultHighlightMaterial);
   CreateModeSwitch(0, picks[Pick]);

   SetMode(Mode0, Switch0);
   
   FinishWidget();
}


PointWidget::~PointWidget()
{
}


void
PointWidget::redraw()
{
   if (mode_switches[0]->get_state())
      ((GeomSphere*)geometries[GeomPoint])->move(variables[PointVar]->point(),
						 widget_scale);
}


void
PointWidget::geom_moved( GeomPick*, int /* axis */, double /* dist */,
			 const Vector& delta, int pick, const BState& )
{
   switch(pick){
   case Pick:
      MoveDelta(delta);
      break;
   }
   execute(0);
}


void
PointWidget::MoveDelta( const Vector& delta )
{
   variables[PointVar]->MoveDelta(delta);

   execute(1);
}


Point
PointWidget::ReferencePoint() const
{
   return variables[PointVar]->point();
}


void
PointWidget::SetPosition( const Point& p )
{
   variables[PointVar]->Move(p);

   execute(0);
}


Point
PointWidget::GetPosition() const
{
   return variables[PointVar]->point();
}


clString
PointWidget::GetMaterialName( const Index mindex ) const
{
   ASSERT(mindex<NumMaterials);
   
   switch(mindex){
   case 0:
      return "Point";
   default:
      return "UnknownMaterial";
   }
}


void
PointWidget::widget_tcl( TCLArgs& args )
{
   if (args[1] == "translate"){
      if (args.count() != 4) {
	 args.error("point widget needs axis translation");
	 return;
      }
      Real trans;
      if (!args[3].get_double(trans)) {
	 args.error("point widget can't parse translation `"+args[3]+"'");
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
	 args.error("point widget unknown axis `"+args[2]+"'");
	 break;
      }
      SetPosition(p);
   }
}

