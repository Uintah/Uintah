
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
const Index NumPcks = 1;
const Index NumMatls = 3;
const Index NumMdes = 1;
const Index NumSwtchs = 1;
// const Index NumSchemes = 1;

enum { GeomPoint, GeomShaft, GeomHead };
enum { Pick };

ArrowWidget::ArrowWidget( Module* module, CrowdMonitor* lock, double widget_scale )
: BaseWidget(module, lock, "ArrowWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale),
  direction(0, 0, 1)
{
   variables[PointVar] = new PointVariable("Point", solve, Scheme1, Point(0, 0, 0));

   GeomGroup* arr = new GeomGroup;
   geometries[GeomPoint] = new GeomSphere;
   materials[PointMatl] = new GeomMaterial(geometries[GeomPoint], DefaultPointMaterial);
   arr->add(materials[PointMatl]);
   geometries[GeomShaft] = new GeomCylinder;
   materials[ShaftMatl] = new GeomMaterial(geometries[GeomShaft], DefaultEdgeMaterial);
   arr->add(materials[ShaftMatl]);
   geometries[GeomHead] = new GeomCappedCone;
   materials[HeadMatl] = new GeomMaterial(geometries[GeomHead], DefaultEdgeMaterial);
   arr->add(materials[HeadMatl]);
   picks[Pick] = new GeomPick(arr, module, this, Pick);
   picks[Pick]->set_highlight(DefaultHighlightMaterial);
   CreateModeSwitch(0, picks[Pick]);

   SetMode(Mode0, Switch0);

   FinishWidget();
}


ArrowWidget::~ArrowWidget()
{
}


void
ArrowWidget::widget_execute()
{
   if (mode_switches[0]->get_state()) {
      Point center(variables[PointVar]->point());
      Vector direct(direction*widget_scale);
      ((GeomSphere*)geometries[GeomPoint])->move(center, widget_scale);
      ((GeomCylinder*)geometries[GeomShaft])->move(center,
						   center + direct * 3.0,
						   0.5*widget_scale);
      ((GeomCappedCone*)geometries[GeomHead])->move(center + direct * 3.0,
						    center + direct * 5.0,
						    widget_scale,
						    0);
   }

   Vector v1, v2;
   direction.find_orthogonal(v1, v2);
   for (Index geom = 0; geom < NumPcks; geom++) {
      picks[geom]->set_principal(direction, v1, v2);
   }
}


void
ArrowWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			 int pick, const BState& )
{
    switch(pick){
    case Pick:
	MoveDelta(delta);
	break;
    }
    execute();
}


void
ArrowWidget::MoveDelta( const Vector& delta )
{
   variables[PointVar]->MoveDelta(delta);

   execute();
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


Point
ArrowWidget::GetPosition() const
{
   return variables[PointVar]->point();
}


void
ArrowWidget::SetDirection( const Vector& v )
{
   direction = v;

   execute();
}


const Vector&
ArrowWidget::GetDirection() const
{
   return direction;
}


clString
ArrowWidget::GetMaterialName( const Index mindex ) const
{
   ASSERT(mindex<NumMaterials);
   
   switch(mindex){
   case 0:
      return "Point";
   case 1:
      return "Shaft";
   case 2:
      return "Head";
   default:
      return "UnknownMaterial";
   }
}


