
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

const Index NumCons = 0;
const Index NumVars = 1;
const Index NumGeoms = 1;
const Index NumMatls = 2;
const Index NumPcks = 1;
// const Index NumSchemes = 1;

enum { GeomPoint };
enum { Pick };

PointWidget::PointWidget( Module* module, CrowdMonitor* lock, double widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale)
{
   variables[PointVar] = new PointVariable("Point", solve, Scheme1, Point(0, 0, 0));

   materials[PointMatl] = PointWidgetMaterial;
   materials[HighMatl] = HighlightWidgetMaterial;

   geometries[GeomPoint] = new GeomSphere;
   GeomMaterial* sphm = new GeomMaterial(geometries[GeomPoint], materials[PointMatl]);
   picks[Pick] = new GeomPick(sphm, module);
   picks[Pick]->set_highlight(materials[HighMatl]);
   picks[Pick]->set_cbdata((void*)Pick);

   FinishWidget(picks[Pick]);
}


PointWidget::~PointWidget()
{
}


void
PointWidget::widget_execute()
{
   ((GeomSphere*)geometries[GeomPoint])->move(variables[PointVar]->point(),
					      1*widget_scale);
}


void
PointWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			 void* cbdata )
{
   switch((int)cbdata){
   case Pick:
      MoveDelta(delta);
      break;
   }
}


void
PointWidget::MoveDelta( const Vector& delta )
{
   variables[PointVar]->MoveDelta(delta);
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
   execute();
}


const Point&
PointWidget::GetPosition() const
{
   return variables[PointVar]->point();
}

