
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
const Index NumPcks = 1;
const Index NumMdes = 1;
const Index NumSwtchs = 1;
// const Index NumSchemes = 1;

enum { GeomPoint };
enum { Pick };

PointWidget::PointWidget( Module* module, CrowdMonitor* lock, double widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumPcks, NumMdes, NumSwtchs, widget_scale)
{
   variables[PointVar] = new PointVariable("Point", solve, Scheme1, Point(0, 0, 0));

   geometries[GeomPoint] = new GeomSphere;
   sphMaterial = new GeomMaterial(geometries[GeomPoint], PointMaterial);
   picks[Pick] = new GeomPick(sphMaterial, module, this, Pick);
   picks[Pick]->set_highlight(HighlightMaterial);
   CreateModeSwitch(0, picks[Pick]);

   SetMode(Mode0, Switch0);
   
   FinishWidget();
}


PointWidget::~PointWidget()
{
}


void
PointWidget::widget_execute()
{
   if (mode_switches[0]->get_state())
      ((GeomSphere*)geometries[GeomPoint])->move(variables[PointVar]->point(),
						 widget_scale);
}


void
PointWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			 int cbdata )
{
   switch(cbdata){
   case Pick:
      MoveDelta(delta);
      break;
   }
   execute();
}


void
PointWidget::MoveDelta( const Vector& delta )
{
   variables[PointVar]->MoveDelta(delta);

   execute();
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


Point
PointWidget::GetPosition() const
{
   return variables[PointVar]->point();
}

void 
PointWidget::SetMaterial( const MaterialHandle& copy)
{
    sphMaterial->setMaterial(copy);
    execute();
}

MaterialHandle 
PointWidget::GetMaterial()
{
    return sphMaterial->getMaterial();
}

