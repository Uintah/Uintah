
/*
 *  CriticalPointWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Widgets/CriticalPointWidget.h>
#include <Geom/Cone.h>
#include <Geom/Cylinder.h>
#include <Geom/Torus.h>
#include <Geom/Sphere.h>

const Index NumCons = 0;
const Index NumVars = 1;
const Index NumGeoms = 15;
const Index NumPcks = 1;
const Index NumMdes = 7;
const Index NumSwtchs = 4;
// const Index NumSchemes = 1;

enum { GeomPoint, GeomShaft, GeomHead,
       GeomCone1, GeomCone2, GeomCone3, GeomCone4,
       GeomCylinder1, GeomCylinder2, GeomCylinder3, GeomCylinder4,
       GeomTorus1, GeomTorus2, GeomTorus3, GeomTorus4 };
enum { Pick };

CriticalPointWidget::CriticalPointWidget( Module* module, CrowdMonitor* lock, double widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumPcks, NumMdes, NumSwtchs, widget_scale),
  direction(0, 0, 1.0), crittype(Regular)
{
   variables[PointVar] = new PointVariable("Point", solve, Scheme1, Point(0, 0, 0));

   GeomGroup* arr = new GeomGroup;
   geometries[GeomPoint] = new GeomSphere;
   GeomMaterial* sphm = new GeomMaterial(geometries[GeomPoint], PointMaterial);
   arr->add(sphm);
   geometries[GeomShaft] = new GeomCylinder;
   GeomMaterial* cylm = new GeomMaterial(geometries[GeomShaft], EdgeMaterial);
   arr->add(cylm);
   geometries[GeomHead] = new GeomCappedCone;
   GeomMaterial* conem = new GeomMaterial(geometries[GeomHead], EdgeMaterial);
   arr->add(conem);
   picks[Pick] = new GeomPick(arr, module, this, Pick);
   picks[Pick]->set_highlight(HighlightMaterial);
   CreateModeSwitch(0, picks[Pick]);
   
   GeomGroup* cyls = new GeomGroup;
   geometries[GeomCylinder1] = new GeomCappedCylinder;
   cyls->add(geometries[GeomCylinder1]);
   geometries[GeomCylinder2] = new GeomCappedCylinder;
   cyls->add(geometries[GeomCylinder2]);
   geometries[GeomCylinder3] = new GeomCappedCylinder;
   cyls->add(geometries[GeomCylinder3]);
   geometries[GeomCylinder4] = new GeomCappedCylinder;
   cyls->add(geometries[GeomCylinder4]);
   GeomMaterial* cylsm = new GeomMaterial(cyls, SpecialMaterial);
   CreateModeSwitch(1, cylsm);

   GeomGroup* torii = new GeomGroup;
   geometries[GeomTorus1] = new GeomTorusArc;
   torii->add(geometries[GeomTorus1]);
   geometries[GeomTorus2] = new GeomTorusArc;
   torii->add(geometries[GeomTorus2]);
   geometries[GeomTorus3] = new GeomTorusArc;
   torii->add(geometries[GeomTorus3]);
   geometries[GeomTorus4] = new GeomTorusArc;
   torii->add(geometries[GeomTorus4]);
   GeomMaterial* torusm = new GeomMaterial(torii, SpecialMaterial);
   CreateModeSwitch(2, torusm);

   GeomGroup* cones = new GeomGroup;
   geometries[GeomCone1] = new GeomCappedCone;
   cones->add(geometries[GeomCone1]);
   geometries[GeomCone2] = new GeomCappedCone;
   cones->add(geometries[GeomCone2]);
   geometries[GeomCone3] = new GeomCappedCone;
   cones->add(geometries[GeomCone3]);
   geometries[GeomCone4] = new GeomCappedCone;
   cones->add(geometries[GeomCone4]);
   GeomMaterial* conesm = new GeomMaterial(cones, ResizeMaterial);
   CreateModeSwitch(3, conesm);

   SetMode(Mode0, Switch0);
   SetMode(Mode1, Switch0|Switch1|Switch3);
   SetMode(Mode2, Switch0|Switch1|Switch3);
   SetMode(Mode3, Switch0|Switch1|Switch3);
   SetMode(Mode4, Switch0|Switch2|Switch3);
   SetMode(Mode5, Switch0|Switch2|Switch3);
   SetMode(Mode6, Switch0|Switch2|Switch3);

   FinishWidget();
}


CriticalPointWidget::~CriticalPointWidget()
{
}


void
CriticalPointWidget::widget_execute()
{
   Vector direct(direction);
   Real extent(4.5*widget_scale);
   Real sphererad(widget_scale), cylinderrad(0.5*widget_scale);
   Real twocenoff(extent-cylinderrad), cenoff(twocenoff/2.0), diam(0.6*widget_scale);
   Real conelen(1.5*widget_scale), conerad(0.8*widget_scale), cyllen(extent-conelen);
   Point center(variables[PointVar]->point());
   Vector v1, v2;
   direct.normal().find_orthogonal(v1,v2);

   Point cylinder1end1, cylinder1end2;
   Point cylinder2end1, cylinder2end2;
   Point cylinder3end1, cylinder3end2;
   Point cylinder4end1, cylinder4end2;
   Point torus1center, torus2center, torus3center, torus4center;
   Real torus1start, torus2start, torus3start, torus4start, torusangle(3.14159);
   Point cone1end1, cone1end2;
   Point cone2end1, cone2end2;
   Point cone3end1, cone3end2;
   Point cone4end1, cone4end2;
   
   switch (crittype) {
   case Regular:
      break;
   case AttractingNode:
      cone1end2 = center+v2*sphererad;
      cone1end1 = cone1end2+v2*conelen;
      cone2end2 = center+v1*sphererad;
      cone2end1 = cone2end2+v1*conelen;
      cone3end2 = center-v2*sphererad;
      cone3end1 = cone3end2-v2*conelen;
      cone4end2 = center-v1*sphererad;
      cone4end1 = cone4end2-v1*conelen;
      cylinder1end1 = cone1end1;
      cylinder1end2 = center+v2*extent;
      cylinder2end1 = cone2end1;
      cylinder2end2 = center+v1*extent;
      cylinder3end1 = cone3end1;
      cylinder3end2 = center-v2*extent;
      cylinder4end1 = cone4end1;
      cylinder4end2 = center-v1*extent;
      break;
   case RepellingNode:
      cylinder1end1 = center;
      cylinder1end2 = center+v2*cyllen;
      cylinder2end1 = center;
      cylinder2end2 = center+v1*cyllen;
      cylinder3end1 = center;
      cylinder3end2 = center-v2*cyllen;
      cylinder4end1 = center;
      cylinder4end2 = center-v1*cyllen;
      cone1end1 = cylinder1end2;
      cone1end2 = cone1end1+v2*conelen;
      cone2end1 = cylinder2end2;
      cone2end2 = cone2end1+v1*conelen;
      cone3end1 = cylinder3end2;
      cone3end2 = cone3end1-v2*conelen;
      cone4end1 = cylinder4end2;
      cone4end2 = cone4end1-v1*conelen;
      break;
   case Saddle:
      cylinder2end1 = center;
      cylinder2end2 = center+v1*cyllen;
      cylinder4end1 = center;
      cylinder4end2 = center-v1*cyllen;
      cone1end2 = center+v2*sphererad;
      cone1end1 = cone1end2+v2*conelen;
      cone2end1 = cylinder2end2;
      cone2end2 = cone2end1+v1*conelen;
      cone3end2 = center-v2*sphererad;
      cone3end1 = cone3end2-v2*conelen;
      cone4end1 = cylinder4end2;
      cone4end2 = cone4end1-v1*conelen;
      cylinder1end1 = cone1end1;
      cylinder1end2 = center+v2*extent;
      cylinder3end1 = cone3end1;
      cylinder3end2 = center-v2*extent;
      break;
   case AttractingFocus:
      Real weird(3.14159-2.2);
      torus1center = center+v2*cenoff;
      torus1start = 0+weird;
      torus2center = center+v1*cenoff;
      torus2start = 3.14159*0.5+weird;
      torus3center = center-v2*cenoff;
      torus3start = 3.14159+weird;
      torus4center = center-v1*cenoff;
      torus4start = 3.14159*1.5+weird;
      cone1end2 = center+v2*sphererad;
      cone1end1 = cone1end2+(v2+v1*1.4)/2.4*conelen;
      cone2end2 = center+v1*sphererad;
      cone2end1 = cone2end2+(v1-v2*1.4)/2.4*conelen;
      cone3end2 = center-v2*sphererad;
      cone3end1 = cone3end2-(v2+v1*1.4)/2.4*conelen;
      cone4end2 = center-v1*sphererad;
      cone4end1 = cone4end2-(v1-v2*1.4)/2.4*conelen;
      break;
   case RepellingFocus:
      torus1center = center+v2*cenoff;
      torus1start = 0;
      torus2center = center+v1*cenoff;
      torus2start = 3.14159*0.5;
      torus3center = center-v2*cenoff;
      torus3start = 3.14159;
      torus4center = center-v1*cenoff;
      torus4start = 3.14159*1.5;
      cone1end1 = center+v2*twocenoff;
      cone1end2 = cone1end1+v1*conelen;
      cone2end1 = center+v1*twocenoff;
      cone2end2 = cone2end1-v2*conelen;
      cone3end1 = center-v2*twocenoff;
      cone3end2 = cone3end1-v1*conelen;
      cone4end1 = center-v1*twocenoff;
      cone4end2 = cone4end1+v2*conelen;
      break;
   case SpiralSaddle:
      weird = 3.14159-2.2;
      torus1center = center+v2*cenoff;
      torus1start = 0+weird;
      torus2center = center+v1*cenoff;
      torus2start = 3.14159*0.5;
      torus3center = center-v2*cenoff;
      torus3start = 3.14159+weird;
      torus4center = center-v1*cenoff;
      torus4start = 3.14159*1.5;
      cone1end1 = center+v1*twocenoff;
      cone1end2 = cone1end1-v2*conelen;
      cone2end2 = center+v1*sphererad;
      cone2end1 = cone2end2+(v1-v2*1.4)/2.4*conelen;
      cone3end1 = center-v1*twocenoff;
      cone3end2 = cone3end1+v2*conelen;
      cone4end2 = center-v1*sphererad;
      cone4end1 = cone4end2-(v1-v2*1.4)/2.4*conelen;
      break;
   }

   if (mode_switches[0]->get_state()) {
      ((GeomSphere*)geometries[GeomPoint])->move(center, sphererad);
      ((GeomCylinder*)geometries[GeomShaft])->move(center, center+direct*twocenoff, cylinderrad);
      ((GeomCappedCone*)geometries[GeomHead])->move(center+direct*twocenoff,
						    center+direct*(twocenoff+2.0*widget_scale),
						    sphererad, 0);
   }
   
   if (mode_switches[1]->get_state()) {
      ((GeomCappedCylinder*)geometries[GeomCylinder1])->move(cylinder1end1, cylinder1end2, diam);
      ((GeomCappedCylinder*)geometries[GeomCylinder2])->move(cylinder2end1, cylinder2end2, diam);
      ((GeomCappedCylinder*)geometries[GeomCylinder3])->move(cylinder3end1, cylinder3end2, diam);
      ((GeomCappedCylinder*)geometries[GeomCylinder4])->move(cylinder4end1, cylinder4end2, diam);
   }
   if (mode_switches[2]->get_state()) {
      ((GeomTorusArc*)geometries[GeomTorus1])->move(torus1center, direct, cenoff, diam, v1,
						    torus1start, torusangle);
      ((GeomTorusArc*)geometries[GeomTorus2])->move(torus2center, direct, cenoff, diam, v1,
						    torus2start, torusangle);
      ((GeomTorusArc*)geometries[GeomTorus3])->move(torus3center, direct, cenoff, diam, v1,
						    torus3start, torusangle);
      ((GeomTorusArc*)geometries[GeomTorus4])->move(torus4center, direct, cenoff, diam, v1,
						    torus4start, torusangle);
   }
   if (mode_switches[3]->get_state()) {
      ((GeomCappedCone*)geometries[GeomCone1])->move(cone1end1, cone1end2, conerad, 0);
      ((GeomCappedCone*)geometries[GeomCone2])->move(cone2end1, cone2end2, conerad, 0);
      ((GeomCappedCone*)geometries[GeomCone3])->move(cone3end1, cone3end2, conerad, 0);
      ((GeomCappedCone*)geometries[GeomCone4])->move(cone4end1, cone4end2, conerad, 0);
   }
   
   for (Index geom = 0; geom < NumPcks; geom++) {
      picks[geom]->set_principal(direct, v1, v2);
   }
}


void
CriticalPointWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
				int cbdata )	
{
   switch((int)cbdata){
   case Pick:
      MoveDelta(delta);
      break;
   }
   execute();
}


void
CriticalPointWidget::NextMode()
{
   Index s;
   for (s=0; s<NumSwitches; s++)
      if (modes[CurrentMode]&(1<<s))
	 mode_switches[s]->set_state(0);
   CurrentMode = (CurrentMode+1) % NumModes;
   crittype = (CriticalType)((crittype+1) % NumCriticalTypes);
   for (s=0; s<NumSwitches; s++)
      if (modes[CurrentMode]&(1<<s))
	 mode_switches[s]->set_state(1);

   execute();
}


void
CriticalPointWidget::MoveDelta( const Vector& delta )
{
   variables[PointVar]->MoveDelta(delta);

   execute();
}


Point
CriticalPointWidget::ReferencePoint() const
{
   return variables[PointVar]->point();
}


void
CriticalPointWidget::SetCriticalType( const CriticalType crit )
{
   crittype = crit;

   execute();
}


Index
CriticalPointWidget::GetCriticalType() const
{
   return crittype;
}


void
CriticalPointWidget::SetPosition( const Point& p )
{
   variables[PointVar]->Move(p);

   execute();
}


const Point&
CriticalPointWidget::GetPosition() const
{
   return variables[PointVar]->point();
}


void
CriticalPointWidget::SetDirection( const Vector& v )
{
   direction = v;

   execute();
}


const Vector&
CriticalPointWidget::GetDirection() const
{
   return direction;
}


