
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
  direction(0, 0, 1.0), crittype(AttractingFocus)
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
   picks[Pick] = new GeomPick(arr, module);
   picks[Pick]->set_highlight(HighlightMaterial);
   picks[Pick]->set_cbdata((void*)Pick);
   CreateModeSwitch(0, picks[Pick]);
   
   GeomGroup* cyls = new GeomGroup;
   geometries[GeomCylinder1] = new GeomCylinder;
   cyls->add(geometries[GeomCylinder1]);
   geometries[GeomCylinder2] = new GeomCylinder;
   cyls->add(geometries[GeomCylinder2]);
   geometries[GeomCylinder3] = new GeomCylinder;
   cyls->add(geometries[GeomCylinder3]);
   geometries[GeomCylinder4] = new GeomCylinder;
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
   GeomMaterial* conesm = new GeomMaterial(cones, SpecialMaterial);
   CreateModeSwitch(3, conesm);

   SetMode(Mode1, Switch0);
   SetMode(Mode2, Switch0|Switch1|Switch3);
   SetMode(Mode3, Switch0|Switch1|Switch3);
   SetMode(Mode4, Switch0|Switch1|Switch3);
   SetMode(Mode5, Switch0|Switch2|Switch3);
   SetMode(Mode6, Switch0|Switch2|Switch3);
   SetMode(Mode7, Switch0|Switch2|Switch3);

   FinishWidget();
}


CriticalPointWidget::~CriticalPointWidget()
{
}


void
CriticalPointWidget::widget_execute()
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
						 widget_scale, 0);

   Real cenoff(1.5), twocenoff(cenoff*2), diam(0.6);
   Vector v1, v2;
   direction.normal().find_orthogonal(v1,v2);
   ((GeomCylinder*)geometries[GeomCylinder1])->move(variables[PointVar]->point(),
						    variables[PointVar]->point()
						    +v2*twocenoff*widget_scale,
						    diam*widget_scale);
   ((GeomCylinder*)geometries[GeomCylinder2])->move(variables[PointVar]->point(),
						    variables[PointVar]->point()
						    +v1*twocenoff*widget_scale,
						    diam*widget_scale);
   ((GeomCylinder*)geometries[GeomCylinder3])->move(variables[PointVar]->point(),
						    variables[PointVar]->point()
						    -v2*twocenoff*widget_scale,
						    diam*widget_scale);
   ((GeomCylinder*)geometries[GeomCylinder4])->move(variables[PointVar]->point(),
						    variables[PointVar]->point()
						    -v1*twocenoff*widget_scale,
						    diam*widget_scale);

   ((GeomTorusArc*)geometries[GeomTorus1])->move(variables[PointVar]->point()
						 +v2*cenoff*widget_scale,
						 direction,
						 cenoff*widget_scale, diam*widget_scale,
						 v1, 0, 3.14159);
   ((GeomTorusArc*)geometries[GeomTorus2])->move(variables[PointVar]->point()
						 +v1*cenoff*widget_scale,
						 direction,
						 cenoff*widget_scale, diam*widget_scale,
						 v1, 3.14159*0.5, 3.14159);
   ((GeomTorusArc*)geometries[GeomTorus3])->move(variables[PointVar]->point()
						 -v2*cenoff*widget_scale,
						 direction,
						 cenoff*widget_scale, diam*widget_scale,
						 v1, 3.14159, 3.14159);
   ((GeomTorusArc*)geometries[GeomTorus4])->move(variables[PointVar]->point()
						 -v1*cenoff*widget_scale,
						 direction,
						 cenoff*widget_scale, diam*widget_scale,
						 v1, 3.14159*1.5, 3.14159);

   Real conelen(1.5), conerad(0.8);
   ((GeomCappedCone*)geometries[GeomCone1])->move(variables[PointVar]->point()
						  +v2*twocenoff*widget_scale,
						  variables[PointVar]->point()
						  +v2*twocenoff*widget_scale+v1*conelen*widget_scale,
						  conerad*widget_scale, 0);
   ((GeomCappedCone*)geometries[GeomCone2])->move(variables[PointVar]->point()
						  +v1*twocenoff*widget_scale,
						  variables[PointVar]->point()
						  +v1*twocenoff*widget_scale-v2*conelen*widget_scale,
						  conerad*widget_scale, 0);
   ((GeomCappedCone*)geometries[GeomCone3])->move(variables[PointVar]->point()
						  -v2*twocenoff*widget_scale,
						  variables[PointVar]->point()
						  -v2*twocenoff*widget_scale-v1*conelen*widget_scale,
						  conerad*widget_scale, 0);
   ((GeomCappedCone*)geometries[GeomCone4])->move(variables[PointVar]->point()
						  -v1*twocenoff*widget_scale,
						  variables[PointVar]->point()
						  -v1*twocenoff*widget_scale+v2*conelen*widget_scale,
						  conerad*widget_scale, 0);

   for (Index geom = 0; geom < NumPcks; geom++) {
      picks[geom]->set_principal(direction, v1, v2);
   }
}


void
CriticalPointWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			 void* cbdata )
{
   switch((int)cbdata){
   case Pick:
      MoveDelta(delta);
      break;
   }
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


