/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  CriticalPointWidget.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Feb. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Dataflow/Widgets/CriticalPointWidget.h>
#include <Core/Geom/GeomCone.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomTorus.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>

namespace SCIRun {



const Index NumCons = 0;
const Index NumVars = 1;
const Index NumGeoms = 15;
const Index NumPcks = 1;
const Index NumMatls = 6;
const Index NumMdes = 7;
const Index NumSwtchs = 4;
// const Index NumSchemes = 1;

enum { GeomPoint, GeomShaft, GeomHead,
       GeomCone1, GeomCone2, GeomCone3, GeomCone4,
       GeomCylinder1, GeomCylinder2, GeomCylinder3, GeomCylinder4,
       GeomTorus1, GeomTorus2, GeomTorus3, GeomTorus4 };
enum { Pick };

/***************************************************************************
 * The constructor initializes the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Variables and constraints are initialized as a function of the
 *      widget_scale.
 * Much of the work is accomplished in the BaseWidget constructor which
 *      includes some consistency checking to ensure full initialization.
 */
CriticalPointWidget::CriticalPointWidget( Module* module, CrowdMonitor* lock, double widget_scale )
  : BaseWidget(module, lock, "CriticalPointWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale),
    crittype(Regular), direction(0, 0, 1.0)
{
  variables[PointVar] = scinew PointVariable("Point", solve, Scheme1, Point(0, 0, 0));

  GeomGroup* arr = scinew GeomGroup;
  geometries[GeomPoint] = scinew GeomSphere;
  materials[PointMaterial] = scinew GeomMaterial(geometries[GeomPoint], DefaultPointMaterial);
  arr->add(materials[PointMaterial]);
  geometries[GeomShaft] = scinew GeomCylinder;
  materials[ShaftMaterial] = scinew GeomMaterial(geometries[GeomShaft], DefaultEdgeMaterial);
  arr->add(materials[ShaftMaterial]);
  geometries[GeomHead] = scinew GeomCappedCone;
  materials[HeadMaterial] = scinew GeomMaterial(geometries[GeomHead], DefaultEdgeMaterial);
  arr->add(materials[HeadMaterial]);
  picks_[Pick] = scinew GeomPick(arr, module, this, Pick);
  picks(Pick)->set_highlight(DefaultHighlightMaterial);
  CreateModeSwitch(0, picks_[Pick]);
   
  GeomGroup* cyls = scinew GeomGroup;
  geometries[GeomCylinder1] = scinew GeomCappedCylinder;
  cyls->add(geometries[GeomCylinder1]);
  geometries[GeomCylinder2] = scinew GeomCappedCylinder;
  cyls->add(geometries[GeomCylinder2]);
  geometries[GeomCylinder3] = scinew GeomCappedCylinder;
  cyls->add(geometries[GeomCylinder3]);
  geometries[GeomCylinder4] = scinew GeomCappedCylinder;
  cyls->add(geometries[GeomCylinder4]);
  materials[CylinderMatl] = scinew GeomMaterial(cyls, DefaultSpecialMaterial);
  CreateModeSwitch(1, materials[CylinderMatl]);

  GeomGroup* torii = scinew GeomGroup;
  geometries[GeomTorus1] = scinew GeomTorusArc;
  torii->add(geometries[GeomTorus1]);
  geometries[GeomTorus2] = scinew GeomTorusArc;
  torii->add(geometries[GeomTorus2]);
  geometries[GeomTorus3] = scinew GeomTorusArc;
  torii->add(geometries[GeomTorus3]);
  geometries[GeomTorus4] = scinew GeomTorusArc;
  torii->add(geometries[GeomTorus4]);
  materials[TorusMatl] = scinew GeomMaterial(torii, DefaultSpecialMaterial);
  CreateModeSwitch(2, materials[TorusMatl]);

  GeomGroup* cones = scinew GeomGroup;
  geometries[GeomCone1] = scinew GeomCappedCone;
  cones->add(geometries[GeomCone1]);
  geometries[GeomCone2] = scinew GeomCappedCone;
  cones->add(geometries[GeomCone2]);
  geometries[GeomCone3] = scinew GeomCappedCone;
  cones->add(geometries[GeomCone3]);
  geometries[GeomCone4] = scinew GeomCappedCone;
  cones->add(geometries[GeomCone4]);
  materials[ConeMatl] = scinew GeomMaterial(cones, DefaultResizeMaterial);
  CreateModeSwitch(3, materials[ConeMatl]);

  SetMode(Mode0, Switch0);
  SetMode(Mode1, Switch0|Switch1|Switch3);
  SetMode(Mode2, Switch0|Switch1|Switch3);
  SetMode(Mode3, Switch0|Switch1|Switch3);
  SetMode(Mode4, Switch0|Switch2|Switch3);
  SetMode(Mode5, Switch0|Switch2|Switch3);
  SetMode(Mode6, Switch0|Switch2|Switch3);

  FinishWidget();
}


/***************************************************************************
 * The destructor frees the widget's allocated structures.
 * The BaseWidget's destructor frees the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Therefore, most widgets' destructors will not need to do anything.
 */
CriticalPointWidget::~CriticalPointWidget()
{
}


/***************************************************************************
 * The widget's redraw method changes widget geometry to reflect the
 *      widget's variable values and its widget_scale.
 * Geometry should only be changed if the mode_switch that displays
 *      that geometry is active.
 * Redraw should also set the principal directions for all picks.
 * Redraw should never be called directly; the BaseWidget execute method
 *      calls redraw after establishing the appropriate locks.
 */
void
CriticalPointWidget::redraw()
{
  const Vector direct(direction);
  const double extent(4.5*widget_scale_);
  const double sphererad(widget_scale_), cylinderrad(0.5*widget_scale_);
  const double twocenoff(extent-cylinderrad), cenoff(twocenoff/2.0), rad(0.6*widget_scale_);
  const double conelen(1.5*widget_scale_), conerad(0.8*widget_scale_), cyllen(extent-conelen);
  const Point center(variables[PointVar]->point());
  Vector v1, v2;
  direct.normal().find_orthogonal(v1,v2);

  Point cylinder1end1, cylinder1end2;
  Point cylinder2end1, cylinder2end2;
  Point cylinder3end1, cylinder3end2;
  Point cylinder4end1, cylinder4end2;
  Point torus1center, torus2center, torus3center, torus4center;
  double torus1start=0, torus2start=0, torus3start=0, torus4start=0,
    torusangle(3.14159);
  Point cone1end1, cone1end2;
  Point cone2end1, cone2end2;
  Point cone3end1, cone3end2;
  Point cone4end1, cone4end2;
   
  switch (crittype)
  {
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
    {
      double weird(3.14159-2.2);
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
    }
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
    {
      const double weird = 3.14159-2.2;
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
    }
    break;

  default:
    break;
  }

  if (mode_switches[0]->get_state())
  {
    geometry<GeomSphere*>(GeomPoint)->move(center, sphererad);
    geometry<GeomCylinder*>(GeomShaft)->move(center, center+direct*twocenoff, cylinderrad);
    geometry<GeomCappedCone*>(GeomHead)->move(center+direct*twocenoff,
					      center+direct*(twocenoff+2.0*widget_scale_),
					      sphererad, 0);
  }
   
  if (mode_switches[1]->get_state())
  {
    geometry<GeomCappedCylinder*>(GeomCylinder1)->move(cylinder1end1, cylinder1end2, rad);
    geometry<GeomCappedCylinder*>(GeomCylinder2)->move(cylinder2end1, cylinder2end2, rad);
    geometry<GeomCappedCylinder*>(GeomCylinder3)->move(cylinder3end1, cylinder3end2, rad);
    geometry<GeomCappedCylinder*>(GeomCylinder4)->move(cylinder4end1, cylinder4end2, rad);
  }
  if (mode_switches[2]->get_state())
  {
    geometry<GeomTorusArc*>(GeomTorus1)->move(torus1center, direct,
					      cenoff, rad, v1,
					      torus1start, torusangle);
    geometry<GeomTorusArc*>(GeomTorus2)->move(torus2center, direct,
					      cenoff, rad, v1,
					      torus2start, torusangle);
    geometry<GeomTorusArc*>(GeomTorus3)->move(torus3center, direct,
					      cenoff, rad, v1,
					      torus3start, torusangle);
    geometry<GeomTorusArc*>(GeomTorus4)->move(torus4center, direct,
					      cenoff, rad, v1,
					      torus4start, torusangle);
  }
  if (mode_switches[3]->get_state())
  {
    geometry<GeomCappedCone*>(GeomCone1)->move(cone1end1, cone1end2, conerad, 0);
    geometry<GeomCappedCone*>(GeomCone2)->move(cone2end1, cone2end2, conerad, 0);
    geometry<GeomCappedCone*>(GeomCone3)->move(cone3end1, cone3end2, conerad, 0);
    geometry<GeomCappedCone*>(GeomCone4)->move(cone4end1, cone4end2, conerad, 0);
  }
   
  for (Index geom = 0; geom < NumPcks; geom++)
  {
    picks(geom)->set_principal(direct, v1, v2);
  }
}


void
CriticalPointWidget::geom_pick(GeomPickHandle p, ViewWindow *vw,
			       int data, const BState &bs)
{
  BaseWidget::geom_pick(p, vw, data, bs);
  pick_position_ = variables[PointVar]->point();
}



/***************************************************************************
 * The widget's geom_moved method receives geometry move requests from
 *      the widget's picks.  The widget's variables must be altered to
 *      reflect these changes based upon which pick made the request.
 * No more than one variable should be Set since this triggers solution of
 *      the constraints--multiple Sets could lead to inconsistencies.
 *      The constraint system only requires that a variable be Set if the
 *      change would cause a constraint to be invalid.  For example, if
 *      all PointVariables are moved by the same delta, then no Set is
 *      required.
 * The last line of the widget's geom_moved method should call the
 *      BaseWidget execute method (which calls the redraw method).
 */
void
CriticalPointWidget::geom_moved( GeomPickHandle, int /* axis */, 
				 double /* dist */,
				 const Vector& /*delta*/, int pick, 
				 const BState&,
				 const Vector &pick_offset)
{
  switch(pick)
  {
  case Pick:
    variables[PointVar]->Move(pick_position_);
    MoveDelta(pick_offset);
    break;
  }
  execute(0);
}


void
CriticalPointWidget::NextMode()
{
  Index s;
  for (s=0; s<mode_switches.size(); s++)
  {
    if (modes[current_mode_]&(1<<s))
    {
      mode_switches[s]->set_state(0);
    }
  }
  current_mode_ = (current_mode_ + 1) % modes.size();
  crittype = (CriticalType)((crittype+1) % NumCriticalTypes);
  for (s=0; s<mode_switches.size(); s++)
  {
    if (modes[current_mode_]&(1<<s))
    {
      mode_switches[s]->set_state(1);
    }
  }

  execute(0);
}


/***************************************************************************
 * This standard method simply moves all the widget's PointVariables by
 *      the same delta.
 * The last line of this method should call the BaseWidget execute method
 *      (which calls the redraw method).
 */
void
CriticalPointWidget::MoveDelta( const Vector& delta )
{
  variables[PointVar]->MoveDelta(delta);

  execute(1);
}


/***************************************************************************
 * This standard method returns a reference point for the widget.  This
 *      point should have some logical meaning such as the center of the
 *      widget's geometry.
 */
Point
CriticalPointWidget::ReferencePoint() const
{
  return variables[PointVar]->point();
}


void
CriticalPointWidget::SetCriticalType( const CriticalType crit )
{
  crittype = crit;

  execute(0);
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

  execute(0);
}


Point
CriticalPointWidget::GetPosition() const
{
  return variables[PointVar]->point();
}


void
CriticalPointWidget::SetDirection( const Vector& v )
{
  direction = v;

  execute(0);
}


const Vector&
CriticalPointWidget::GetDirection() const
{
  return direction;
}


/***************************************************************************
 * This standard method returns a string describing the functionality of
 *      a widget's material property.  The string is used in the 
 *      BaseWidget UI.
 */
string
CriticalPointWidget::GetMaterialName( const Index mindex ) const
{
  ASSERT(mindex<materials.size());
   
  switch(mindex)
  {
  case 0:
    return "Point";
  case 1:
    return "Shaft";
  case 2:
    return "Head";
  case 3:
    return "Cylinder";
  case 4:
    return "Torus";
  case 5:
    return "Cone";
  default:
    return "UnknownMaterial";
  }
}


void
CriticalPointWidget::widget_tcl( GuiArgs& args )
{
  if (args[1] == "translate")
  {
    if (args.count() != 4)
    {
      args.error("criticalpoint widget needs axis translation");
      return;
    }
    double trans;
    if (!string_to_double(args[3], trans))
    {
      args.error("criticalpoint widget can't parse translation `"+args[3]+"'");
      return;
    }
    Point p(GetPosition());
    switch (args[2][0])
    {
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
      args.error("criticalpoint widget unknown axis `"+args[2]+"'");
      break;
    }
    SetPosition(p);
  }
}

} // End namespace SCIRun

