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
 *  CrosshairWidget.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Dataflow/Widgets/CrosshairWidget.h>
#include <Core/Geom/GeomCylinder.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Network/Module.h>

namespace SCIRun {


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

/***************************************************************************
 * The constructor initializes the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Variables and constraints are initialized as a function of the
 *      widget_scale.
 * Much of the work is accomplished in the BaseWidget constructor which
 *      includes some consistency checking to ensure full initialization.
 */
CrosshairWidget::CrosshairWidget( Module* module, CrowdMonitor* lock, double widget_scale )
  : BaseWidget(module, lock, "CrosshairWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale),
    axis1(1, 0, 0), axis2(0, 1, 0), axis3(0, 0, 1)
{
  variables[CenterVar] = scinew PointVariable("Crosshair", solve, Scheme1, Point(0, 0, 0));

  geometries[GeomCenter] = scinew GeomSphere;
  materials[PointMatl] = scinew GeomMaterial(geometries[GeomCenter], DefaultPointMaterial);
  picks_[PickAxes] = scinew GeomPick(materials[PointMatl], module, this, PickAxes);
  picks(PickAxes)->set_highlight(DefaultHighlightMaterial);
  CreateModeSwitch(0, picks_[PickAxes]);

  GeomGroup* axes = scinew GeomGroup;
  geometries[GeomAxis1] = scinew GeomCappedCylinder;
  axes->add(geometries[GeomAxis1]);
  geometries[GeomAxis2] = scinew GeomCappedCylinder;
  axes->add(geometries[GeomAxis2]);
  geometries[GeomAxis3] = scinew GeomCappedCylinder;
  axes->add(geometries[GeomAxis3]);
  materials[AxesMatl] = scinew GeomMaterial(axes, DefaultEdgeMaterial);
  picks_[Pick] = scinew GeomPick(materials[AxesMatl], module, this, Pick);
  picks(Pick)->set_highlight(DefaultHighlightMaterial);
  CreateModeSwitch(1, picks_[Pick]);

  SetMode(Mode0, Switch0|Switch1);
  SetMode(Mode1, Switch0);

  FinishWidget();
}


/***************************************************************************
 * The destructor frees the widget's allocated structures.
 * The BaseWidget's destructor frees the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Therefore, most widgets' destructors will not need to do anything.
 */
CrosshairWidget::~CrosshairWidget()
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
CrosshairWidget::redraw()
{
  Point center(variables[CenterVar]->point());
   
  if (mode_switches[0]->get_state())
    geometry<GeomSphere*>(GeomCenter)->move(center, widget_scale_);

  if (mode_switches[1]->get_state())
  {
    const double axislen(100.0*widget_scale_);
    const double axisrad(0.5*widget_scale_);
    geometry<GeomCappedCylinder*>(GeomAxis1)->move(center - (axis1 * axislen),
						   center + (axis1 * axislen),
						   axisrad);
    geometry<GeomCappedCylinder*>(GeomAxis2)->move(center - (axis2 * axislen),
						   center + (axis2 * axislen),
						   axisrad);
    geometry<GeomCappedCylinder*>(GeomAxis3)->move(center - (axis3 * axislen),
						   center + (axis3 * axislen),
						   axisrad);
  }

  for (Index geom = 0; geom < NumPcks; geom++)
  {
    picks(geom)->set_principal(axis1, axis2, axis3);
  }
}

void
CrosshairWidget::geom_pick(GeomPickHandle p, ViewWindow *vw,
			   int data, const BState &bs)
{
  BaseWidget::geom_pick(p, vw, data, bs);
  pick_position_ = variables[CenterVar]->point();
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
CrosshairWidget::geom_moved( GeomPickHandle, int /* axis */, double /* dist */,
			     const Vector& /*delta*/, int pick, const BState&,
			     const Vector &pick_offset)
{
  switch(pick)
  {
  case Pick:
  case PickAxes:
    variables[CenterVar]->Move(pick_position_);
    MoveDelta(pick_offset);
    break;
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
CrosshairWidget::MoveDelta( const Vector& delta )
{
  variables[CenterVar]->MoveDelta(delta);

  execute(1);
}


/***************************************************************************
 * This standard method returns a reference point for the widget.  This
 *      point should have some logical meaning such as the center of the
 *      widget's geometry.
 */
Point
CrosshairWidget::ReferencePoint() const
{
  return variables[CenterVar]->point();
}


void
CrosshairWidget::SetPosition( const Point& p )
{
  variables[CenterVar]->Move(p);

  execute(0);
}


Point
CrosshairWidget::GetPosition() const
{
  return variables[CenterVar]->point();
}


void
CrosshairWidget::SetAxes( const Vector& v1, const Vector& v2, const Vector& v3 )
{
  if ((v1.length2() > 1e-6) &&
      (v2.length2() > 1e-6) &&
      (v3.length2() > 1e-6))
  {
    axis1 = v1.normal();
    axis2 = v2.normal();
    axis3 = v3.normal();

    execute(0);
  }
}


void
CrosshairWidget::GetAxes( Vector& v1, Vector& v2, Vector& v3 ) const
{
  v1 = axis1;
  v2 = axis2;
  v3 = axis3;
}


/***************************************************************************
 * This standard method returns a string describing the functionality of
 *      a widget's material property.  The string is used in the 
 *      BaseWidget UI.
 */
string
CrosshairWidget::GetMaterialName( const Index mindex ) const
{
  ASSERT(mindex<materials.size());
   
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
CrosshairWidget::widget_tcl( GuiArgs& args )
{
  if (args[1] == "translate")
  {
    if (args.count() != 4)
    {
      args.error("crosshair widget needs axis translation");
      return;
    }
    double trans;
    if (!string_to_double(args[3], trans))
    {
      args.error("crosshair widget can't parse translation `"+args[3]+"'");
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
      args.error("crosshair widget unknown axis `"+args[2]+"'");
      break;
    }
    SetPosition(p);
  }
}

} // End namespace SCIRun

