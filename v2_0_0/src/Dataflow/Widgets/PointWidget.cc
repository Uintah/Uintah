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
 *  PointWidget.cc
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Dataflow/Widgets/PointWidget.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Network/Module.h>

namespace SCIRun {



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

/***************************************************************************
 * The constructor initializes the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Variables and constraints are initialized as a function of the
 *      widget_scale.
 * Much of the work is accomplished in the BaseWidget constructor which
 *      includes some consistency checking to ensure full initialization.
 */
PointWidget::PointWidget( Module* module, CrowdMonitor* lock, double widget_scale )
  : BaseWidget(module, lock, "PointWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale)
{
  variables[PointVar] = scinew PointVariable("Point", solve, Scheme1, Point(0, 0, 0));

  geometries[GeomPoint] = scinew GeomSphere;
  materials[PointMatl] = scinew GeomMaterial(geometries[GeomPoint], DefaultPointMaterial);
  picks_[Pick] = scinew GeomPick(materials[PointMatl], module, this, Pick);
  picks(Pick)->set_highlight(DefaultHighlightMaterial);
  CreateModeSwitch(0, picks_[Pick]);

  SetMode(Mode0, Switch0);
   
  FinishWidget();
}


/***************************************************************************
 * The destructor frees the widget's allocated structures.
 * The BaseWidget's destructor frees the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Therefore, most widgets' destructors will not need to do anything.
 */
PointWidget::~PointWidget()
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
PointWidget::redraw()
{
  if (mode_switches[0]->get_state())
  {
    geometry<GeomSphere*>(GeomPoint)->move(variables[PointVar]->point(),
					   widget_scale_);
  }
}


void
PointWidget::geom_pick(GeomPickHandle p, ViewWindow *vw,
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
PointWidget::geom_moved( GeomPickHandle, int /* axis */, double /*dist*/,
			 const Vector& /*delta*/, int pick, const BState&,
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


/***************************************************************************
 * This standard method simply moves all the widget's PointVariables by
 *      the same delta.
 * The last line of this method should call the BaseWidget execute method
 *      (which calls the redraw method).
 */
void
PointWidget::MoveDelta( const Vector& delta )
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


/***************************************************************************
 * This standard method returns a string describing the functionality of
 *      a widget's material property.  The string is used in the 
 *      BaseWidget UI.
 */
string
PointWidget::GetMaterialName( const Index mindex ) const
{
  ASSERT(mindex<materials.size());
   
  switch(mindex)
  {
  case 0:
    return "Point";
  default:
    return "UnknownMaterial";
  }
}


void
PointWidget::widget_tcl( GuiArgs& args )
{
  if (args[1] == "translate")
  {
    if (args.count() != 4)
    {
      args.error("point widget needs axis translation");
      return;
    }
    double trans;
    if (!string_to_double(args[3], trans))
    {
      args.error("point widget can't parse translation `"+args[3]+"'");
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
      args.error("point widget unknown axis `"+args[2]+"'");
      break;
    }
    SetPosition(p);
  }
}

} // End namespace SCIRun

