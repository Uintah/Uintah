
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
#include <Malloc/Allocator.h>

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

/***************************************************************************
 * The constructor initializes the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Variables and constraints are initialized as a function of the
 *      widget_scale.
 * Much of the work is accomplished in the BaseWidget constructor which
 *      includes some consistency checking to ensure full initialization.
 */
ArrowWidget::ArrowWidget( Module* module, CrowdMonitor* lock, double widget_scale )
: BaseWidget(module, lock, "ArrowWidget", NumVars, NumCons, NumGeoms, NumPcks, NumMatls, NumMdes, NumSwtchs, widget_scale),
  direction(0, 0, 1)
{
   variables[PointVar] = scinew PointVariable("Point", solve, Scheme1, Point(0, 0, 0));

   GeomGroup* arr = scinew GeomGroup;
   geometries[GeomPoint] = scinew GeomSphere;
   materials[PointMatl] = scinew GeomMaterial(geometries[GeomPoint], DefaultPointMaterial);
   arr->add(materials[PointMatl]);
   geometries[GeomShaft] = scinew GeomCylinder;
   materials[ShaftMatl] = scinew GeomMaterial(geometries[GeomShaft], DefaultEdgeMaterial);
   arr->add(materials[ShaftMatl]);
   geometries[GeomHead] = scinew GeomCappedCone;
   materials[HeadMatl] = scinew GeomMaterial(geometries[GeomHead], DefaultEdgeMaterial);
   arr->add(materials[HeadMatl]);
   picks[Pick] = scinew GeomPick(arr, module, this, Pick);
   picks[Pick]->set_highlight(DefaultHighlightMaterial);
   CreateModeSwitch(0, picks[Pick]);

   SetMode(Mode0, Switch0);

   FinishWidget();
}


/***************************************************************************
 * The destructor frees the widget's allocated structures.
 * The BaseWidget's destructor frees the widget's constraints, variables,
 *      geometry, picks, materials, modes, switches, and schemes.
 * Therefore, most widgets' destructors will not need to do anything.
 */
ArrowWidget::~ArrowWidget()
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
ArrowWidget::redraw()
{
   if (mode_switches[0]->get_state()) {
      Point center(variables[PointVar]->point());
      Vector direct(direction*widget_scale);
      ((GeomSphere*)geometries[GeomPoint])->move(center, widget_scale);

      if (direct.length2() > 0) {
	 ((GeomCylinder*)geometries[GeomShaft])->move(center,
						      center + direct * 3.0,
						      0.5*widget_scale);
	 ((GeomCappedCone*)geometries[GeomHead])->move(center + direct * 3.0,
						       center + direct * 5.0,
						       widget_scale,
						       0);
      }
   }

   Vector v1, v2;
   direction.find_orthogonal(v1, v2);
   for (Index geom = 0; geom < NumPcks; geom++) {
      picks[geom]->set_principal(direction, v1, v2);
   }
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
ArrowWidget::geom_moved( GeomPick*, int /* axis */, double /* dist */,
			 const Vector& delta, int pick, const BState& )
{
    switch(pick){
    case Pick:
	MoveDelta(delta);
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
ArrowWidget::MoveDelta( const Vector& delta )
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
ArrowWidget::ReferencePoint() const
{
   return variables[PointVar]->point();
}


void
ArrowWidget::SetPosition( const Point& p )
{
   variables[PointVar]->Move(p);

   execute(0);
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

   execute(0);
}


const Vector&
ArrowWidget::GetDirection() const
{
   return direction;
}


/***************************************************************************
 * This standard method returns a string describing the functionality of
 *      a widget's material property.  The string is used in the 
 *      BaseWidget UI.
 */
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


void
ArrowWidget::widget_tcl( TCLArgs& args )
{
   if (args[1] == "translate"){
      if (args.count() != 4) {
	 args.error("arrow widget needs axis translation");
	 return;
      }
      Real trans;
      if (!args[3].get_double(trans)) {
	 args.error("arrow widget can't parse translation `"+args[3]+"'");
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
	 args.error("arrow widget unknown axis `"+args[2]+"'");
	 break;
      }
      SetPosition(p);
   }
}

