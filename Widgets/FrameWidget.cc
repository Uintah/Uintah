
/*
 *  FrameWidget.h
 *
 *  Written by:
 *   James Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Jan. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */


#include <Widgets/FrameWidget.h>
#include <Constraints/DistanceConstraint.h>
#include <Constraints/PythagorasConstraint.h>
#include <Constraints/PlaneConstraint.h>
#include <Geom/Cylinder.h>
#include <Geom/Sphere.h>

const Index NumCons = 8;
const Index NumVars = 7;
const Index NumGeoms = 12;
const Index NumMatls = 3;
const Index NumPcks = 9;
const Index NumSchemes = 6;

enum { FrameW_ConstULDR, FrameW_ConstURDL, FrameW_ConstPyth, FrameW_ConstPlane,
       FrameW_ConstULUR, FrameW_ConstULDL, FrameW_ConstDRUR, FrameW_ConstDRDL };
enum { FrameW_GeomPointUL, FrameW_GeomPointUR, FrameW_GeomPointDR, FrameW_GeomPointDL,
       FrameW_GeomCylU, FrameW_GeomCylR, FrameW_GeomCylD, FrameW_GeomCylL,
       FrameW_GeomResizeU, FrameW_GeomResizeR, FrameW_GeomResizeD, FrameW_GeomResizeL };
enum { FrameW_PickSphUL, FrameW_PickSphUR, FrameW_PickSphDR, FrameW_PickSphDL, FrameW_PickCyls,
       FrameW_PickResizeU, FrameW_PickResizeR, FrameW_PickResizeD, FrameW_PickResizeL };

FrameWidget::FrameWidget( Module* module, CrowdMonitor* lock, Real widget_scale )
: BaseWidget(module, lock, NumVars, NumCons, NumGeoms, NumMatls, NumPcks, widget_scale*0.1),
  oldaxis1(1, 0, 0), oldaxis2(0, 1, 0)
{
   Real INIT = 1.0*widget_scale;
   // Schemes 5/6 are used by the picks in GeomMoved!!
   variables[FrameW_PointUL] = new PointVariable("PntUL", Scheme1, Point(0, 0, 0));
   variables[FrameW_PointUR] = new PointVariable("PntUR", Scheme2, Point(INIT, 0, 0));
   variables[FrameW_PointDR] = new PointVariable("PntDR", Scheme3, Point(INIT, INIT, 0));
   variables[FrameW_PointDL] = new PointVariable("PntDL", Scheme4, Point(0, INIT, 0));
   variables[FrameW_Dist1] = new RealVariable("DIST1", Scheme6, INIT);
   variables[FrameW_Dist2] = new RealVariable("DIST2", Scheme5, INIT);
   variables[FrameW_Hypo] = new RealVariable("HYPO", Scheme5, sqrt(2*INIT*INIT));

   constraints[FrameW_ConstPlane] = new PlaneConstraint("ConstPlane",
							NumSchemes,
							variables[FrameW_PointUL],
							variables[FrameW_PointUR],
							variables[FrameW_PointDR],
							variables[FrameW_PointDL]);
   constraints[FrameW_ConstPlane]->VarChoices(Scheme1, 2, 3, 2, 1);
   constraints[FrameW_ConstPlane]->VarChoices(Scheme2, 2, 3, 0, 3);
   constraints[FrameW_ConstPlane]->VarChoices(Scheme3, 0, 3, 0, 1);
   constraints[FrameW_ConstPlane]->VarChoices(Scheme4, 2, 1, 0, 1);
   constraints[FrameW_ConstPlane]->VarChoices(Scheme5, 2, 3, 0, 1);
   constraints[FrameW_ConstPlane]->VarChoices(Scheme6, 2, 3, 0, 1);
   constraints[FrameW_ConstPlane]->Priorities(P_Highest, P_Highest,
					      P_Highest, P_Highest);
   constraints[FrameW_ConstULDR] = new DistanceConstraint("Const13",
							  NumSchemes,
							  variables[FrameW_PointUL],
							  variables[FrameW_PointDR],
							  variables[FrameW_Hypo]);
   constraints[FrameW_ConstULDR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[FrameW_ConstULDR]->VarChoices(Scheme2, 1, 0, 1);
   constraints[FrameW_ConstULDR]->VarChoices(Scheme3, 0, 0, 0);
   constraints[FrameW_ConstULDR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[FrameW_ConstULDR]->VarChoices(Scheme5, 2, 2, 1);
   constraints[FrameW_ConstULDR]->VarChoices(Scheme6, 1, 0, 1);
   constraints[FrameW_ConstULDR]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[FrameW_ConstURDL] = new DistanceConstraint("Const24",
							  NumSchemes,
							  variables[FrameW_PointUR],
							  variables[FrameW_PointDL],
							  variables[FrameW_Hypo]);
   constraints[FrameW_ConstURDL]->VarChoices(Scheme1, 1, 0, 1);
   constraints[FrameW_ConstURDL]->VarChoices(Scheme2, 1, 1, 1);
   constraints[FrameW_ConstURDL]->VarChoices(Scheme3, 1, 0, 1);
   constraints[FrameW_ConstURDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[FrameW_ConstURDL]->VarChoices(Scheme5, 1, 0, 1);
   constraints[FrameW_ConstURDL]->VarChoices(Scheme6, 2, 2, 1);
   constraints[FrameW_ConstURDL]->Priorities(P_HighMedium, P_HighMedium, P_Default);
   constraints[FrameW_ConstPyth] = new PythagorasConstraint("ConstPyth",
							    NumSchemes,
							    variables[FrameW_Dist1],
							    variables[FrameW_Dist2],
							    variables[FrameW_Hypo]);
   constraints[FrameW_ConstPyth]->VarChoices(Scheme1, 1, 0, 1);
   constraints[FrameW_ConstPyth]->VarChoices(Scheme2, 1, 0, 1);
   constraints[FrameW_ConstPyth]->VarChoices(Scheme3, 1, 0, 1);
   constraints[FrameW_ConstPyth]->VarChoices(Scheme4, 1, 0, 1);
   constraints[FrameW_ConstPyth]->VarChoices(Scheme5, 2, 2, 0);
   constraints[FrameW_ConstPyth]->VarChoices(Scheme6, 2, 2, 1);
   constraints[FrameW_ConstPyth]->Priorities(P_Default, P_Default, P_HighMedium);
   constraints[FrameW_ConstULUR] = new DistanceConstraint("Const12",
							  NumSchemes,
							  variables[FrameW_PointUL],
							  variables[FrameW_PointUR],
							  variables[FrameW_Dist1]);
   constraints[FrameW_ConstULUR]->VarChoices(Scheme1, 1, 1, 1);
   constraints[FrameW_ConstULUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[FrameW_ConstULUR]->VarChoices(Scheme3, 1, 0, 0);
   constraints[FrameW_ConstULUR]->VarChoices(Scheme4, 1, 0, 1);
   constraints[FrameW_ConstULUR]->VarChoices(Scheme5, 1, 1, 1);
   constraints[FrameW_ConstULUR]->VarChoices(Scheme6, 0, 0, 0);
   constraints[FrameW_ConstULUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[FrameW_ConstULDL] = new DistanceConstraint("Const14",
							  NumSchemes,
							  variables[FrameW_PointUL],
							  variables[FrameW_PointDL],
							  variables[FrameW_Dist2]);
   constraints[FrameW_ConstULDL]->VarChoices(Scheme1, 1, 1, 1);
   constraints[FrameW_ConstULDL]->VarChoices(Scheme2, 1, 0, 0);
   constraints[FrameW_ConstULDL]->VarChoices(Scheme3, 1, 0, 1);
   constraints[FrameW_ConstULDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[FrameW_ConstULDL]->VarChoices(Scheme5, 1, 1, 1);
   constraints[FrameW_ConstULDL]->VarChoices(Scheme6, 0, 0, 0);
   constraints[FrameW_ConstULDL]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[FrameW_ConstDRUR] = new DistanceConstraint("Const32",
							  NumSchemes,
							  variables[FrameW_PointDR],
							  variables[FrameW_PointUR],
							  variables[FrameW_Dist2]);
   constraints[FrameW_ConstDRUR]->VarChoices(Scheme1, 1, 0, 1);
   constraints[FrameW_ConstDRUR]->VarChoices(Scheme2, 0, 0, 0);
   constraints[FrameW_ConstDRUR]->VarChoices(Scheme3, 1, 1, 1);
   constraints[FrameW_ConstDRUR]->VarChoices(Scheme4, 1, 0, 0);
   constraints[FrameW_ConstDRUR]->VarChoices(Scheme5, 1, 1, 1);
   constraints[FrameW_ConstDRUR]->VarChoices(Scheme6, 0, 0, 0);
   constraints[FrameW_ConstDRUR]->Priorities(P_Default, P_Default, P_LowMedium);
   constraints[FrameW_ConstDRDL] = new DistanceConstraint("Const34",
							  NumSchemes,
							  variables[FrameW_PointDR],
							  variables[FrameW_PointDL],
							  variables[FrameW_Dist1]);
   constraints[FrameW_ConstDRDL]->VarChoices(Scheme1, 1, 0, 1);
   constraints[FrameW_ConstDRDL]->VarChoices(Scheme2, 1, 0, 0);
   constraints[FrameW_ConstDRDL]->VarChoices(Scheme3, 1, 1, 1);
   constraints[FrameW_ConstDRDL]->VarChoices(Scheme4, 0, 0, 0);
   constraints[FrameW_ConstDRDL]->VarChoices(Scheme5, 1, 1, 1);
   constraints[FrameW_ConstDRDL]->VarChoices(Scheme6, 0, 0, 0);
   constraints[FrameW_ConstDRDL]->Priorities(P_Default, P_Default, P_LowMedium);

   materials[FrameW_PointMatl] = PointWidgetMaterial;
   materials[FrameW_EdgeMatl] = EdgeWidgetMaterial;
   materials[FrameW_HighMatl] = HighlightWidgetMaterial;

   Index geom, pick;
   GeomGroup* pts = new GeomGroup;
   for (geom = FrameW_GeomPointUL, pick = FrameW_PickSphUL;
	geom <= FrameW_GeomPointDL; geom++, pick++) {
      geometries[geom] = new GeomSphere;
      picks[pick] = new GeomPick(geometries[geom], module);
      picks[pick]->set_highlight(materials[FrameW_HighMatl]);
      picks[pick]->set_cbdata((void*)pick);
      pts->add(picks[pick]);
   }
   GeomMaterial* ptsm = new GeomMaterial(pts, materials[FrameW_PointMatl]);
   
   GeomGroup* resizes = new GeomGroup;
   for (geom = FrameW_GeomResizeU, pick = FrameW_PickResizeU;
	geom <= FrameW_GeomResizeL; geom++, pick++) {
      geometries[geom] = new GeomCappedCylinder;
      picks[pick] = new GeomPick(geometries[geom], module);
      picks[pick]->set_highlight(materials[FrameW_HighMatl]);
      picks[pick]->set_cbdata((void*)pick);
      resizes->add(picks[pick]);
   }
   GeomMaterial* resizem = new GeomMaterial(resizes, materials[FrameW_PointMatl]);

   GeomGroup* cyls = new GeomGroup;
   for (geom = FrameW_GeomCylU; geom <= FrameW_GeomCylL; geom++) {
      geometries[geom] = new GeomCylinder;
      cyls->add(geometries[geom]);
   }
   picks[FrameW_PickCyls] = new GeomPick(cyls, module);
   picks[FrameW_PickCyls]->set_highlight(materials[FrameW_HighMatl]);
   picks[FrameW_PickCyls]->set_cbdata((void*)FrameW_PickCyls);
   GeomMaterial* cylsm = new GeomMaterial(picks[FrameW_PickCyls], materials[FrameW_EdgeMatl]);

   GeomGroup* w = new GeomGroup;
   w->add(ptsm);
   w->add(resizem);
   w->add(cylsm);

   SetEpsilon(widget_scale*1e-6);
   
   FinishWidget(w);
}


FrameWidget::~FrameWidget()
{
}


void
FrameWidget::widget_execute()
{
   ((GeomSphere*)geometries[FrameW_GeomPointUL])->move(variables[FrameW_PointUL]->GetPoint(),
						       1*widget_scale);
   ((GeomSphere*)geometries[FrameW_GeomPointUR])->move(variables[FrameW_PointUR]->GetPoint(),
						       1*widget_scale);
   ((GeomSphere*)geometries[FrameW_GeomPointDR])->move(variables[FrameW_PointDR]->GetPoint(),
						       1*widget_scale);
   ((GeomSphere*)geometries[FrameW_GeomPointDL])->move(variables[FrameW_PointDL]->GetPoint(),
						       1*widget_scale);
   Point p(variables[FrameW_PointUL]->GetPoint() + (variables[FrameW_PointUR]->GetPoint()
						    - variables[FrameW_PointUL]->GetPoint()) / 2.0);
   ((GeomCappedCylinder*)geometries[FrameW_GeomResizeU])->move(p - (GetAxis2() * 0.6 * widget_scale),
							       p + (GetAxis2() * 0.6 * widget_scale),
							       0.75*widget_scale);
   p = variables[FrameW_PointUR]->GetPoint() + (variables[FrameW_PointDR]->GetPoint()
						- variables[FrameW_PointUR]->GetPoint()) / 2.0;
   ((GeomCappedCylinder*)geometries[FrameW_GeomResizeR])->move(p - (GetAxis1() * 0.6 * widget_scale),
							       p + (GetAxis1() * 0.6 * widget_scale),
							       0.75*widget_scale);
   p = variables[FrameW_PointDR]->GetPoint() + (variables[FrameW_PointDL]->GetPoint()
						- variables[FrameW_PointDR]->GetPoint()) / 2.0;
   ((GeomCappedCylinder*)geometries[FrameW_GeomResizeD])->move(p - (GetAxis2() * 0.6 * widget_scale),
							       p + (GetAxis2() * 0.6 * widget_scale),
							       0.75*widget_scale);
   p = variables[FrameW_PointDL]->GetPoint() + (variables[FrameW_PointUL]->GetPoint()
						- variables[FrameW_PointDL]->GetPoint()) / 2.0;
   ((GeomCappedCylinder*)geometries[FrameW_GeomResizeL])->move(p - (GetAxis1() * 0.6 * widget_scale),
							       p + (GetAxis1() * 0.6 * widget_scale),
							       0.75*widget_scale);
   ((GeomCylinder*)geometries[FrameW_GeomCylU])->move(variables[FrameW_PointUL]->GetPoint(),
						      variables[FrameW_PointUR]->GetPoint(),
						      0.5*widget_scale);
   ((GeomCylinder*)geometries[FrameW_GeomCylR])->move(variables[FrameW_PointUR]->GetPoint(),
						      variables[FrameW_PointDR]->GetPoint(),
						      0.5*widget_scale);
   ((GeomCylinder*)geometries[FrameW_GeomCylD])->move(variables[FrameW_PointDR]->GetPoint(),
						      variables[FrameW_PointDL]->GetPoint(),
						      0.5*widget_scale);
   ((GeomCylinder*)geometries[FrameW_GeomCylL])->move(variables[FrameW_PointDL]->GetPoint(),
						      variables[FrameW_PointUL]->GetPoint(),
						      0.5*widget_scale);

   ((DistanceConstraint*)constraints[FrameW_ConstULUR])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[FrameW_ConstDRDL])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[FrameW_ConstULDL])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[FrameW_ConstDRUR])->SetMinimum(3.2*widget_scale);
   ((DistanceConstraint*)constraints[FrameW_ConstULDR])->SetMinimum(sqrt(2*3.2*3.2)*widget_scale);
   ((DistanceConstraint*)constraints[FrameW_ConstURDL])->SetMinimum(sqrt(2*3.2*3.2)*widget_scale);

   SetEpsilon(widget_scale*1e-6);

   Vector spvec1(variables[FrameW_PointUR]->GetPoint() - variables[FrameW_PointUL]->GetPoint());
   Vector spvec2(variables[FrameW_PointDL]->GetPoint() - variables[FrameW_PointUL]->GetPoint());
   if ((spvec1.length2() > 0.0) && (spvec2.length2() > 0.0)) {
      spvec1.normalize();
      spvec2.normalize();
      Vector v = Cross(spvec1, spvec2);
      for (Index geom = 0; geom < NumPcks; geom++) {
	 if ((geom == FrameW_PickResizeU) || (geom == FrameW_PickResizeD))
	    picks[geom]->set_principal(spvec2);
	 else if ((geom == FrameW_PickResizeL) || (geom == FrameW_PickResizeR))
	    picks[geom]->set_principal(spvec1);
	 else
	    picks[geom]->set_principal(spvec1, spvec2, v);
      }
   }
}

void
FrameWidget::geom_moved( int /* axis */, double /* dist */, const Vector& delta,
			 void* cbdata )
{
   Vector delt(delta);
   ((DistanceConstraint*)constraints[FrameW_ConstULUR])->SetDefault(GetAxis1());
   ((DistanceConstraint*)constraints[FrameW_ConstDRDL])->SetDefault(GetAxis1());
   ((DistanceConstraint*)constraints[FrameW_ConstULDL])->SetDefault(GetAxis2());
   ((DistanceConstraint*)constraints[FrameW_ConstDRUR])->SetDefault(GetAxis2());
   
   for (Index v=0; v<NumVars; v++)
      variables[v]->Reset();
   
   switch((int)cbdata){
   case FrameW_PickSphUL:
      variables[FrameW_PointUL]->SetDelta(delta);
      break;
   case FrameW_PickSphUR:
      variables[FrameW_PointUR]->SetDelta(delta);
      break;
   case FrameW_PickSphDR:
      variables[FrameW_PointDR]->SetDelta(delta);
      break;
   case FrameW_PickSphDL:
      variables[FrameW_PointDL]->SetDelta(delta);
      break;
   case FrameW_PickResizeU:
      if (((variables[FrameW_PointUL]->GetPoint()+delta)-variables[FrameW_PointDL]->GetPoint()).length()
	  < 3.2*widget_scale) {
	 delt = ((variables[FrameW_PointDL]->GetPoint() + delta.normal()*3.2*widget_scale)
		 - variables[FrameW_PointUL]->GetPoint());
      }
      variables[FrameW_PointUL]->MoveDelta(delt);
      variables[FrameW_PointUR]->SetDelta(delt, Scheme6);
      break;
   case FrameW_PickResizeR:
      if (((variables[FrameW_PointUR]->GetPoint()+delta)-variables[FrameW_PointUL]->GetPoint()).length()
	  < 3.2*widget_scale) {
	 delt = ((variables[FrameW_PointUL]->GetPoint() + delta.normal()*3.2*widget_scale)
		 - variables[FrameW_PointUR]->GetPoint());
      }
      variables[FrameW_PointUR]->MoveDelta(delt);
      variables[FrameW_PointDR]->SetDelta(delt, Scheme5);
      break;
   case FrameW_PickResizeD:
      if (((variables[FrameW_PointDR]->GetPoint()+delta)-variables[FrameW_PointUR]->GetPoint()).length()
	  < 3.2*widget_scale) {
	 delt = ((variables[FrameW_PointUR]->GetPoint() + delta.normal()*3.2*widget_scale)
		 - variables[FrameW_PointDR]->GetPoint());
      }
      variables[FrameW_PointDR]->MoveDelta(delt);
      variables[FrameW_PointDL]->SetDelta(delt, Scheme6);
      break;
   case FrameW_PickResizeL:
      if (((variables[FrameW_PointDL]->GetPoint()+delta)-variables[FrameW_PointDR]->GetPoint()).length()
	  < 3.2*widget_scale) {
	 delt = ((variables[FrameW_PointDR]->GetPoint() + delta.normal()*3.2*widget_scale)
		 - variables[FrameW_PointDL]->GetPoint());
      }
      variables[FrameW_PointDL]->MoveDelta(delt);
      variables[FrameW_PointUL]->SetDelta(delt, Scheme5);
      break;
   case FrameW_PickCyls:
      variables[FrameW_PointUL]->MoveDelta(delta);
      variables[FrameW_PointUR]->MoveDelta(delta);
      variables[FrameW_PointDR]->MoveDelta(delta);
      variables[FrameW_PointDL]->MoveDelta(delta);
      break;
   }
}


void
FrameWidget::SetPosition( const Point& UL, const Point& UR, const Point& DL )
{
   variables[FrameW_PointUL]->Move(UL);
   variables[FrameW_PointUR]->Move(UR);
   variables[FrameW_PointDL]->Move(DL);
   variables[FrameW_Dist1]->Move((UR-UL).length());
   variables[FrameW_Dist2]->Move((DL-UL).length());
   variables[FrameW_PointDR]->Set(UR+(DL-UL), Scheme5); // This should set Hypo...

   execute();
}


void
FrameWidget::GetPosition( Point& UL, Point& UR, Point& DL )
{
   UL = variables[FrameW_PointUL]->GetPoint();
   UR = variables[FrameW_PointUR]->GetPoint();
   DL = variables[FrameW_PointDL]->GetPoint();
}


void
FrameWidget::SetPosition( const Point& center, const Vector& normal,
			  const Real size1, const Real size2 )
{
   Real s1(size1/2.0), s2(size2/2.0);
   Vector axis1, axis2;
   normal.find_orthogonal(axis1, axis2);
   
   variables[FrameW_PointUL]->Move(center-axis1*s1-axis2*s2);
   variables[FrameW_PointDR]->Move(center+axis1*s1+axis2*s2);
   variables[FrameW_PointUR]->Move(center+axis1*s1-axis2*s2);
   variables[FrameW_PointDL]->Move(center-axis1*s1+axis2*s2);
   variables[FrameW_Dist1]->Move(size1);
   variables[FrameW_Dist2]->Set(size2); // This should set the Hypo...

   execute();
}


void
FrameWidget::GetPosition( Point& center, Vector& normal,
			  Real& size1, Real& size2 )
{
   center = (variables[FrameW_PointDR]->GetPoint()
	     + ((variables[FrameW_PointUL]->GetPoint()-variables[FrameW_PointDR]->GetPoint())
		/ 2.0));
   normal = Cross(GetAxis1(), GetAxis2());
   size1 = variables[FrameW_Dist1]->GetReal();
   size2 = variables[FrameW_Dist2]->GetReal();
}


void
FrameWidget::SetSize( const Real size1, const Real size2 )
{
   ASSERT((size1>=0.0)&&(size1>=0.0));

   Point center(variables[FrameW_PointDR]->GetPoint()
		+ ((variables[FrameW_PointUL]->GetPoint()-variables[FrameW_PointDR]->GetPoint())
		   / 2.0));
   Vector axis1((variables[FrameW_PointUR]->GetPoint() - variables[FrameW_PointUL]->GetPoint())/2.0);
   Vector axis2((variables[FrameW_PointDL]->GetPoint() - variables[FrameW_PointUL]->GetPoint())/2.0);
   Real ratio1(size1/variables[FrameW_Dist1]->GetReal());
   Real ratio2(size2/variables[FrameW_Dist2]->GetReal());

   variables[FrameW_PointUL]->Move(center-axis1*ratio1-axis2*ratio2);
   variables[FrameW_PointDR]->Move(center+axis1*ratio1+axis2*ratio2);
   variables[FrameW_PointUR]->Move(center+axis1*ratio1-axis2*ratio2);
   variables[FrameW_PointDL]->Move(center-axis1*ratio1+axis2*ratio2);

   variables[FrameW_Dist1]->Move(size1);
   variables[FrameW_Dist2]->Set(size2); // This should set the Hypo...

   execute();
}

void
FrameWidget::GetSize( Real& size1, Real& size2 ) const
{
   size1 = variables[FrameW_Dist1]->GetReal();
   size2 = variables[FrameW_Dist2]->GetReal();
}

   
Vector
FrameWidget::GetAxis1()
{
   Vector axis(variables[FrameW_PointUR]->GetPoint() - variables[FrameW_PointUL]->GetPoint());
   if (axis.length2() <= 1e-6)
      return oldaxis1;
   else
      return (oldaxis1 = axis.normal());
}


Vector
FrameWidget::GetAxis2()
{
   Vector axis(variables[FrameW_PointDL]->GetPoint() - variables[FrameW_PointUL]->GetPoint());
   if (axis.length2() <= 1e-6)
      return oldaxis2;
   else
      return (oldaxis2 = axis.normal());
}


