
/*
 *  DPNavierStokes.cc: Solve Navier Stokes systems with Diffpack
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   Oct. 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

extern void AuditDefaultAllocator();

#define NUMT double
#define ARRAY_RANGECHECK
#define HANDLE_0PTR_CHECK
#define F77HANDLING 1

inline double abs(double x)
{
    return x<0?-x:x;
}


#include <FEM.h>                // finite element toolbox
#include <DegFreeFE.h>          // field dof <-> linear system dof
#include <TimePrm.h>            // time discretization parameters
#include <Store4Plotting.h>     // storage of data for later visualization
#include <MenuUDC.h>
#include <LinEqAdm.h>           // linear solver toolbox

/* outline of classes:
  NavierStokes     - main simulator
  ElmMatVecPenalty - computation of elemental matrix/vector for penalty term
  ReducedIntg      - the integrand for reduced integration (penalty term)
  NonReducedIntg   - the integrand for "normal" integration
  PressureIntg     - integrand (div v) for pressure recovery and smoothing

  For documentation see the report 'The Diffpack Implementation of
  a Navier-Stokes Solver Based on the Penalty Function Method'.
*/

class DPNavierStokes;

/*<NonReducedIntg:*/
class NonReducedIntg : public IntegrandCalc
{
  DPNavierStokes& data;
public:
  NonReducedIntg (DPNavierStokes& data_) : data(data_) {}
 ~NonReducedIntg () {}
  virtual void integrands (ElmMatVec& elmat, FiniteElement& fe);
};
/*>NonReducedIntg:*/

/*<ReducedIntg:*/
class ReducedIntg : public IntegrandCalc
{
  DPNavierStokes& data;
public:
  ReducedIntg (DPNavierStokes& data_) : data(data_) {}
 ~ReducedIntg () {}
  virtual void integrands (ElmMatVec& elmat, FiniteElement& fe);
};
/*>ReducedIntg:*/


/*<ElmMatVecPenalty:*/
class ElmMatVecPenalty : public ElmMatVecCalcStd
{
  DPNavierStokes& data;
  NonReducedIntg v_terms;
  ReducedIntg lambda_term;
public:
  ElmMatVecPenalty (DPNavierStokes& data_)
    : ElmMatVecCalcStd (GAUSS_POINTS, 0),
      data(data_), v_terms(data_), lambda_term(data_) {}
 ~ElmMatVecPenalty () {}

  virtual void calcElmMatVec
    (
     int            elm_no,
     ElmMatVec&     elmat,
     FiniteElement& fe,
     FEM&           fem
     // IntegrandCalc objects are made inside this function body
    );

private:
  // this one is included to avoid warning (hides virtual...)
  virtual void calcElmMatVec
    (
     int            /*elm_no*/,
     ElmMatVec&     /*elmat*/,
     FiniteElement& /*fe*/,
     IntegrandCalc& /*integrand*/,
     FEM&           /*fem*/
    ){}
};
/*>ElmMatVecPenalty:*/

class PressureIntg;

/*<DPNavierStokes:*/
class DPNavierStokes : public FEM, public MenuUDC,
                     public Store4Plotting
{
  friend class NonReducedIntg;
  friend class    ReducedIntg;
  friend class ElmMatVecPenalty;
  friend class PressureIntg;

protected:
    friend class DPNavierStokesM;
    friend class DPMakeSystemM;
  Handle(GridFE)         grid;            // finite element mesh
  Handle(DegFreeFE)      dof;             // matrix dof <-> u dof
  int lineq_vec_length;
  Handle(FieldsFE)       u;               // velocity field
  Handle(FieldsFE)       u_prev;          // velocity field
  Handle(FieldFE)        p;               // pressure field (derived quantity)

  TimePrm                tip;             // time discretization parameters

  Vec(real)              *nonlin_solution; // nonlinear solution
  Vec(real)              linear_solution; // solution of linear subsystem

  // problem dependent data
  real theta;               // "theta"-rule time discretization parameter
  real lambda;              // penalty parameter
  real mu;                  // dynamic viscosity
  real density;

  int  inlet_profile;       // indicator for type of inflow profile
  real inlet_velocity;      // scaling of inflow velocity

  ElmMatVecPenalty elmv;    // object for administering the penalty term
  String users_comment;

public:
  DPNavierStokes ();//const clString& id);
 ~DPNavierStokes ();

  void define (MenuSystem& menu, int level = MAIN);
  void scan   (MenuSystem& menu);
  void adm    (MenuSystem& menu);
  void driver ();
  BooLean ok () const;

  void fillEssBC ();

protected:
  void inletVelocity (Ptv(real)& v, const Ptv(real)& x);
  void setIC ();
  void timeLoop ();
  void solveAtThisTimeLevel ();
  void calcDerivedQuantities ();
  void saveResults ();
  void updateDataStructures ();
  void compare2analyticalSolution ();

#if 0
  /* SCIRUN STUFF */
    virtual void execute();
    virtual Module* clone(int deep);
#endif
};
/*>DPNavierStokes:*/

/*Class:DPNavierStokes

NAME:              
DPNavierStokes - simulator for incompressible fluid flow (penalty method)

SYNTAX:	           @DPNavierStokes

KEYWORDS:          

fluid flow, incompressibility, penalty function method, Navier-Stokes equations


DESCRIPTION:       

 The class implements a numerical solution method for the incompressible
 Navier-Stokes equations using a penalty function approach where the pressure
 and the continuity equation are in some sense eliminated from the original
 equations. A finite element method is used for the spatial discretization
 together with a finite difference method in time.

 More information in this simulator is given in the report
 `{\em` The Diffpack Implementation of
 a Navier-Stokes Solver Based on the Penalty Function Method.`}`

NOTE:

 With the penalty function method it is not necessary to use mixed finite
 element formulations. This simplifies the implementation in Diffpack.
 If the "DPNavierStokes" class is extended to handle mixed finite elements,
 one should notice that the present version of the class orders the
 elements according to the so called special numbering, whereas a mixed
 method will require the general numbering convention when defining the
 equations in the weak form (cf. class "DegFreeFE").

SEE ALSO:

 class "Elasticity"


DEVELOPED BY:   

                SINTEF Applied Mathematics, Oslo, Norway, and
                University of Oslo, Dept. of Math., Norway

AUTHOR:	        

                Hans Petter Langtangen, SINTEF/UiO

End:
*/

// for pressure calculation (post-processing):
/*<PressureIntg:*/
class PressureIntg : public IntegrandCalc
{
  DPNavierStokes& data;
public:
  PressureIntg (DPNavierStokes& data_) : data(data_) {}
 ~PressureIntg () {}
  virtual void integrands (ElmMatVec& elmat, FiniteElement& fe);
};
/*>PressureIntg:*/


                                                                 // -*- C++ -*-
/*****************************************************************************/
/*                                                                           */
/* (c) Copyright 1994, 95, 96                                                */
/*     SINTEF, Oslo, Norway.                                                 */
/*     All rights reserved.                                                  */
/*                                                                           */
/*     See the file Copyright.h for further details.                         */
/*                                                                           */
/*****************************************************************************/

//#include<Copyright.h>
// $Id$

#include <ElmMatVec.h>
#include <FiniteElement.h>
#include <readOrMakeGrid.h>
#include <DegFreeFE.h>
#include <VecSimple_Ptv_real.h>
#include <MatDiag_real.h>

DPNavierStokes:: DPNavierStokes ()//const clString& id)
: //Module("DPNavierStokes", id, Filter),
  FEM (), /*lineq (EXTERNAL_SOLUTION), */ elmv (*this)
{ lambda = 1.0e+4; mu = 1.0e-3; density = 1000; theta = 0; }

DPNavierStokes:: ~DPNavierStokes ()
{
#ifdef DP_DEBUG
  DBP("In ~DPNavierStokes");
#endif
}

void DPNavierStokes:: define (MenuSystem& menu, int level)
{
  menu.addItem (level,
                "gridfile",
                "gridfile",
                "filename or PREPROCESSOR=PreproSupElSet/geometry/partition",
                "PREPROCESSOR=PreproBox/d=2 [0,0.05]x[0,0.01]/\
                d=2 elm_tp=ElmTensorProd1 div=[2,2], grading=[1,1]",
                "S",
                'g','g');
  menu.addItem (level,
                "redefine boundary indicators",
                "redefineBoInds",
                "GridFE/Boundary::redefineBoInds syntax",
                "nb=2 names=D1 D2  1=(1 3) 2=(2 4)",
                "S",
                'b','b');
  menu.addItem (level,
                "time integration parameters",
                "TimePrm",
                "TimePrm::scan syntax",
                "dt=0",
                "S",
                't','t');
  menu.addItem (level,
                "viscosity",
                "mu",
                "dynamic visc. coefficient",
                "1.0e-3",
                "R[0:1.0e+20]1",
                'v','v');
  menu.addItem (level,
                "penalty parameter",
                "lambda",
                "",
                "1.0e+4",
                "R[0:1.0e+20]1",
                'l','l');
  menu.addItem (level,
                "density",
                "rho",
                "",
                "1.0e+3",
                "R[0:1.0e+20]1",
                'd','d');
  menu.addItem (level,
                "inlet profile",
                "inlet_profile",
                "",
                "1",
                "I[0:10]1",
                'i','i');
  menu.addItem (level,
                "characteristic inlet velocity",
                "inlet_velocity",
                "used in inlet profile expressions",
                "1.0e-4",
                "R[0:1.0e+10]1",
                'U','U');
  menu.addItem (level,
                "user's comment",
                "comment",
                "for result file headings",
                " ",
                "S",
                 'c','c');
  LinEqAdm::defineStatic (menu, level+1);
  Store4Plotting:: defineStatic (menu, level+1);
}


void DPNavierStokes:: scan (MenuSystem& menu)
{
    //String gridfile = menu.get ("gridfile");
    //grid.rebind (new GridFE());
    //readOrMakeGrid (grid(), gridfile);  // fill grid

  String redef = menu.get ("redefine boundary indicators");
  if (!redef.contains("NONE"))
    grid->redefineBoInds (redef);

cerr << "time...\n";
  tip.scan (menu.get ("time integration parameters"));
cerr << "viscosity...\n";
  mu = menu.get ("viscosity").getReal();
  lambda = menu.get ("penalty parameter").getReal();
  density = menu.get ("density").getReal();

cerr << "comment...\n";
  users_comment = "NS: " + menu.get ("user's comment");  // for resfile

  const int nsd = grid->getNoSpaceDim();
cerr << "Store4Plotting...\n";
  Store4Plotting:: scan (menu, nsd);
  if (nsd <= 0 || nsd > 3)
    fatalerrorFP("DPNavierStokes::scan",
                 "nsd=%d is illegal, wrong grid?",nsd);

cerr << "inlet...\n";
  inlet_profile = menu.get ("inlet profile").getInt();
  inlet_velocity = menu.get ("characteristic inlet velocity").getReal();

  //cerr << "lineq...\n";
  //lineq.scan (menu);

  // initialization:
  p.rebind (new FieldFE (grid(),"p"));
  u.rebind (new FieldsFE (grid(),"v"));
  u_prev.rebind (new FieldsFE (grid(),"v_prev"));
  dof.rebind (new DegFreeFE (grid(), nsd));
  lineq_vec_length = u->getNoValues(); // alt: dof->getTotalNoDof
  linear_solution.redim (lineq_vec_length);
  //lineq.attach (linear_solution);
cerr << "scan done\n";
}

void DPNavierStokes:: adm (MenuSystem& menu)
{
    MenuUDC::attach (menu);
    define (menu);
    menu.prompt();
    scan (menu);
}

BooLean DPNavierStokes:: ok () const
{
  BooLean b = dpTRUE;
  HANDLE_OK("DPNavierStokes::ok",grid,b);
  HANDLE_OK("DPNavierStokes::ok",p,b);
  HANDLE_OK("DPNavierStokes::ok",u,b);
  HANDLE_OK("DPNavierStokes::ok",u_prev,b);
  OBJECT_OK("DPNavierStokes::ok",dof,b);
  OBJECT_OK("DPNavierStokes::ok",linear_solution,b);

  if (theta < 0 || theta > 1) {
    errorFP("DPNavierStokes::ok","illegal value, theta=%g",theta);
    b = dpFALSE;
  }
  return b;
}

void DPNavierStokes:: driver ()
{
  if (!ok())
    errorFP("DPNavierStokes",
            "object is not properly initialized");
  s_o->setRealFormat ("%13.6e");
  cerr << "setIC...\n";
  setIC ();
  cerr << "timeLoop...\n";
  timeLoop ();
}


void DPNavierStokes:: setIC ()
{
//Ptv(real) pt (nsd);
  const int nno = grid->getNoNodes();
  const int nsd = grid->getNoSpaceDim();
  int i,k;
  for (i = 1; i <= nno; i++)  {
    for (k = 1; k <= nsd; k++)  {
//    grid->getCoor(pt,i);
      u_prev()(k).values()(i) = 0.0;
      u()(k).values()(i) = 0.0;
    }
  }
}


void DPNavierStokes:: timeLoop ()  // general, could be inherited
{
  tip.initTimeLoop();
  while (!tip.finished())
    {
      tip.increaseTime();

      cerr << "solveAthis...\n";
      solveAtThisTimeLevel ();
      //calcDerivedQuantities ();
      //saveResults ();

      //updateDataStructures ();
    }
}


void DPNavierStokes:: solveAtThisTimeLevel ()
{
    cerr << "fillEssBC\n";
  fillEssBC ();

  // initialize nonlinear solution with the solution at the previous
  // time step:
    nonlin_solution=new Vec(real);
    cerr << "vec_length=" << lineq_vec_length << endl;
    nonlin_solution->redim(lineq_vec_length);
  dof->field2vec (u_prev(), *nonlin_solution);
  // if there are new BC at this time step, update the present guess:
  dof->fillEssBC (*nonlin_solution);

#ifdef DP_DEBUG
nonlin_solution.print(s_o,"nonlin_solution before nonlinear iteration");
#endif

//  dof->vec2field (nonlin_solution, u());   // load solution into u

#ifdef DP_DEBUG
  for (int k = 1; k <= grid->getNoSpaceDim(); k++)
    u()(k).values().print(s_o,oform("u(%d) after nonlinear it.",k));
#endif
//  compare2analyticalSolution ();
}

#if 0
void DPNavierStokes:: makeAndSolveLinearSystem ()
{
    cerr << "makeAndSolve...\n";
  dof->vec2field (nonlin_solution, u());  // put most recent guess in u
                                         // (convenient in integrands)
  if (nlsolver->getCurrentState().method == NEWTON_RAPHSON)
    dof->fillEssBC2zero();
  else
    dof->unfillEssBC2zero();

  // make linear system
    cerr << "makeSystem\n";
  makeSystem (dof(), elmv, NULL /* elmv has the integrands */, lineq);

/* NOTE: this first solution to selective reduced integration was wrong:
  makeSystem initializes the system matrix to zero such that only the
  lambda-term would survive in the code below:
  // integrate velocity terms by standard integration:
  elmv.integrator.setRelativeOrder (0);
  makeSystem (dof(), elmv, v_terms, lineq);
  // integrate the penalty term by reduced integration:
  elmv.integrator.setRelativeOrder (-1);
  makeSystem (dof(), elmv, lambda_term, lineq);
  elmv.integrator.setRelativeOrder (0);            // ready for standard-itg.
*/
  // startvector for iterative solvers: not necessary, the penalty method
  // does not work well for iterative solvers...

  // solve linear system (solution is stored in linear_solution):
    cerr << "lineq.solve()\n";
  lineq.solve(); 

#ifdef DP_DEBUG
linear_solution.print(s_o,"linear_solution"); // debug output
#endif
    cerr << "done...\n";
}
#endif

void DPNavierStokes:: updateDataStructures ()
{
  u_prev() = u();  // FieldsFE::operator=(FieldsFE) is invoked
                   // recall that u_prev=u copies the handles only!
}

void DPNavierStokes:: inletVelocity (Ptv(real)& v, const Ptv(real)& x)
{
  if (inlet_profile == 1)
    {
      // parabolic profile at x=0, max value in y=0 and v=0 at y=0.01
      v = 0.0;
      v(1) = inlet_velocity * (1 - pow2(x(2)/0.01));
    }
  else if (inlet_profile == 2 || inlet_profile == 3)
    {
      // uniform plug flow (2) or Coutte (3):
      v = 0.0;
      v(1) = inlet_velocity;
    }
  else
    errorFP("DPNavierStokes::inletVelocity","inlet_velocity=%d not impl.",
            inlet_velocity);
}


void DPNavierStokes:: fillEssBC ()
{
  // convention:
  // bo-ind 1: inlet boundary with prescribed velocity field
  // bo-ind 2: outlet boundary with both normal derivatives equal to zero
  // bo-ind 3: u=0 and dv/dn=dw/n=0
  // bo-ind 4: v=0 and du/dn=dw/n=0
  // bo-ind 5: w=0 and du/dn=dv/n=0
  // rigid wall is a combination of 3,4,5

  dof->initEssBC ();
  const int nno = grid->getNoNodes();
  const int nsd = grid->getNoSpaceDim();
  Ptv(real) v (nsd);
  Ptv(real) x (nsd);
  int i,k;
  for (i = 1; i <= nno; i++) {
    if (grid->BoNode (i, 1)) {
      // call inlet velocity profile
      grid->getCoor (x, i);
      inletVelocity (v, x);
      for (k = 1; k <= nsd; k++)
        dof->fillEssBC (dof->fields2dof(i,k), v(k));
    }
    for (k = 1; k <= nsd; k++) {
      if (grid->BoNode (i, 2+k))
        dof->fillEssBC (dof->fields2dof(i,k), 0.0);
    }
  }
}


void ElmMatVecPenalty:: calcElmMatVec
  (
   int            elm_no,
   ElmMatVec&     elmat,
   FiniteElement& fe,
   FEM&           /*fem*/
   // IntegrandCalc objects are made inside this function body
  )
{
  // --- selective reduced integration: ---

  integrator->setRelativeOrder (0);          // ordinary integration
  fe.refill (elm_no, integrator());
  data.numItgOverElm (elmat, fe, v_terms);
  integrator->setRelativeOrder (-1);         // reduced integration
  fe.refill (elm_no, integrator());
  data.numItgOverElm (elmat, fe, lambda_term);
  integrator->setRelativeOrder (0);          // ready for other, normal integr.
}


void NonReducedIntg:: integrands (ElmMatVec& elmat, FiniteElement& fe)
{
  int i,j;    // shape function counters
  int k,r,s;  // 1,..,nsd counters
  int ig,jg;  // element dof, based on i,j,r,s

  real nabla2,h1,h2,h3;

  const int nsd = fe.getNoSpaceDim();
  const int nbf = fe.getNoBasisFunc();
  const real detJxW = fe.detJxW();

  // these should be private members of the class to increase the efficency!
  // VecSimple(..) implies allocation of dynamic memory!
  Ptv(real) u_pt (nsd);                       // U at present intg. point
  Ptv(real) up_pt (nsd);                      // U, previous time level
  VecSimple(Ptv(real)) Du_pt (nsd);           // grad U at present intg. point
  VecSimple(Ptv(real)) Dup_pt (nsd);          // grad U, previous time level

  real dt, theta, fact;
  if (data.tip.stationary())
    { dt = 1;  theta = 0;  fact = 0; }
  else
    { dt = data.tip.Delta ();  theta = data.theta;  fact = 1; }

  for (k = 1; k <= nsd; k++) {
    Du_pt(k).redim (nsd);  Dup_pt(k).redim (nsd);
    u_pt(k) = data.u()(k).valueFEM (fe);
    data.u()(k).derivativeFEM (Du_pt(k), fe);

    if (!data.tip.stationary()) {
      up_pt(k) = data.u_prev()(k).valueFEM (fe);
      data.u_prev()(k).derivativeFEM (Dup_pt(k), fe);
    }
  }
  real conlas=0, dijlas=0;         // contributions from last time level
  real dirchl,con,dij,cij,sum;

  if (1)//data.nlsolver->getCurrentState().method == NEWTON_RAPHSON)
    {
      for (i = 1; i <= nbf; i++) {
        for (j = 1; j <= nbf; j++) {
          nabla2 = 0;
          for (k = 1; k <= nsd; k++)
            nabla2 += fe.dN(i,k)*fe.dN(j,k);

          dirchl = data.mu*nabla2*dt;

          sum = 0;
          for (k = 1; k <= nsd; k++)
            sum += fe.dN(j,k)*u_pt(k);
          con = data.density*fe.N(i)*sum*dt;

          dij = dirchl + con;
          cij = data.density*fe.N(i)*fe.N(j);

          if (theta > 0) {
            sum = 0;
            for (k = 1; k <= nsd; k++)
              sum += fe.dN(j,k)*up_pt(k);
            conlas = data.density*fe.N(i)*sum*dt;
            dijlas = dirchl + conlas;
          }

          for (r = 1; r <= nsd; r++)
            for (s = 1; s <= nsd; s++) {
              ig = nsd*(i-1)+r;
              jg = nsd*(j-1)+s;

              h1 = cij*dt*Dup_pt(r)(s) * (1-theta);
              elmat.A(ig,jg) += h1*detJxW;

              if (r == s) {
                h2 = dij*(1-theta) + fact*cij;
                elmat.A(ig,jg) += h2*detJxW;

                // should be rewritten in a less primitive form!
                real sli = data.u()(r).values()(data.grid->
                           loc2glob(fe.getElmNo(),j));
                real slt = data.u_prev()(r).values()(data.grid->
                           loc2glob(fe.getElmNo(),j));
                h3 = fact*cij*(sli - slt)
                   + dij*sli*(1-theta) + dijlas*slt*theta;
                elmat.b(ig) += -h3*detJxW;
	      }
	    }
        }
      }
    }
  else if (0)//data.nlsolver->getCurrentState().method == SUCCESSIVE_SUBST)
    {
      warningFP("DPNavierStokes::integrands","SuccessiveSubst not impl.");
    }
#if 0
  else
    fatalerrorFP("DPNavierStokes::integrands",
                 "current nonlinear method=%d, illegal value",
                 data.nlsolver->getCurrentState().method);
#endif
}

void ReducedIntg:: integrands (ElmMatVec& elmat, FiniteElement& fe)
{
  int i,j;    // shape function counters
  int k,r,s;  // 1,..,nsd counters
  int ig,jg;  // element dof, based on i,j,r,s

  real h1,h2;

  const int nsd = fe.getNoSpaceDim();
  const int nbf = fe.getNoBasisFunc();
  const real detJxW = fe.detJxW();

  // these should be private members of the class to increase the efficency!
  // VecSimple(..) implies allocation of dynamic memory!
  Ptv(real) u_pt (nsd);                       // U at present intg. point
  Ptv(real) up_pt (nsd);                      // U, previous time level
  VecSimple(Ptv(real)) Du_pt (nsd);           // grad U at present intg. point
  VecSimple(Ptv(real)) Dup_pt (nsd);          // grad U, previous time level

  real dt, theta;
  if (data.tip.stationary())
    { dt = 1;  theta = 0; }
  else
    { dt = data.tip.Delta ();  theta = data.theta; }

  for (k = 1; k <= nsd; k++) {
    Du_pt(k).redim (nsd);   Dup_pt(k).redim (nsd);
    u_pt(k) = data.u()(k).valueFEM (fe);
    data.u()(k).derivativeFEM (Du_pt(k), fe);

    if (!data.tip.stationary()) {
      up_pt(k) = data.u_prev()(k).valueFEM (fe);
      data.u_prev()(k).derivativeFEM (Dup_pt(k), fe);
    }
  }
  real sum, div, divlas=0;

  if (1)//data.nlsolver->getCurrentState().method == NEWTON_RAPHSON)
    {
      for (i = 1; i <= nbf; i++)
        for (j = 1; j <= nbf; j++)
          for (r = 1; r <= nsd; r++)
            for (s = 1; s <= nsd; s++) {
              ig = nsd*(i-1)+r;
              jg = nsd*(j-1)+s;

              h1 = data.lambda*fe.dN(i,r)*fe.dN(j,s)*dt*(1-theta);
              elmat.A(ig,jg) += h1*detJxW;
            }
      sum = 0;
      for (k = 1; k <= nsd; k++)
        sum += Du_pt(k)(k);           // sum = div(u)

      div = data.lambda*sum*dt;
      if (theta > 0) {
        sum = 0;
        for (k = 1; k <= nsd; k++)
          sum += Dup_pt(k)(k);           // sum = div(u)
        divlas = data.lambda*sum*dt;
      }
      for (i = 1; i <= nbf; i++)
        for (r = 1; r <= nsd; r++) {
          ig = nsd*(i-1)+r;
          h2 = div*fe.dN(i,r)*(1-theta) - divlas*fe.dN(i,r)*theta;
          elmat.b(ig) += - h2*detJxW;
        }
    }
  else if (0)//data.nlsolver->getCurrentState().method == SUCCESSIVE_SUBST)
    {
      warningFP("DPNavierStokes::integrands, lambda term",
                "SuccessiveSubst not impl.");
    }
#if 0
  else
    fatalerrorFP("DPNavierStokes::integrands",
                 "current nonlinear method=%d, illegal value",
                 data.nlsolver->getCurrentState().method);
#endif
}


void PressureIntg:: integrands (ElmMatVec& elmat, FiniteElement& fe)
{
  const int nsd = fe.getNoSpaceDim();
  static VecSimple(Ptv(real)) Du_pt (nsd);
  real div = 0;
  for (int k = 1; k <= nsd; k++)
    {
      data.u()(k).derivativeFEM (Du_pt(k), fe);
      div += Du_pt(k)(k);
    }
  const real pressure = - data.lambda * div;

  const int nbf = fe.getNoBasisFunc();
  const real detJxW = fe.detJxW();

  for (int i = 1; i <= nbf; i++)
    elmat.b(i) += pressure*fe.N(i)*detJxW;
}


void DPNavierStokes:: calcDerivedQuantities ()
{
  MatDiag(real) mass_matrix (grid->getNoNodes());
  makeMassMatrix (grid(), mass_matrix, dpTRUE);        // class FEM function
  mass_matrix.factLU();

  PressureIntg penalty_integrand (*this);
  smoothField (p(), mass_matrix, penalty_integrand);   // class FEM function
}

void DPNavierStokes:: saveResults ()
{
  dump (p(), &tip, users_comment.chars());
  dump (u(), &tip, users_comment.chars());

  // debug output for small grids:
  const int nno = grid->getNoNodes();
  const int nsd = grid->getNoSpaceDim();
  if (nno < 30) {
    int i,r;
    for (i=1; i<=nno; i++) {
      s_o<<oform("\n%3i ",i);
      for (r=1; r<=nsd; r++)
	s_o << oform(" u(%d)=%12.5e  ",r,u()(r).valueNode(i));
      s_o << oform("   p=%12.5e",p().valueNode(i));
    }
    s_o << "\n\n";
  }
}

void DPNavierStokes:: compare2analyticalSolution ()
{
  // give an int as the analytical solution, create a routine that evaluates
  // the analytical solution for this int at a point and compare here, node
  // by node.
}


#undef Vector
#undef Handle

/* SCIRUN STUFF */
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Malloc/Allocator.h>
#include <Multitask/Task.h>
#include <TCL/TCLvar.h>
#include <values.h>
#include <Modules/DP/DPGridPort.h>
#include <Modules/DP/DPVecPort.h>
#include <Modules/DP/DPProblemPort.h>


class DPNavierStokesM : public Module {
public:
    DPGridIPort* ingrid;
    DPProblemOPort* outproblem;
    DPVecOPort* outnonlin;

    /* SCIRUN STUFF */
    DPNavierStokes* problem;
    DPNavierStokesM(const clString& id);
    virtual void execute();
    virtual Module* clone(int deep);
};

extern "C" {
Module* make_DPNavierStokes(const clString& id)
{
    return new DPNavierStokesM(id);
}
};

Module* DPNavierStokesM::clone(int deep)
{
    NOT_FINISHED("DPNavierStokes::clone");
    return 0;
}

extern int global_argc;
extern char** global_argv;

DPNavierStokesM::DPNavierStokesM(const clString& id)
: Module("DPNavierStokes", id, Filter)
{
    ingrid=new DPGridIPort(this, "DPGrid", DPGridIPort::Atomic);
    add_iport(ingrid);

    outproblem=new DPProblemOPort(this, "problem", DPProblemIPort::Atomic);
    add_oport(outproblem);
    outnonlin=new DPVecOPort(this, "Nonelinear", DPVecIPort::Atomic);
    add_oport(outnonlin);

    problem=0;
}


void DPNavierStokesM::execute()
{
    cerr << "NS is pid: " << getpid() << endl;
    DPGridHandle gridh;
    if(!ingrid->get(gridh))
	return;
    if(!problem){
	problem=new DPNavierStokes;
    }
    cerr << "Calling rebind...\n";
    problem->grid.rebind(gridh->grid);
    cerr << "Calling adm\n";
    ifstream in("3Dcase.i");
    if(!in){
	cerr << "Error opening 3Dcase.i\n";
    }
    s_i.rebind(in);
    problem->adm(global_menu);
    problem->driver();
    outnonlin->send(new DPVec(problem->nonlin_solution));
    outproblem->send(new DPProblem(problem));
}


struct DPInitialize {
    DPInitialize();
};

DPInitialize::DPInitialize()
{
    initDIFFPACK(global_argc, global_argv);
    global_menu.init("Penalty Navier Stokes simulator", "NS");
}

DPInitialize dp_init;

#include <Modules/DP/DPLinEqPort.h>

class DPMakeSystemM : public Module {
public:
    DPProblemIPort* inproblem;
    DPVecIPort* innonlin;
    DPLinEqOPort* outlineq;

    /* SCIRUN STUFF */
    DPMakeSystemM(const clString& id);
    virtual void execute();
    virtual Module* clone(int deep);
};

extern "C" {
Module* make_DPMakeSystem(const clString& id)
{
    return new DPMakeSystemM(id);
}
};

Module* DPMakeSystemM::clone(int deep)
{
    NOT_FINISHED("DPMakeSystem:clone");
    return 0;
}

DPMakeSystemM::DPMakeSystemM(const clString& id)
: Module("DPMakeSystem", id, Filter)
{
    inproblem=new DPProblemIPort(this, "DPProblem", DPProblemIPort::Atomic);
    add_iport(inproblem);

    innonlin=new DPVecIPort(this, "Nonelinear", DPVecIPort::Atomic);
    add_iport(innonlin);
    outlineq=new DPLinEqOPort(this, "Nonelinear", DPLinEqIPort::Atomic);
    add_oport(outlineq);
}


void DPMakeSystemM::execute()
{
    DPProblemHandle prob;
    if(!inproblem->get(prob))
	return;
    DPNavierStokes* problem=prob->problem;
    DPVecHandle nl;
    if(!innonlin->get(nl))
	return;
    doubleVec* nonlin_solution=nl->vec;
    cerr << "makeAndSolve...\n";
nonlin_solution->print(s_o,"nonlin_solution before makeAndSolve");
    problem->dof->vec2field (*nonlin_solution, problem->u());  // put most recent guess in u
                                         // (convenient in integrands)
    //if (nlsolver->getCurrentState().method == NEWTON_RAPHSON)
    problem->dof->fillEssBC2zero();
    //else
    //dof->unfillEssBC2zero();

  // make linear system
    cerr << "makeSystem\n";
    DPLinEq* lineq=new DPLinEq(new LinEqAdm(EXTERNAL_SOLUTION));
    doubleVec* linear_solution=new doubleVec;
    linear_solution->redim(nonlin_solution->size());
    lineq->lineq->attach (*linear_solution);
    lineq->sol=linear_solution;
    problem->makeSystem (problem->dof(), problem->elmv, NULL /* elmv has the integrands */, *lineq->lineq);
    outlineq->send(lineq);
}
