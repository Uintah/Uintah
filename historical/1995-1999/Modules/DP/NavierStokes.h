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

#ifndef NavierStokes_h_IS_INCLUDED
#define NavierStokes_h_IS_INCLUDED

#include <FEM.h>                // finite element toolbox
#include <DegFreeFE.h>          // field dof <-> linear system dof
#include <LinEqAdm.h>           // linear solver toolbox
#include <TimePrm.h>            // time discretization parameters
#include <NonLinEqSolver.h>     // nonlinear solver interface
#include <NonLinEqSolver_prm.h> // parameters for nonlinear solvers
#include <NonLinEqSolverUDC.h>  // user's definition of nonlinear PDEs
#include <Store4Plotting.h>     // storage of data for later visualization
#include <MenuUDC.h>

/* outline of classes:
  NavierStokes     - main simulator
  ElmMatVecPenalty - computation of elemental matrix/vector for penalty term
  ReducedIntg      - the integrand for reduced integration (penalty term)
  NonReducedIntg   - the integrand for "normal" integration
  PressureIntg     - integrand (div v) for pressure recovery and smoothing

  For documentation see the report 'The Diffpack Implementation of
  a Navier-Stokes Solver Based on the Penalty Function Method'.
*/

class NavierStokes;

/*<NonReducedIntg:*/
class NonReducedIntg : public IntegrandCalc
{
  NavierStokes& data;
public:
  NonReducedIntg (NavierStokes& data_) : data(data_) {}
 ~NonReducedIntg () {}
  virtual void integrands (ElmMatVec& elmat, FiniteElement& fe);
};
/*>NonReducedIntg:*/

/*<ReducedIntg:*/
class ReducedIntg : public IntegrandCalc
{
  NavierStokes& data;
public:
  ReducedIntg (NavierStokes& data_) : data(data_) {}
 ~ReducedIntg () {}
  virtual void integrands (ElmMatVec& elmat, FiniteElement& fe);
};
/*>ReducedIntg:*/


/*<ElmMatVecPenalty:*/
class ElmMatVecPenalty : public ElmMatVecCalcStd
{
  NavierStokes& data;
  NonReducedIntg v_terms;
  ReducedIntg lambda_term;
public:
  ElmMatVecPenalty (NavierStokes& data_)
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

/*<NavierStokes:*/
class NavierStokes : public FEM, public NonLinEqSolverUDC, public MenuUDC,
                     public Store4Plotting
{
  friend class NonReducedIntg;
  friend class    ReducedIntg;
  friend class ElmMatVecPenalty;
  friend class PressureIntg;

protected:
  Handle(GridFE)         grid;            // finite element mesh
  Handle(DegFreeFE)      dof;             // matrix dof <-> u dof
  Handle(FieldsFE)       u;               // velocity field
  Handle(FieldsFE)       u_prev;          // velocity field
  Handle(FieldFE)        p;               // pressure field (derived quantity)

  TimePrm                tip;             // time discretization parameters

  Vec(real)              nonlin_solution; // nonlinear solution
  Vec(real)              linear_solution; // solution of linear subsystem
  prm(NonLinEqSolver)    nlsolver_prm;    // init prm for nlsolver
  Handle(NonLinEqSolver) nlsolver;        // interface to nonlinear solvers
  LinEqAdm               lineq;           // interface to linear solvers & data

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
  NavierStokes ();
 ~NavierStokes ();

  void define (MenuSystem& menu, int level = MAIN);
  void scan   (MenuSystem& menu);
  void adm    (MenuSystem& menu);
  void driver ();
  BooLean ok () const;

  void fillEssBC ();
  virtual void makeAndSolveLinearSystem ();  // for solution by NonLinEqSolver

protected:
  void inletVelocity (Ptv(real)& v, const Ptv(real)& x);
  void setIC ();
  void timeLoop ();
  void solveAtThisTimeLevel ();
  void calcDerivedQuantities ();
  void saveResults ();
  void updateDataStructures ();
  void compare2analyticalSolution ();
};
/*>NavierStokes:*/

/*Class:NavierStokes

NAME:              
NavierStokes - simulator for incompressible fluid flow (penalty method)

SYNTAX:	           @NavierStokes

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
 If the "NavierStokes" class is extended to handle mixed finite elements,
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
  NavierStokes& data;
public:
  PressureIntg (NavierStokes& data_) : data(data_) {}
 ~PressureIntg () {}
  virtual void integrands (ElmMatVec& elmat, FiniteElement& fe);
};
/*>PressureIntg:*/

#endif
/* LOG HISTORY of this file:
 * $Log$
 * Revision 1.1  1997/04/30 03:01:18  sparker
 * Checking in everything I've done in the last 1000 years
 *
*/
