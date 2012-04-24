/*
 * Copyright (c) 2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef Pressure_Expr_h
#define Pressure_Expr_h

//-- ExprLib Includes --//
#include <expression/Expression.h>

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/Operators/Operators.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

//-- Uintah Includes --//
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/Stencil4.h>
#include <Core/Grid/Variables/CCVariable.h>

namespace Uintah{
  class SolverInterface;
  class SolverParameters;
}

namespace Wasatch{

/**
 *  \brief obtain the tag for the pressure
 */
Expr::Tag pressure_tag();

/**
 *  \class 	Pressure
 *  \ingroup 	Expressions
 *  \ingroup	WasatchCore
 *  \author 	James C. Sutherland
 *  \author 	Tony Saad
 *  \date 	January, 2011
 *
 *  \brief Expression to form and solve the poisson system for pressure.
 *
 *  NOTE: this expression BREAKS WITH CONVENTION!  Notably, it has
 *  uintah tenticles that reach into it, and mixes SpatialOps and
 *  Uintah constructs.  This is because we don't (currently) have a
 *  robust interface to deal with parallel linear solves through the
 *  expression library, but Uintah has a reasonably robust interface.
 *
 *  This expression does play well with expression graphs, however.
 *  There are only a few places where Uintah reaches in.
 *
 *  Because of the hackery going on here, this expression is placed in
 *  the Wasatch namespace.  This should reinforce the concept that it
 *  is not intended for external use.
 */
class Pressure
 : public Expr::Expression<SVolField>
{
  const Expr::Tag fxt_, fyt_, fzt_, dilatationt_, d2rhodt2t_, timestept_;

  const bool doX_, doY_, doZ_, doDens_;
  bool didAllocateMatrix_;
  int  materialID_;
  const bool useRefPressure_;
  const double refPressureValue_;
  const SCIRun::IntVector refPressureLocation_;
  const bool use3DLaplacian_;
  
  const Uintah::SolverParameters& solverParams_;
  Uintah::SolverInterface& solver_;
  const Uintah::VarLabel* matrixLabel_;
  const Uintah::VarLabel* pressureLabel_;
  const Uintah::VarLabel* prhsLabel_;
  
  const double* timestep_;

  const SVolField* dilatation_;
  const SVolField* d2rhodt2_;
  const XVolField* fx_;
  const YVolField* fy_;
  const ZVolField* fz_;

  // build interpolant operators
  typedef OperatorTypeBuilder< Interpolant, XVolField, SpatialOps::structured::SSurfXField >::type  FxInterp;
  typedef OperatorTypeBuilder< Interpolant, YVolField, SpatialOps::structured::SSurfYField >::type  FyInterp;
  typedef OperatorTypeBuilder< Interpolant, ZVolField, SpatialOps::structured::SSurfZField >::type  FzInterp;
  const FxInterp* interpX_;
  const FyInterp* interpY_;
  const FzInterp* interpZ_;

  // divergence operators
  typedef SpatialOps::structured::BasicOpTypes<SVolField>::DivX  DivX;
  typedef SpatialOps::structured::BasicOpTypes<SVolField>::DivY  DivY;
  typedef SpatialOps::structured::BasicOpTypes<SVolField>::DivZ  DivZ;
  const DivX* divXOp_;
  const DivY* divYOp_;
  const DivZ* divZOp_;

  typedef Uintah::CCVariable<Uintah::Stencil4> MatType;
  MatType matrix_;
  const Uintah::Patch* patch_;

  Pressure( const std::string& pressureName,
            const std::string& pressureRHSName,
            const Expr::Tag& fxtag,
            const Expr::Tag& fytag,
            const Expr::Tag& fztag,
            const Expr::Tag& diltationtag,
            const Expr::Tag& d2rhodt2tag,
            const Expr::Tag& timesteptag,
            const bool       userefpressure,
            const double     refPressureValue,
            const SCIRun::IntVector refPressureLocation,
            const bool       use3dlaplacian,
            const Uintah::SolverParameters& solverParams,
            Uintah::SolverInterface& solver );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag fxt_, fyt_, fzt_, dilatationt_, d2rhodt2t_, timestept_;
    const bool userefpressure_;
    const double refpressurevalue_;
    const SCIRun::IntVector refpressurelocation_;
    const bool use3dlaplacian_;
    const Uintah::SolverParameters& sparams_;
    Uintah::SolverInterface& solver_;
  public:
    Builder( const Expr::TagList& result,
             const Expr::Tag& fxtag,
             const Expr::Tag& fytag,
             const Expr::Tag& fztag,
             const Expr::Tag& diltationtag,            
             const Expr::Tag& d2rhodt2tag,
             const Expr::Tag& timesteptag,
             const bool       useRefPressure,
             const double     refPressureValue,
             const SCIRun::IntVector refPressureLocation,
             const bool       use3DLaplacian,            
             const Uintah::SolverParameters& sparams,
             Uintah::SolverInterface& solver );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  };

  ~Pressure();

  /**
   *  \brief allows Wasatch::TaskInterface to reach in and give this
   *         expression the information requried to schedule the
   *         linear solver.
   */
  void schedule_solver( const Uintah::LevelP& level,
                        Uintah::SchedulerP sched,
                        const Uintah::MaterialSet* const materials,
                        const int RKStage );

  /**
   *  \brief allows Wasatch::TaskInterface to reach in and provide
   *         this expression with a way to set the variables that it
   *         needs to.
   */
  void declare_uintah_vars( Uintah::Task& task,
                            const Uintah::PatchSubset* const patches,
                            const Uintah::MaterialSubset* const materials,
                            const int RKStage );

  void set_patch( const Uintah::Patch* const patch ){ patch_ = const_cast<Uintah::Patch*> (patch); }

  /**
   *  \brief allows Wasatch::TaskInterface to reach in and provide
   *         this expression with a way to retrieve Uintah-specific
   *         variables from the data warehouse.
   *
   *  This should be done very carefully.  Any "external" dependencies
   *  should not be introduced here.  This is only for variables that
   *  are very uintah-specific and only used internally to this
   *  expression.  Specifically, the pressure-rhs field and the LHS
   *  matrix.  All other variables should be expressed as dependencies
   *  through the advertise_dependents method.
   */
  void bind_uintah_vars( Uintah::DataWarehouse* const dw,
                         const Uintah::Patch* const patch,
                         const int material,
                         const int RKStage );
  /**
   * \brief Calculates pressure coefficient matrix.
   */
  void setup_matrix();

  //Uintah::CCVariable<Uintah::Stencil7> pressure_matrix(){ return matrix_ ;}

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};

} // namespace Wasatch

#endif // Pressure_Expr_h
