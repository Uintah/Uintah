/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/BCHelper.h>
#include <CCA/Components/Wasatch/WasatchBCHelper.h>

//-- Uintah Includes --//
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/CCVariable.h>

namespace Uintah{
  class SolverInterface;
}

namespace WasatchCore{

/**
 *  \class 	  Pressure
 *  \ingroup 	Expressions
 *  \ingroup	 WasatchCore
 *  \authors 	James C. Sutherland, Tony Saad, Amir Biglari
 *  \date 	   January, 2011
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
  typedef SpatialOps::SingleValueField TimeField;

  const Expr::Tag volFracTag_, rhoStarTag_;
  const bool doX_, doY_, doZ_, hasIntrusion_;
  bool didAllocateMatrix_;
  bool didMatrixUpdate_;
  bool hasMovingGeometry_;
  int  materialID_;
  int  rkStage_;
  const bool useRefPressure_;
  const double refPressureValue_;
  const Uintah::IntVector refPressureLocation_;
  const bool use3DLaplacian_;
  const bool enforceSolvability_;
  const bool isConstDensity_;
  
  Uintah::SolverInterface& solver_;
  const Uintah::VarLabel* matrixLabel_;
  const Uintah::VarLabel* pressureLabel_;
  const Uintah::VarLabel* prhsLabel_;
  
  DECLARE_FIELDS(TimeField, timestep_, t_)
  DECLARE_FIELDS(SVolField, pSource_, volfrac_, rhoStar_)
  DECLARE_FIELD(XVolField, fx_)
  DECLARE_FIELD(YVolField, fy_)
  DECLARE_FIELD(ZVolField, fz_)

  // interpolant operators
  typedef OperatorTypeBuilder< Interpolant, XVolField, SpatialOps::SSurfXField >::type  FxInterp;
  typedef OperatorTypeBuilder< Interpolant, YVolField, SpatialOps::SSurfYField >::type  FyInterp;
  typedef OperatorTypeBuilder< Interpolant, ZVolField, SpatialOps::SSurfZField >::type  FzInterp;
  const FxInterp* interpX_;
  const FyInterp* interpY_;
  const FzInterp* interpZ_;

  
  typedef OperatorTypeBuilder< Interpolant, SVolField, XVolField >::type  S2XInterpT;
  typedef OperatorTypeBuilder< Interpolant, SVolField, YVolField >::type  S2YInterpT;
  typedef OperatorTypeBuilder< Interpolant, SVolField, ZVolField >::type  S2ZInterpT;
  const S2XInterpT* s2XInterOp_;
  const S2YInterpT* s2YInterOp_;
  const S2ZInterpT* s2ZInterOp_;

  // divergence operators
  typedef SpatialOps::BasicOpTypes<SVolField>::DivX  DivX;
  typedef SpatialOps::BasicOpTypes<SVolField>::DivY  DivY;
  typedef SpatialOps::BasicOpTypes<SVolField>::DivZ  DivZ;
  const DivX* divXOp_;
  const DivY* divYOp_;
  const DivZ* divZOp_;
  
  typedef Uintah::CCVariable<Uintah::Stencil7> MatType;
  MatType matrix_;
  const Uintah::Patch* patch_;
  WasatchBCHelper* bcHelper_;

  Pressure( const std::string& pressureName,
            const std::string& pressureRHSName,
            const Expr::Tag& fxtag,
            const Expr::Tag& fytag,
            const Expr::Tag& fztag,
            const Expr::Tag& pSourceTag,
            const Expr::Tag& timesteptag,
            const Expr::Tag& volfractag,
            const Expr::Tag& rhoStarTag,
            const bool hasMovingGeometry,
            const bool       userefpressure,
            const double     refPressureValue,
            const Uintah::IntVector refPressureLocation,
            const bool       use3dlaplacian,
            const bool       enforceSolvability,
            const bool       isConstDensity,
            Uintah::SolverInterface& solver );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag fxt_, fyt_, fzt_, psrct_, dtt_, volfract_, rhoStarTag_;
    
    const bool hasMovingGeometry_;
    const bool userefpressure_;
    const double refpressurevalue_;
    const Uintah::IntVector refpressurelocation_;
    const bool use3dlaplacian_;
    const bool enforceSolvability_;
    const bool isConstDensity_;

    Uintah::SolverInterface& solver_;
  public:
    Builder( const Expr::TagList& result,
             const Expr::Tag& fxtag,
             const Expr::Tag& fytag,
             const Expr::Tag& fztag,
             const Expr::Tag& pSourceTag,
             const Expr::Tag& timesteptag,
             const Expr::Tag& volfractag,
             const Expr::Tag& rhoStarTag,
             const bool hasMovingGeometry,
             const bool       useRefPressure,
             const double     refPressureValue,
             const Uintah::IntVector refPressureLocation,
             const bool       use3DLaplacian,
             const bool       enforceSolvability,
             const bool       isConstDensity,
             Uintah::SolverInterface& solver );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  };

  ~Pressure();

  /**
   *  \brief allows WasatchCore::TaskInterface to reach in and give this
   *         expression the information requried to schedule the
   *         linear solver.
   */
  void schedule_solver( const Uintah::LevelP& level,
                        Uintah::SchedulerP sched,
                        const Uintah::MaterialSet* const materials,
                        const int RKStage );

  /**
   *  \brief Allows WasatchCore::TaskInterface to reach in set the boundary conditions
             on pressure at the appropriate time - namely after the linear solve. 
   */  
  // NOTE: Maybe we should not expose this to the outside?
  void schedule_set_pressure_bcs( const Uintah::LevelP& level,
                                  Uintah::SchedulerP sched,
                                  const Uintah::MaterialSet* const materials,
                                  const int RKStage );
  
  /**
   *  \brief allows WasatchCore::TaskInterface to reach in and provide
   *         this expression with a way to set the variables that it
   *         needs to.
   */
  void declare_uintah_vars( Uintah::Task& task,
                            const Uintah::PatchSubset* const patches,
                            const Uintah::MaterialSubset* const materials,
                            const int RKStage );
  
  void set_bchelper( WasatchBCHelper* bcHelper ) { bcHelper_ = bcHelper;}

  /**
   *  \brief allows WasatchCore::TaskInterface to reach in and provide
   *         this expression with a way to retrieve Uintah-specific
   *         variables from the data warehouse.
   *
   *  This should be done very carefully.  Any "external" dependencies
   *  should not be introduced here.  This is only for variables that
   *  are very uintah-specific and only used internally to this
   *  expression.  Specifically, the pressure-rhs field and the LHS
   *  matrix.  All other variables should be expressed as dependencies
   *  through the DECLARE_FIELD MACRO.
   */
  void bind_uintah_vars( Uintah::DataWarehouse* const dw,
                         const Uintah::Patch* const patch,
                         const int material,
                         const int RKStage );
  /**
   * \brief Calculates pressure coefficient matrix for variable density flows.
   */
  void setup_matrix( const SVolField* const rhoStar,
                     const SVolField* const volfrac );
  
  /**
   * \brief Calculates pressure coefficient matrix for constant density flows.
   */
  void setup_matrix( const SVolField* const volfrac );

  void process_embedded_boundaries( const SVolField& volfrac );

  /**
   * \brief Special function to apply pressure boundary conditions after the pressure solve.
            This is needed because process_after_evaluate is executed before the pressure solve.
            We may need to split the pressure expression into a pressure_rhs and a pressure...
   */  
  void process_bcs ( const Uintah::ProcessorGroup* const pg,
                     const Uintah::PatchSubset* const patches,
                     const Uintah::MaterialSubset* const materials,
                     Uintah::DataWarehouse* const oldDW,
                     Uintah::DataWarehouse* const newDW );
  
  //Uintah::CCVariable<Uintah::Stencil7> pressure_matrix(){ return matrix_ ;}
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};

} // namespace WasatchCore

#endif // Pressure_Expr_h
