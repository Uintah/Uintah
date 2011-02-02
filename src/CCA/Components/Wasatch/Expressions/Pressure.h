#ifndef Pressure_Expr_h
#define Pressure_Expr_h
//-- SpatialOps includes --//
#include <spatialops/structured/FVStaggered.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>

//-- ExprLib Includes --//
#include <expression/Expr_Expression.h>

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/Operators/Operators.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>

//-- Uintah Includes --//
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/Stencil7.h>
#include <Core/Grid/Variables/CCVariable.h>

namespace Uintah{
  class SolverInterface;
  class SolverParameters;
}

namespace Wasatch{

/**
 *  \class Pressure
 *
 *  \brief Expression to form and solve the poisson system for pressure.
 *  \author James C. Sutherland
 *  \author Tony Saad
 *  \date January, 2011
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
  const Expr::Tag fxt_, fyt_, fzt_, d2rhodt2t_;

  const bool doX_, doY_, doZ_, doDens_;

  const Uintah::SolverParameters& solverParams_;
  Uintah::SolverInterface& solver_;
  const Uintah::VarLabel* matrixLabel_;
  const Uintah::VarLabel* pressureLabel_;
  const Uintah::VarLabel* prhsLabel_;

  const SVolField* d2rhodt2_;
  const XVolField* fx_;
  const YVolField* fy_;
  const ZVolField* fz_;

  // build interpolant operators
  typedef OperatorTypeBuilder< Interpolant, XVolField, SVolField >::type  fxInterp;
  typedef OperatorTypeBuilder< Interpolant, YVolField, SVolField >::type  fyInterp;
  typedef OperatorTypeBuilder< Interpolant, ZVolField, SVolField >::type  fzInterp;
  const fxInterp* interpX_;
  const fyInterp* interpY_;
  const fzInterp* interpZ_;
  
  typedef Uintah::CCVariable<Uintah::Stencil7> MatType;
  MatType matrix_;

  Pressure( const Expr::Tag& fxtag,
            const Expr::Tag& fytag,
            const Expr::Tag& fztag,
            const Expr::Tag& d2rhodt2tag,
            const Uintah::SolverParameters& solverParams,
            Uintah::SolverInterface& solver,
            const Expr::ExpressionID& id,
            const Expr::ExpressionRegistry& reg );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag fxt_, fyt_, fzt_, d2rhodt2t_;
    const Uintah::SolverParameters& sparams_;
    Uintah::SolverInterface& solver_;
  public:
    Builder( Expr::Tag& fxtag,
             Expr::Tag& fytag,
             Expr::Tag& fztag,
             const Expr::Tag& d2rhodt2tag,
             Uintah::SolverParameters& sparams,
             Uintah::SolverInterface& solver );

    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg ) const;
  };

  ~Pressure();

  /**
   *  \brief allows Wasatch::TaskInterface to reach in and give this
   *         expression the information requried to schedule the
   *         linear solver.
   */
  void schedule_solver( const Uintah::LevelP& level, Uintah::SchedulerP& sched, const Uintah::MaterialSet* materials );

  /**
   *  \brief allows Wasatch::TaskInterface to reach in and provide
   *         this expression with a way to set the variables that it
   *         needs to.
   */
  void declare_uintah_vars( Uintah::Task& task,
                            const Uintah::PatchSubset* const patches,
                            const Uintah::MaterialSubset* const materials );

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
                         const Uintah::PatchSubset* const patches,
                         const Uintah::MaterialSubset* const materials );
  

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};

} // namespace Wasatch

#endif // Pressure_Expr_h
