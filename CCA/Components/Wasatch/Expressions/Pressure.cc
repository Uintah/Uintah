#include "Pressure.h"

//-- Wasatch Includes --//

//-- Uintah Includes --//
#include <CCA/Ports/SolverInterface.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>


Pressure::Pressure( const Expr::Tag& fxtag,
                    const Expr::Tag& fytag,
                    const Expr::Tag& fztag,
                    const Expr::Tag& d2rhodt2tag,
                    const Uintah::SolverParameters& solverParams,
                    Uintah::SolverInterface& solver,
                    const Expr::ExpressionID& id,
                    const Expr::ExpressionRegistry& reg )
  : Expr::Expression<FieldT>(id,reg),

    fxt_( fxtag ),
    fyt_( fytag ),
    fzt_( fztag ),

    d2rhodt2t_( d2rhodt2tag ),

    doX_( fxtag != Expr::Tag() ),
    doY_( fytag != Expr::Tag() ),
    doZ_( fztag != Expr::Tag() ),

    doDens_( d2rhodt2tag != Expr::Tag() ),

    solverParams_( solverParams ),
    solver_( solver ),

    // note that this does not provide any ghost entries in the matrix...
    matrixLabel_( Uintah::VarLabel::create( "pressure_matrix", MatType::getTypeDescription() ) )
{
}

//--------------------------------------------------------------------

Pressure::~Pressure()
{
  Uintah::VarLabel::destroy( prhsLabel_   );
  Uintah::VarLabel::destroy( matrixLabel_ );
}

//--------------------------------------------------------------------

void
Pressure::schedule_solver( const LevelP& level,
                           SchedulerP& sched,
                           const MaterialSet* materials )
{
  // need to get the pressure label...
  // need to get the pressure rhs label...
  /*
  solver->scheduleSolve( level, sched, materials, matrixLabel_, 
                        Task::NewDW, lb_->pressure, false, lb_->pressure_rhs, Task::NewDW, 0, Task::OldDW, solverParams_ );
  */
}

//--------------------------------------------------------------------

void
Pressure::declare_uintah_vars( Uintah::Task& task,
                               const Uintah::PatchSubset* const patches,
                               const Uintah::MaterialSubset* const materials )
{
  // need to verify that this is consistent with what is being done in the solver test...
  task.computes( matrixLabel_, patches, Uintah::Task::NormalDomain, materials, Uintah::Task::NormalDomain );
}

//--------------------------------------------------------------------

void
Pressure::bind_uintah_vars( Uintah::DataWarehouse* const,
                            const Uintah::PatchSubset* const patches,
                            const Uintah::MaterialSubset* const materials )
{
  for( int ip=0; ip<patches->size(); ++ip ){
    const Uintah::Patch* const patch = patches->get( ip );
    for( int im=0; im<materials->size(); ++im ){
      const int material = materials->get( im );
      dw->allocateAndPut( matrix_, matrixLabel_, material, patch );
    }
  }
}

//--------------------------------------------------------------------

void
Pressure::advertise_dependents( Expr::ExprDeps& exprDeps )
{
  if( doX_    )  exprDeps.requires_expression( fxt_ );
  if( doY_    )  exprDeps.requires_expression( fyt_ );
  if( doZ_    )  exprDeps.requires_expression( fzt_ );
  if( doDens_ )  exprDeps.requires_expression( d2rhodt2t_ );
}

//--------------------------------------------------------------------

void
Pressure::bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<SVolField>& svfm = fml.field_manager<SVolField>();
  const Expr::FieldManager<XVolField>& xvfm = fml.field_manager<XVolField>();
  const Expr::FieldManager<YVolField>& yvfm = fml.field_manager<YVolField>();
  const Expr::FieldManager<ZVolField>& zvfm = fml.field_manager<ZVolField>();

  if( doX_    )  fx_       = &field_ref( fxt_       );
  if( doX_    )  fy_       = &field_ref( fyt_       );
  if( doX_    )  fz_       = &field_ref( fzt_       );
  if( doDens_ )  d2rhodt2_ = &field_ref( d2rhodt2t_ );
}

//--------------------------------------------------------------------

void
Pressure::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  interpX_ = opDB.retrieve_operator<InterpXVolSVol>();
  interpY_ = opDB.retrieve_operator<InterpYVolSVol>();
  interpZ_ = opDB.retrieve_operator<InterpZVolSVol>();
}

//--------------------------------------------------------------------

void
Pressure::evaluate()
{
  std::vector<SVolField>& results = this->get_value_vec();

  // jcs: can we do the linear solve in place? We probably can. If so,
  // we would only need one field, not two...
  SVolField& pressure = results[0];
  SVolField& rhs = results[1];

  SpatialOps::SpatFldPtr<SVolField> tmp = SpatialOps::SpatialFieldStore<SVolField>::self().get( pressure );

  //___________________________________________________
  // calculate the RHS field for the poisson solve.
  // Note that this is "automagically" plugged into
  // the solver in the "schedule_solver" method.
  if( doX_ ){
    interpX_->apply_to_field( *fx_, *tmp );
    rhs += *tmp;
  }

  if( doY_ ){
    interpY_->apply_to_field( *fy_, *tmp );
    rhs += *tmp;
  }

  if( doZ_ ){
    interpZ_->apply_to_field( *fz_, *tmp );
    rhs += *tmp;
  }

  if( doDens_ ){
    rhs += *d2rhodt2t_;
  }

  //_________________________________________________
  // construct the LHS matrix, solve for pressure....
  // matrix_ = ????
}

//--------------------------------------------------------------------

Pressure::Builder::Builder( const Expr::Tag& fxtag,
                            const Expr::Tag& fytag,
                            const Expr::Tag& fztag,
                            const Expr::Tag& d2rhodt2tag,
                            const Uintah::SolverParameters& sparams,
                            Uintah::SolverInterface& solver )
 : fxt_( fxtag ),
   fyt_( fytag ),
   fzt_( fztag ),
   d2rhodt2t_( d2rhodt2tag ),
   sparams_( sparams ),
   solver_( solver )
{}

//--------------------------------------------------------------------

Expr::ExpressionBase*
Pressure::Builder::build( const Expr::ExpressionID& id,
                          const Expr::ExpressionRegistry& reg ) const
{
  return new Pressure( fxt_, fyt_, fzt_, d2rhodt2t_, sparams_, solver_, id, reg );
}

} // namespace Wasatch
