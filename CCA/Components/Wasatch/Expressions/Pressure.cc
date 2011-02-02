#include "Pressure.h"

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <CCA/Components/Wasatch/FieldTypes.h>

//-- Uintah Includes --//
#include <CCA/Ports/SolverInterface.h>
  
//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

namespace Wasatch {

Pressure::Pressure( const Expr::Tag& fxtag,
                    const Expr::Tag& fytag,
                    const Expr::Tag& fztag,
                    const Expr::Tag& d2rhodt2tag,
                    const Uintah::SolverParameters& solverParams,
                    Uintah::SolverInterface& solver,
                    const Expr::ExpressionID& id,
                    const Expr::ExpressionRegistry& reg )
  : Expr::Expression<SVolField>(id,reg),

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
    matrixLabel_  ( Uintah::VarLabel::create( "pressure_matrix", Uintah::CCVariable<Uintah::Stencil7>::getTypeDescription() ) ),
  pressureLabel_( Uintah::VarLabel::create( "pressure_field", 
                                           Wasatch::getUintahFieldTypeDescriptor<SVolField>(),
                                           Wasatch::getUintahGhostDescriptor<SVolField>() ) ),
  prhsLabel_    ( Uintah::VarLabel::create( "pressure_rhs", 
                                           Wasatch::getUintahFieldTypeDescriptor<SVolField>(),
                                           Wasatch::getUintahGhostDescriptor<SVolField>() ) )
  
{
}

//--------------------------------------------------------------------

Pressure::~Pressure()
{
  Uintah::VarLabel::destroy( pressureLabel_ );
  Uintah::VarLabel::destroy( prhsLabel_     );
  Uintah::VarLabel::destroy( matrixLabel_   );
}

//--------------------------------------------------------------------

void
Pressure::schedule_solver( const Uintah::LevelP& level,
                          Uintah::SchedulerP& sched,
                          const Uintah::MaterialSet* materials )
{
  // need to get the pressure label...
  // need to get the pressure rhs label...
  const Uintah::VarLabel* pressure_lbl;
  const Uintah::VarLabel* rhs_lbl;
  //solver->scheduleSolve( level, sched, materials, matrixLabel_, 
  //                      Task::NewDW, pressureLabel_, false, prshLabel_, Task::NewDW, 0, Task::OldDW, solverParams_ );
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
Pressure::bind_uintah_vars( Uintah::DataWarehouse* const dw,
                            const Uintah::PatchSubset* const patches,
                            const Uintah::MaterialSubset* const materials )
{
  // We should probably move the matrix construction to the evaluate() method.
  // need a way to get access to the patch so that we can loop over the cells.
  // p is current cell
  int p = 0;
  // n: north, s: south, e: east, w: west, t: top, b: bottom coefficient
  int n=0, s=0, e=0, w=0, t=0, b=0;
  
  if (doX_) {
    p += 2;
    e = -1;
    w = -1;
  }
  if (doY_) {
    p += 2;
    n = -1;
    s = -1;
  }
  if (doZ_) {
    p += 2;
    t = -1;
    b = -1;
  }
  
  //
  for( int ip=0; ip<patches->size(); ++ip ){
    const Uintah::Patch* const patch = patches->get( ip );
    for( int im=0; im<materials->size(); ++im ){
      const int material = materials->get( im );
      dw->allocateAndPut( matrix_, matrixLabel_, material, patch );
      //
      // construct the coefficient matrix: \nabla^2
      for(Uintah::CellIterator iter(patch->getExtraCellIterator()); !iter.done(); iter++){
        IntVector iCell = *iter;
        Uintah::Stencil7&  coefs = matrix_[iCell];
        coefs.p = p; 
        coefs.n = n;   coefs.s = s;
        coefs.e = e;   coefs.w = w; 
        coefs.t = t;   coefs.b = b;
      }
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

  if( doX_    )  fx_       = &xvfm.field_ref( fxt_       );
  if( doY_    )  fy_       = &yvfm.field_ref( fyt_       );
  if( doZ_    )  fz_       = &zvfm.field_ref( fzt_       );
  if( doDens_ )  d2rhodt2_ = &svfm.field_ref( d2rhodt2t_ );
}

//--------------------------------------------------------------------

void
Pressure::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  interpX_ = opDB.retrieve_operator<fxInterp>();
  interpY_ = opDB.retrieve_operator<fyInterp>();
  interpZ_ = opDB.retrieve_operator<fzInterp>();
}

//--------------------------------------------------------------------

void
Pressure::evaluate()
{
  typedef std::vector<SVolField*> SVolFieldVec;
  SVolFieldVec& results = this->get_value_vec();

  // jcs: can we do the linear solve in place? We probably can. If so,
  // we would only need one field, not two...
  SVolField* p = results[0];
  SVolField* r = results[1];
  SVolField& pressure = *p;
  SVolField& rhs = *r;
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
    rhs += *d2rhodt2_;
  }

  //_________________________________________________
  // construct the LHS matrix, solve for pressure....
  // matrix_ = ????
}

//--------------------------------------------------------------------

Pressure::Builder::Builder( Expr::Tag& fxtag,
                            Expr::Tag& fytag,
                            Expr::Tag& fztag,
                            const Expr::Tag& d2rhodt2tag,
                            Uintah::SolverParameters& sparams,
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
