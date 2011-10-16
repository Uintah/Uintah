#include "Pressure.h"

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/BCHelperTools.h>

//-- Uintah Includes --//
#include <CCA/Ports/SolverInterface.h>
  
//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/structured/MemoryWindow.h>

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
  
    didAllocateMatrix_(false),

    solverParams_( solverParams ),
    solver_( solver ),

    // note that this does not provide any ghost entries in the matrix...
    matrixLabel_  ( Uintah::VarLabel::create( "pressure_matrix", Uintah::CCVariable<Uintah::Stencil7>::getTypeDescription() ) ),
    pressureLabel_( Uintah::VarLabel::create( (this->names())[0].name(), 
                                              Wasatch::get_uintah_field_type_descriptor<SVolField>(),
                                              Wasatch::get_uintah_ghost_descriptor<SVolField>() ) ),
    prhsLabel_    ( Uintah::VarLabel::create( (this->names())[1].name(), 
                                              Wasatch::get_uintah_field_type_descriptor<SVolField>(),
                                              Wasatch::get_uintah_ghost_descriptor<SVolField>() ) )
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
                           Uintah::SchedulerP sched,
                           const Uintah::MaterialSet* const materials,
                           int RKStage)
{
  // TODO: investigate why projection only works for the first RK stage when running
  // in parallel (specifically hypre)
  solver_.scheduleSolve( level, sched, materials, matrixLabel_, 
                        Uintah::Task::NewDW, pressureLabel_, true, prhsLabel_, Uintah::Task::NewDW, 0, Uintah::Task::OldDW, &solverParams_ );

//  if (RKStage==1) {
//    solver_.scheduleSolve( level, sched, materials, matrixLabel_, 
//                          Uintah::Task::NewDW, pressureLabel_, true, prhsLabel_, Uintah::Task::NewDW, 0, Uintah::Task::OldDW, &solverParams_ );
//  } else {
////    solver_.scheduleSolve( level, sched, materials, matrixLabel_, 
////                          Uintah::Task::NewDW, pressureLabel_, true, prhsLabel_, Uintah::Task::NewDW, pressureLabel_, Uintah::Task::NewDW, &solverParams_ );
//  }

}

//--------------------------------------------------------------------

void
Pressure::declare_uintah_vars( Uintah::Task& task,
                               const Uintah::PatchSubset* const patches,
                               const Uintah::MaterialSubset* const materials,
                              int RKStage)
{
  if (RKStage ==1) 
    task.computes( matrixLabel_, patches, Uintah::Task::NormalDomain, materials, Uintah::Task::NormalDomain );
  else 
    task.modifies( matrixLabel_, patches, Uintah::Task::NormalDomain, materials, Uintah::Task::NormalDomain );
}

//--------------------------------------------------------------------
  
void
Pressure::bind_uintah_vars( Uintah::DataWarehouse* const dw,
                           const Uintah::Patch* const patch,
                           const int material,
                           int RKStage)
{ 
  if (didAllocateMatrix_) {
    //std::cout << "RKStage "<< RKStage << "did allocate matrix \n";
    // Todd: instead of checking for allocation - check for new timestep or some other ingenious solution
    // check for transferfrom - transfer matrix from old to new DW
    if (RKStage==1) dw->put( matrix_, matrixLabel_, material, patch ); 
    else dw->getModifiable( matrix_, matrixLabel_, material, patch ); 
    //setup_matrix(patch);
  } else {
    //std::cout << "RKStage "<< RKStage << "before allocate and put matrix \n";
    dw->allocateAndPut( matrix_, matrixLabel_, material, patch );    
    ///std::cout << "after allocate and put matrix \n";    
    setup_matrix(patch);
    didAllocateMatrix_=true;
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
  
  divXOp_ = opDB.retrieve_operator<DivX>();
  divYOp_ = opDB.retrieve_operator<DivY>();
  divZOp_ = opDB.retrieve_operator<DivZ>();
  
}

//--------------------------------------------------------------------

void
Pressure::setup_matrix(const Uintah::Patch* const patch) 
{
  // construct the coefficient matrix: \nabla^2
  // We should probably move the matrix construction to the evaluate() method.
  // need a way to get access to the patch so that we can loop over the cells.
  // p is current cell
  double p = 0.0;
  // n: north, s: south, e: east, w: west, t: top, b: bottom coefficient
  double n=0.0, s=0.0, e=0.0, w=0.0, t=0.0, b=0.0;
  
  IntVector l,h;
  l = patch->getCellLowIndex();
  h = patch->getCellHighIndex();
  const Uintah::Vector spacing = patch->dCell();
  
  if (doX_) {
    const double dx2 = spacing[0]*spacing[0];
    e = 1.0/dx2;
    w = 1.0/dx2;
    p -= 2.0/dx2;
  }
  if (doY_) {
    const double dy2 = spacing[1]*spacing[1];
    n = 1.0/dy2;
    s = 1.0/dy2;
    p -= 2.0/dy2;
  }
  if (doZ_) {
    const double dz2 = spacing[2]*spacing[2];
    t = 1.0/dz2;
    b = 1.0/dz2;
    p -= 2.0/dz2;
  }

  for(Uintah::CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){    
    IntVector iCell = *iter;
    Uintah::Stencil7&  coefs = matrix_[iCell];
    coefs.w = w;
    coefs.e = e;
    coefs.n = n;
    coefs.s = s;
    coefs.b = b;
    coefs.t = t;
    coefs.p = p;           
  }
  
  //
//  typedef std::vector<SVolField*> SVolFieldVec;
//  SVolFieldVec& results = this->get_value_vec();
//  SVolField& pressureField = *results[0];
//  SVolField& pRhs = *results[1];
//  pRhs = 0.0;
//  
//  set_pressure_bc(this->name(),matrix_, pressureField, pRhs, patch);
}
    
//--------------------------------------------------------------------
    
void
Pressure::evaluate()
{
  typedef std::vector<SVolField*> SVolFieldVec;
  SVolFieldVec& results = this->get_value_vec();

  // jcs: can we do the linear solve in place? We probably can. If so,
  // we would only need one field, not two...
  //SVolField* p = results[0];
  //SVolField* r = results[1];
  SVolField& pressure = *results[0];
  //SVolField& rhs = *r;
  SVolField& rhs = *results[1];
  rhs = 0.0;
  //
  // set_pressure_bc((this->names())[0],matrix_, pressure, rhs, patch_);
  //
  namespace SS = SpatialOps::structured;
  const SS::MemoryWindow& w = rhs.window_with_ghost();
  SpatialOps::SpatFldPtr<SS::SSurfXField> tmpx  = SpatialOps::SpatialFieldStore<SS::SSurfXField >::self().get( w );
  SpatialOps::SpatFldPtr<SS::SSurfYField> tmpy  = SpatialOps::SpatialFieldStore<SS::SSurfYField >::self().get( w );
  SpatialOps::SpatFldPtr<SS::SSurfZField> tmpz  = SpatialOps::SpatialFieldStore<SS::SSurfZField >::self().get( w );
  SpatialOps::SpatFldPtr<SVolField> tmp = SpatialOps::SpatialFieldStore<SVolField>::self().get( rhs );
  //set_pressure_bc(this->name(), pressure, rhs, patch);
  //___________________________________________________
  // calculate the RHS field for the poisson solve.
  // Note that this is "automagically" plugged into
  // the solver in the "schedule_solver" method.
  if( doX_ ){
    interpX_->apply_to_field( *fx_, *tmpx );
    divXOp_->apply_to_field( *tmpx, *tmp );
    rhs += *tmp;
  }

  if( doY_ ){
    interpY_->apply_to_field( *fy_, *tmpy );
    divYOp_->apply_to_field( *tmpy, *tmp );
    rhs += *tmp;
  }

  if( doZ_ ){
    interpZ_->apply_to_field( *fz_, *tmpz );
    divZOp_->apply_to_field( *tmpz, *tmp );
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
