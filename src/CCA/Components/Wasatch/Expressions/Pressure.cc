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

#include "Pressure.h"

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/BCHelperTools.h>

//-- Uintah Includes --//
#include <CCA/Ports/SolverInterface.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Material.h>

#include <Core/Grid/Variables/VarTypes.h>  // delt_vartype
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <CCA/Ports/LoadBalancer.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/structured/MemoryWindow.h>
#include <spatialops/FieldExpressions.h>

namespace Wasatch {

//==================================================================

Expr::Tag pressure_tag()
{
  return Expr::Tag( "pressure", Expr::STATE_NONE );
}

//==================================================================

Pressure::Pressure( const std::string& pressureName,
                    const std::string& pressureRHSName,
                    const Expr::Tag& fxtag,
                    const Expr::Tag& fytag,
                    const Expr::Tag& fztag,
                    const Expr::Tag& dilatationtag,                   
                    const Expr::Tag& d2rhodt2tag,
                    const Expr::Tag& timesteptag,
                    const bool       useRefPressure,
                    const double     refPressureValue,
                    const SCIRun::IntVector refPressureLocation,
                    const bool       use3DLaplacian,                   
                    const Uintah::SolverParameters& solverParams,
                    Uintah::SolverInterface& solver )
  : Expr::Expression<SVolField>(),

    fxt_( fxtag ),
    fyt_( fytag ),
    fzt_( fztag ),

    dilatationt_ ( dilatationtag ),
    d2rhodt2t_( d2rhodt2tag ),

    timestept_( timesteptag ),
  
    doX_( fxtag != Expr::Tag() ),
    doY_( fytag != Expr::Tag() ),
    doZ_( fztag != Expr::Tag() ),

    doDens_( d2rhodt2tag != Expr::Tag() ),

    didAllocateMatrix_(false),
  
    useRefPressure_( useRefPressure ),
    refPressureValue_( refPressureValue ),
    refPressureLocation_( refPressureLocation ),
  
    use3DLaplacian_( use3DLaplacian ),
  
    solverParams_( solverParams ),
    solver_( solver ),

    // note that this does not provide any ghost entries in the matrix...
    matrixLabel_  ( Uintah::VarLabel::create( "pressure_matrix", Uintah::CCVariable<Uintah::Stencil4>::getTypeDescription() ) ),
    pressureLabel_( Uintah::VarLabel::create( pressureName,
                                              Wasatch::get_uintah_field_type_descriptor<SVolField>(),
                                              Wasatch::get_uintah_ghost_descriptor<SVolField>() ) ),
    prhsLabel_    ( Uintah::VarLabel::create( pressureRHSName,
                                              Wasatch::get_uintah_field_type_descriptor<SVolField>(),
                                              Wasatch::get_uintah_ghost_descriptor<SVolField>() ) )
{}

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
                           const int RKStage )
{
  // TODO: investigate why projection only works for the first RK stage when running
  // in parallel (specifically hypre)
  solver_.scheduleSolve( level, sched, materials, matrixLabel_,
                        Uintah::Task::NewDW, pressureLabel_, true, prhsLabel_, Uintah::Task::NewDW, pressureLabel_, Uintah::Task::OldDW, &solverParams_ );

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
                               const int RKStage )
{
  if( RKStage == 1 ) task.computes( matrixLabel_, patches, Uintah::Task::ThisLevel, materials, Uintah::Task::NormalDomain );
  else               task.modifies( matrixLabel_, patches, Uintah::Task::ThisLevel, materials, Uintah::Task::NormalDomain );
}

//--------------------------------------------------------------------

void
Pressure::bind_uintah_vars( Uintah::DataWarehouse* const dw,
                           const Uintah::Patch* const patch,
                           const int material,
                           const int RKStage )
{
  materialID_ = material;
  if (didAllocateMatrix_) {
    //std::cout << "RKStage "<< RKStage << "did allocate matrix \n";
    // Todd: instead of checking for allocation - check for new timestep or some other ingenious solution
    // check for transferfrom - transfer matrix from old to new DW
    if (RKStage==1 ) dw->put( matrix_, matrixLabel_, materialID_, patch );
    else   dw->getModifiable( matrix_, matrixLabel_, materialID_, patch );
    //setup_matrix(patch);
  } else {
    dw->allocateAndPut( matrix_, matrixLabel_, materialID_, patch );
    setup_matrix();
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
  exprDeps.requires_expression( dilatationt_ );
  exprDeps.requires_expression( timestept_ );
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
  dilatation_ = &svfm.field_ref( dilatationt_ );
  
  const Expr::FieldManager<double>& doublefm = fml.field_manager<double>();
  timestep_ = &doublefm.field_ref( timestept_ );  
}

//--------------------------------------------------------------------

void
Pressure::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  interpX_ = opDB.retrieve_operator<FxInterp>();
  interpY_ = opDB.retrieve_operator<FyInterp>();
  interpZ_ = opDB.retrieve_operator<FzInterp>();

  divXOp_ = opDB.retrieve_operator<DivX>();
  divYOp_ = opDB.retrieve_operator<DivY>();
  divZOp_ = opDB.retrieve_operator<DivZ>();

}

//--------------------------------------------------------------------

void
Pressure::setup_matrix()
{
  // construct the coefficient matrix: \nabla^2
  // We should probably move the matrix construction to the evaluate() method.
  // need a way to get access to the patch so that we can loop over the cells.
  // p is current cell
  double p = 0.0;
  // n: north, s: south, e: east, w: west, t: top, b: bottom coefficient
  double w = 0.0, s = 0.0, b = 0.0;
  
  const SCIRun::IntVector l    = patch_->getCellLowIndex();
  const SCIRun::IntVector h    = patch_->getCellHighIndex();
  const Uintah::Vector spacing = patch_->dCell();

  if ( doX_ || use3DLaplacian_ ) {
    const double dx2 = spacing[0]*spacing[0];
    w = 1.0/dx2;
    p -= 2.0/dx2;
  }
  if ( doY_ || use3DLaplacian_ ) {
    const double dy2 = spacing[1]*spacing[1];
    s = 1.0/dy2;
    p -= 2.0/dy2;
  }
  if ( doZ_ || use3DLaplacian_ ) {
    const double dz2 = spacing[2]*spacing[2];
    b = 1.0/dz2;
    p -= 2.0/dz2;
  }

  for(Uintah::CellIterator iter(patch_->getCellIterator()); !iter.done(); iter++){
    // NOTE: for the conjugate gradient solver in Hypre, we must pass a positive
    // definite matrix. For the Laplacian on a structured grid, the matrix A corresponding
    // to the Laplacian operator is not positive definite - but "- A" is. Hence,
    // we multiply all coefficients by -1.
    IntVector iCell = *iter;
    Uintah::Stencil4&  coefs = matrix_[iCell];
    coefs.w = -w;
    coefs.s = -s;
    coefs.b = -b;
    coefs.p = -p;
  }
  // When boundary conditions are present, modify the pressure matrix coefficients at the boundary
  update_pressure_matrix((this->names())[0], matrix_, patch_, materialID_);
  if (useRefPressure_) set_ref_pressure_coefs(matrix_, patch_, refPressureLocation_);
}

//--------------------------------------------------------------------

void
Pressure::evaluate()
{
  using namespace SpatialOps;

  typedef std::vector<SVolField*> SVolFieldVec;
  SVolFieldVec& results = this->get_value_vec();

  // jcs: can we do the linear solve in place? We probably can. If so,
  // we would only need one field, not two...
  SVolField& pressure = *results[0];
  pressure <<= 0.0;
  
  SVolField& rhs      = *results[1];
  rhs <<= 0.0;
  
  // start by subtracting the dilatation from the previous timestep or integrator
  // stage. This is needed to account for any non-divergence free initial conditions
  rhs <<= - *dilatation_/ *timestep_;
		
  //
  namespace SS = SpatialOps::structured;  
  SpatialOps::SpatFldPtr<SS::SSurfXField> tmpx;
  SpatialOps::SpatFldPtr<SS::SSurfYField> tmpy;
  SpatialOps::SpatFldPtr<SS::SSurfZField> tmpz;
  
  if (doX_) {
    const SS::MemoryWindow& wx = fx_->window_with_ghost();
    tmpx  = SpatialOps::SpatialFieldStore<SS::SSurfXField >::self().get( wx );
  }

  if (doY_) {
    const SS::MemoryWindow& wy = fy_->window_with_ghost();
    tmpy  = SpatialOps::SpatialFieldStore<SS::SSurfYField >::self().get( wy );
  }

  if (doZ_) {
    const SS::MemoryWindow& wz = fz_->window_with_ghost();
    tmpz  = SpatialOps::SpatialFieldStore<SS::SSurfZField >::self().get( wz );
  }

  SpatialOps::SpatFldPtr<SVolField> tmp = SpatialOps::SpatialFieldStore<SVolField>::self().get( rhs );
  //set_pressure_bc(this->name(), pressure, rhs, patch);
  //___________________________________________________
  // calculate the RHS field for the poisson solve.
  // Note that this is "automagically" plugged into
  // the solver in the "schedule_solver" method.
  // NOTE THE NEGATIVE SIGNS! SINCE WE ARE USING CG SOLVER, WE MUST SOLVE FOR
  // - Laplacian(p) = - p_rhs
  if( doX_ ){
    interpX_->apply_to_field( *fx_, *tmpx );
    divXOp_ ->apply_to_field( *tmpx, *tmp );
    rhs <<= rhs - *tmp;
  }

  if( doY_ ){
    interpY_->apply_to_field( *fy_, *tmpy );
    divYOp_ ->apply_to_field( *tmpy, *tmp );
    rhs <<= rhs - *tmp;
  }

  if( doZ_ ){
    interpZ_->apply_to_field( *fz_, *tmpz );
    divZOp_ ->apply_to_field( *tmpz, *tmp );
    rhs <<= rhs - *tmp;
  }

  if( doDens_ ){
    rhs <<= rhs - *d2rhodt2_;
  }
  
  if (useRefPressure_) set_ref_pressure_rhs( rhs, patch_, refPressureValue_, refPressureLocation_ );
  //
  // fix pressure rhs and modify pressure matrix
  update_pressure_rhs((this->names())[0],matrix_, pressure, rhs, patch_, materialID_);
  //set_pressure_rhs((this->names())[0],matrix_, rhs, patch_);
}

//--------------------------------------------------------------------

Pressure::Builder::Builder( const Expr::TagList& result,
                            const Expr::Tag& fxtag,
                            const Expr::Tag& fytag,
                            const Expr::Tag& fztag,
                            const Expr::Tag& dilatationtag,                           
                            const Expr::Tag& d2rhodt2tag,
                            const Expr::Tag& timesteptag,
                            const bool       userefpressure,
                            const double     refPressureValue,
                            const SCIRun::IntVector refPressureLocation,
                            const bool       use3dlaplacian,                           
                            const Uintah::SolverParameters& sparams,
                            Uintah::SolverInterface& solver )
 : ExpressionBuilder(result),
   fxt_( fxtag ),
   fyt_( fytag ),
   fzt_( fztag ),
   dilatationt_ ( dilatationtag ),
   d2rhodt2t_( d2rhodt2tag ),
   timestept_( timesteptag ),
   userefpressure_( userefpressure ),
   refpressurevalue_( refPressureValue ),
   refpressurelocation_( refPressureLocation ),
   use3dlaplacian_( use3dlaplacian ),
   sparams_( sparams ),
   solver_( solver )
{}

//--------------------------------------------------------------------

Expr::ExpressionBase*
Pressure::Builder::build() const
{
  const Expr::TagList& ptags = get_computed_field_tags();
  return new Pressure( ptags[0].name(), ptags[1].name(), fxt_, fyt_, fzt_, dilatationt_, d2rhodt2t_, timestept_, userefpressure_, refpressurevalue_, refpressurelocation_, use3dlaplacian_, sparams_, solver_ );
}

} // namespace Wasatch
