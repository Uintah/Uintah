/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

#include <CCA/Components/Wasatch/Expressions/Pressure.h>

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/TagNames.h>

//-- Uintah Includes --//
#include <CCA/Ports/SolverInterface.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Material.h>

#include <Core/Grid/Variables/VarTypes.h>  // delt_vartype
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <CCA/Ports/LoadBalancerPort.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>

namespace WasatchCore {

//==================================================================

Pressure::Pressure( const std::string& pressureName,
                    const std::string& pressureRHSName,
                    const Expr::Tag& fxtag,
                    const Expr::Tag& fytag,
                    const Expr::Tag& fztag,
                    const Expr::Tag& pSourceTag,
                    const Expr::Tag& dtTag,
                    const Expr::Tag& volfractag,
                    const Expr::Tag& rhoStarTag,
                    const bool hasMovingGeometry,
                    const bool       useRefPressure,
                    const double     refPressureValue,
                    const Uintah::IntVector refPressureLocation,
                    const bool       use3DLaplacian,
                    const bool       enforceSolvability,
                    const bool isConstDensity,
                    Uintah::SolverParameters& solverParams,
                    Uintah::SolverInterface& solver)
  : Expr::Expression<SVolField>(),
    volFracTag_(volfractag),
    rhoStarTag_(rhoStarTag),
    doX_( fxtag != Expr::Tag() ),
    doY_( fytag != Expr::Tag() ),
    doZ_( fztag != Expr::Tag() ),
    hasIntrusion_(volfractag != Expr::Tag()),

    didAllocateMatrix_(false),
    didMatrixUpdate_(false),
    hasMovingGeometry_(hasMovingGeometry),

    materialID_(0),
    rkStage_(1),

    useRefPressure_( useRefPressure ),
    refPressureValue_( refPressureValue ),
    refPressureLocation_( refPressureLocation ),

    use3DLaplacian_( use3DLaplacian ),
    enforceSolvability_(enforceSolvability),
    isConstDensity_(isConstDensity),

    solverParams_( solverParams ),
    solver_( solver ),

    // note that this does not provide any ghost entries in the matrix...
    matrixLabel_  ( Uintah::VarLabel::create( "pressure_matrix", Uintah::CCVariable<Uintah::Stencil7>::getTypeDescription() ) ),
    pressureLabel_( Uintah::VarLabel::create( pressureName,
                                              WasatchCore::get_uintah_field_type_descriptor<SVolField>() ) ),
    prhsLabel_    ( Uintah::VarLabel::create( pressureRHSName,
                                              WasatchCore::get_uintah_field_type_descriptor<SVolField>() ) )
{
   timestep_ = create_field_request<TimeField>(TagNames::self().timestep);
   t_ = create_field_request<TimeField>(TagNames::self().time);
   if(doX_)  fx_ = create_field_request<XVolField>(fxtag);
   if(doY_)  fy_ = create_field_request<YVolField>(fytag);
   if(doZ_)  fz_ = create_field_request<ZVolField>(fztag);
   pSource_ = create_field_request<SVolField>(pSourceTag);
  
  if (!isConstDensity_) rhoStar_ = create_field_request<SVolField>(rhoStarTag);
  
  if (hasIntrusion_)    volfrac_ = create_field_request<SVolField>(volfractag);
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
                           const int RKStage )
{
  if (enforceSolvability_) {
    solver_.scheduleEnforceSolvability<WasatchCore::SelectUintahFieldType<SVolField>::type >(level, sched, materials, prhsLabel_, RKStage);
  }
  solver_.scheduleSolve( level, sched, materials, matrixLabel_, Uintah::Task::NewDW,
                         pressureLabel_, true,
                         prhsLabel_, Uintah::Task::NewDW,
                         pressureLabel_, RKStage == 1 ? Uintah::Task::OldDW : Uintah::Task::NewDW,
                         &solverParams_, RKStage == 1 ? false:true);
  if(useRefPressure_) {
    solver_.scheduleSetReferenceValue<WasatchCore::SelectUintahFieldType<SVolField>::type >(level, sched, materials, pressureLabel_, RKStage, refPressureLocation_, refPressureValue_);
  }
}

//--------------------------------------------------------------------

void
Pressure::schedule_set_pressure_bcs( const Uintah::LevelP& level,
                          Uintah::SchedulerP sched,
                          const Uintah::MaterialSet* const materials,
                          const int RKStage )
{
  // hack in a task to apply boundary condition on the pressure after the pressure solve
  Uintah::Task* task = scinew Uintah::Task("Pressure: process pressure bcs", this,
                                           &Pressure::process_bcs);
  const Uintah::Ghost::GhostType gt = get_uintah_ghost_type<SVolField>();
  const int ng = get_n_ghost<SVolField>();
  task->requires( Uintah::Task::NewDW,pressureLabel_, gt, ng );
  //task->modifies( pressureLabel_);
  Uintah::LoadBalancerPort * lb = sched->getLoadBalancer();
  sched->addTask( task, lb->getPerProcessorPatchSet( level ), materials );
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
  if ( hasIntrusion_ ) {
    const Uintah::Ghost::GhostType gt = get_uintah_ghost_type<SVolField>();
    const int ng = get_n_ghost<SVolField>();
    task.requires(Uintah::Task::NewDW, Uintah::VarLabel::find(volFracTag_.name()), gt, ng);
  }
}

//--------------------------------------------------------------------

void
Pressure::bind_uintah_vars( Uintah::DataWarehouse* const dw,
                           const Uintah::Patch* const patch,
                           const int material,
                           const int RKStage )
{
  materialID_ = material;
  patch_ = const_cast<Uintah::Patch*> (patch);
  rkStage_ = RKStage;

  if( didAllocateMatrix_ ){
    if (RKStage==1 ) dw->put( matrix_, matrixLabel_, materialID_, patch );
    else             dw->getModifiable( matrix_, matrixLabel_, materialID_, patch );
  }
  else{
    dw->allocateAndPut( matrix_, matrixLabel_, materialID_, patch );
    
    typedef SelectUintahFieldType<SVolField>::const_type ConstUintahField;
    ConstUintahField svolFrac;
    ConstUintahField rhoStar_;
    
    SpatialOps::SpatFldPtr<SVolField> volfrac;
    SpatialOps::SpatFldPtr<SVolField> rhoStar;
    
    const Uintah::Ghost::GhostType gt = get_uintah_ghost_type<SVolField>();
    const int ng = get_n_ghost<SVolField>();
    
    if (!isConstDensity_) {
      dw->get( rhoStar_, Uintah::VarLabel::find(rhoStarTag_.name()), material, patch, gt, ng );
      const AllocInfo ainfo( dw, dw, material, patch, nullptr );
      const SpatialOps::GhostData gd( get_n_ghost<SVolField>() );
      rhoStar = wrap_uintah_field_as_spatialops<SVolField>( rhoStar_, ainfo, gd );
    }
    
    if( volFracTag_ != Expr::Tag() ){
      const Uintah::Ghost::GhostType gt = get_uintah_ghost_type<SVolField>();
      const int ng = get_n_ghost<SVolField>();
      dw->get( svolFrac, Uintah::VarLabel::find(volFracTag_.name()), material, patch, gt, ng );
      const AllocInfo ainfo( dw, dw, material, patch, nullptr );
      const SpatialOps::GhostData gd( get_n_ghost<SVolField>() );
      volfrac = wrap_uintah_field_as_spatialops<SVolField>( svolFrac, ainfo, gd );
    }
    
    if(isConstDensity_) setup_matrix( &*volfrac );
  }
  
  didAllocateMatrix_=true;
}

//--------------------------------------------------------------------

void
Pressure::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  interpX_ = opDB.retrieve_operator<FxInterp>();
  interpY_ = opDB.retrieve_operator<FyInterp>();
  interpZ_ = opDB.retrieve_operator<FzInterp>();

  s2XInterOp_ = opDB.retrieve_operator<S2XInterpT>();
  s2YInterOp_ = opDB.retrieve_operator<S2YInterpT>();
  s2ZInterOp_ = opDB.retrieve_operator<S2ZInterpT>();

  divXOp_ = opDB.retrieve_operator<DivX>();
  divYOp_ = opDB.retrieve_operator<DivY>();
  divZOp_ = opDB.retrieve_operator<DivZ>();  
}

//--------------------------------------------------------------------

void
Pressure::setup_matrix( const SVolField* const volfrac )
{
  // construct the coefficient matrix: \nabla^2
  // We should probably move the matrix construction to the evaluate() method.
  // need a way to get access to the patch so that we can loop over the cells.
  // p is current cell
  double p = 0.0;
  // n: north, s: south, e: east, w: west, t: top, b: bottom coefficient
  double w = 0.0, s = 0.0, b = 0.0;
  
  const Uintah::IntVector l    = patch_->getCellLowIndex();
  const Uintah::IntVector h    = patch_->getCellHighIndex();
  const Uintah::Vector spacing = patch_->dCell();
  
  if ( doX_ || use3DLaplacian_ ) {
    const double dx2 = spacing[0]*spacing[0];
    w  = 1.0/dx2;
    p -= 2.0/dx2;
  }
  if ( doY_ || use3DLaplacian_ ) {
    const double dy2 = spacing[1]*spacing[1];
    s  = 1.0/dy2;
    p -= 2.0/dy2;
  }
  if ( doZ_ || use3DLaplacian_ ) {
    const double dz2 = spacing[2]*spacing[2];
    b  = 1.0/dz2;
    p -= 2.0/dz2;
  }
  
  for( Uintah::CellIterator iter(patch_->getCellIterator()); !iter.done(); iter++ ){
    // NOTE: for the conjugate gradient solver in Hypre, we must pass a positive
    // definite matrix. For the Laplacian on a structured grid, the matrix A corresponding
    // to the Laplacian operator is not positive definite - but "- A" is. Hence,
    // we multiply all coefficients by -1.
    Uintah::IntVector iCell = *iter;
    Uintah::Stencil7&  coefs = matrix_[iCell];
    coefs.w = -w;
    coefs.e = -w;
    coefs.s = -s;
    coefs.n = -s;
    coefs.b = -b;
    coefs.t = -b;
    coefs.p = -p;
  }

  // update the coefficient matrix with intrusion information
  if( hasIntrusion_ && volfrac )
    process_embedded_boundaries( *volfrac );
  
  // When boundary conditions are present, modify the pressure matrix coefficients at the boundary
  if( patch_->hasBoundaryFaces() && bcHelper_ )
    bcHelper_->update_pressure_matrix( matrix_, volfrac, patch_ );
}
  
//--------------------------------------------------------------------

void
Pressure::setup_matrix( const SVolField* const rhoStar,
                        const SVolField* const volfrac )
{
  // construct the coefficient matrix: \nabla^2
  // We should probably move the matrix construction to the evaluate() method.
  // need a way to get access to the patch so that we can loop over the cells.
  // p is current cell
  // n: north, s: south, e: east, w: west, t: top, b: bottom coefficient
  double w = 0.0, s = 0.0, b = 0.0;
  using namespace SpatialOps;
  const Uintah::IntVector l    = patch_->getCellLowIndex();
  const Uintah::IntVector h    = patch_->getCellHighIndex();
  const Uintah::Vector spacing = patch_->dCell();
  
  if ( doX_ || use3DLaplacian_ ) {
    const double dx2 = spacing[0]*spacing[0];
    w  = 1.0/dx2;
  }
  if ( doY_ || use3DLaplacian_ ) {
    const double dy2 = spacing[1]*spacing[1];
    s  = 1.0/dy2;
  }
  if ( doZ_ || use3DLaplacian_ ) {
    const double dz2 = spacing[2]*spacing[2];
    b  = 1.0/dz2;
  }

  const SVolField& r = *rhoStar;

  const int ng = get_n_ghost<SVolField>();
  const Uintah::IntVector patchCellOffset = patch_->getExtraCellLowIndex(ng);
  for( Uintah::CellIterator iter(patch_->getCellIterator()); !iter.done(); iter++ ){
    // NOTE: for the conjugate gradient solver in Hypre, we must pass a positive
    // definite matrix. For the Laplacian on a structured grid, the matrix A corresponding
    // to the Laplacian operator is not positive definite - but "- A" is. Hence,
    // we multiply all coefficients by -1.
    Uintah::IntVector iCell = *iter;
    Uintah::Stencil7&  coefs = matrix_[iCell];

    Uintah::IntVector iCellOffset = iCell - patchCellOffset;
    
    // interior
    const IntVec intCellIJK( iCellOffset[0],
                            iCellOffset[1],
                            iCellOffset[2] );

    
    const IntVec eIJK( iCellOffset[0] + 1, iCellOffset[1],     iCellOffset[2]     ); // east
    const IntVec wIJK( iCellOffset[0] - 1, iCellOffset[1],     iCellOffset[2]     ); // west
    const IntVec nIJK( iCellOffset[0],     iCellOffset[1] + 1, iCellOffset[2]     ); // north
    const IntVec sIJK( iCellOffset[0],     iCellOffset[1] - 1, iCellOffset[2]     ); // south
    const IntVec tIJK( iCellOffset[0],     iCellOffset[1],     iCellOffset[2] + 1 ); // top
    const IntVec bIJK( iCellOffset[0],     iCellOffset[1],     iCellOffset[2] - 1 ); // bottom
    
    
    const double rP  = r( intCellIJK );
    const double rE  = r( eIJK );
    const double rW  = r( wIJK );
    const double rN  = r( nIJK );
    const double rS  = r( sIJK );
    const double rT  = r( tIJK );
    const double rB  = r( bIJK );
    
    // here we are computing: div(1/r * grad(p) )
    // r is the density. r should be first interpolated to the faces where grad(p) lives
    // at an arbitrary face between cells 0 and 1: r_face = (r1 + r0)/2
    // hence, 1/r_face = 2/(r0 + r1)
    coefs.w = - (2.0/(rP + rW))*w;
    coefs.e = - (2.0/(rP + rE))*w;

    coefs.n = - (2.0/(rP + rN))*s;
    coefs.s = - (2.0/(rP + rS))*s;

    coefs.t = - (2.0/(rP + rT))*b;
    coefs.b = - (2.0/(rP + rB))*b;
    
//    coefs.p = (fE + 2.0*fP + fW)*0.5*w + (fN + 2.0*fP + fS)*0.5*s + (fT + 2.0*fP + fB)*0.5*b;
    coefs.p = -(coefs.w + coefs.e + coefs.n + coefs.s + coefs.t + coefs.b);

  }

  // update the coefficient matrix with intrusion information
  if( hasIntrusion_ && volfrac )
    process_embedded_boundaries( *volfrac );

  // When boundary conditions are present, modify the pressure matrix coefficients at the boundary
  if( patch_->hasBoundaryFaces() && bcHelper_ )
    bcHelper_->update_pressure_matrix( matrix_, volfrac, patch_ );
}

//--------------------------------------------------------------------

void
Pressure::evaluate()
{
  using namespace SpatialOps;

  typedef typename Expr::Expression<SVolField>::ValVec SVolFieldVec;
  SVolFieldVec& results = this->get_value_vec();

  // jcs: can we do the linear solve in place? We probably can. If so,
  // we would only need one field, not two...
  SVolField& pressure = *results[0];
  // when we are at the 2nd or 3rd RK stages, do NOT initialize the pressure
  // in the new DW to zero because we're using that as our initial guess.
  // This will reduce the pressure solve iteration count.
  if( rkStage_ == 1 ) pressure <<= 0.0;

  SVolField& rhs = *results[1];

  const TimeField& tstep = timestep_->field_ref();
  std::ostringstream strs;
  strs << "_timestep_"<< (int) tstep[0] << "_rkstage_"<< rkStage_ << "_patch";

  solverParams_.setOutputFileName( "_WASATCH" + strs.str() );

  // NOTE THE NEGATIVE SIGN! SINCE WE ARE USING CG SOLVER, WE MUST SOLVE FOR
  // - Laplacian(p) = - p_rhs
  const SVolField& pSrc = pSource_->field_ref();
  rhs <<= - pSrc;
  
  //___________________________________________________
  // calculate the RHS field for the poisson solve.
  // Note that this is "automagically" plugged into
  // the solver in the "schedule_solver" method.
  // NOTE THE NEGATIVE SIGNS! SINCE WE ARE USING CG SOLVER, WE MUST SOLVE FOR
  // - Laplacian(p) = - p_rhs
  if (!isConstDensity_) {
    const SVolField& rhoStar = rhoStar_->field_ref();
    if( doX_ ) rhs <<= rhs - (*divXOp_)((*interpX_)(fx_->field_ref() / (*s2XInterOp_)(rhoStar) ));
    if( doY_ ) rhs <<= rhs - (*divYOp_)((*interpY_)(fy_->field_ref() / (*s2YInterOp_)(rhoStar) ));
    if( doZ_ ) rhs <<= rhs - (*divZOp_)((*interpZ_)(fz_->field_ref() / (*s2ZInterOp_)(rhoStar) ));
  } else {
    if( doX_ ) rhs <<= rhs - (*divXOp_)((*interpX_)(fx_->field_ref() ));
    if( doY_ ) rhs <<= rhs - (*divYOp_)((*interpY_)(fy_->field_ref() ));
    if( doZ_ ) rhs <<= rhs - (*divZOp_)((*interpZ_)(fz_->field_ref() ));
  }
  if( hasIntrusion_) rhs <<= rhs * volfrac_->field_ref();

  // update pressure rhs for any BCs
  if( patch_->hasBoundaryFaces() )
    bcHelper_->update_pressure_rhs(rhs, patch_); // this will update the rhs with relevant boundary conditions.

  if (!isConstDensity_) {
    const SVolField& rhoStar = rhoStar_->field_ref();
    if (hasMovingGeometry_ && hasIntrusion_) {
      setup_matrix( &rhoStar, &(volfrac_->field_ref()) );
    } else {
      setup_matrix( &rhoStar, 0 ); // update coefficient matrix to account for varying density without an intrusion
    }
  } else {
    // if we have moving geometry, then we need to update the coefficient matrix
    if( hasMovingGeometry_ && hasIntrusion_ ) {
      setup_matrix( &(volfrac_->field_ref()) );
    }
  }  
}

//--------------------------------------------------------------------

void Pressure::process_embedded_boundaries( const SVolField& volfrac )
{
  using SpatialOps::IntVec;

  // cell offset used to calculate local cell index with respect to patch.
  const int ng = get_n_ghost<SVolField>();
  const Uintah::IntVector patchCellOffset = patch_->getExtraCellLowIndex(ng);

  if( !didMatrixUpdate_ || hasMovingGeometry_ ){
    
    // didMatrixUpdate_: boolean that tracks whether we have updated the
    // pressure coef matrix or not when embedded geometries are present
    didMatrixUpdate_ = true;
    
    for(Uintah::CellIterator iter(patch_->getCellIterator()); !iter.done(); iter++){
      Uintah::IntVector iCell = *iter;
      Uintah::Stencil7&  coefs = matrix_[iCell];
      
      const Uintah::IntVector iCellOffset = iCell - patchCellOffset;
      
      // interior
      const IntVec intCellIJK( iCellOffset[0],
                               iCellOffset[1],
                               iCellOffset[2] );
      
      const double volFrac = volfrac(intCellIJK);
      
      // we are inside an embedded geometry, set pressure to zero.
      if( volFrac < 1.0 ) {
        coefs.e = 0.0;
        coefs.w = 0.0;
        coefs.n = 0.0;        
        coefs.s = 0.0;
        coefs.t = 0.0;
        coefs.b = 0.0;
        coefs.p = 1.0;
      }
      
      if( volFrac > 0.0 ){

        const IntVec eIJK( iCellOffset[0] + 1, iCellOffset[1],     iCellOffset[2]     ); // east
        const IntVec wIJK( iCellOffset[0] - 1, iCellOffset[1],     iCellOffset[2]     ); // west
        const IntVec nIJK( iCellOffset[0],     iCellOffset[1] + 1, iCellOffset[2]     ); // north
        const IntVec sIJK( iCellOffset[0],     iCellOffset[1] - 1, iCellOffset[2]     ); // south
        const IntVec tIJK( iCellOffset[0],     iCellOffset[1],     iCellOffset[2] + 1 ); // top
        const IntVec bIJK( iCellOffset[0],     iCellOffset[1],     iCellOffset[2] - 1 ); // bottom

        const double volFracEast  = volfrac( eIJK );
        const double volFracWest  = volfrac( wIJK );
        const double volFracNorth = volfrac( nIJK );
        const double volFracSouth = volfrac( sIJK );
        const double volFracTop   = volfrac( tIJK );
        const double volFracBot   = volfrac( bIJK );

        // neighbors are embedded boundaries
        if( doX_ && volFracEast < 1.0 ){
          coefs.p += coefs.e;
          coefs.e  = 0.0;
        }
        if( doY_ && volFracNorth < 1.0 ){
          coefs.p += coefs.n;
          coefs.n  = 0.0;
        }
        if( doZ_ && volFracTop < 1.0 ){
          coefs.p += coefs.t;
          coefs.t  = 0.0;
        }
        if( doX_ && volFracWest < 1.0 ){
          coefs.p += coefs.w;
          coefs.w  = 0.0;
        }
        if( doY_ && volFracSouth < 1.0 ){
          coefs.p += coefs.s;
          coefs.s = 0.0;
        }
        if( doZ_ && volFracBot < 1.0 ){
          coefs.p += coefs.b;
          coefs.b = 0.0;
        }
      }
    }
  }  
}

//--------------------------------------------------------------------
  
void
Pressure::process_bcs ( const Uintah::ProcessorGroup* const pg,
                        const Uintah::PatchSubset* const patches,
                        const Uintah::MaterialSubset* const materials,
                        Uintah::DataWarehouse* const oldDW,
                        Uintah::DataWarehouse* const newDW )
{
  using namespace SpatialOps;
  typedef SelectUintahFieldType<SVolField>::const_type UintahField;
  typedef SpatFldPtr<SVolField> SVolFieldPtr;
  UintahField pressureField_;

  const Uintah::Ghost::GhostType gt = get_uintah_ghost_type<SVolField>();
  const int ng = get_n_ghost<SVolField>();

  //__________________
  // loop over materials
  for( int im=0; im<materials->size(); ++im ){

    const int material = materials->get(im);

    //____________________
    // loop over patches
    for( int ip=0; ip<patches->size(); ++ip ){
      const Uintah::Patch* const patch = patches->get(ip);
      if( patch->hasBoundaryFaces() && bcHelper_  ){
        newDW->get( pressureField_, pressureLabel_, material, patch, gt, ng);
        const AllocInfo ainfo( oldDW, newDW, im, patch, pg );
        const SpatialOps::GhostData gd( get_n_ghost<SVolField>() );
        SVolFieldPtr pressure = wrap_uintah_field_as_spatialops<SVolField>( pressureField_, ainfo, gd );
        bcHelper_->apply_pressure_bc(*pressure,patch);
      }
    }
  }
}

//--------------------------------------------------------------------

Pressure::Builder::Builder( const Expr::TagList& result,
                            const Expr::Tag& fxtag,
                            const Expr::Tag& fytag,
                            const Expr::Tag& fztag,
                            const Expr::Tag& pSourceTag,
                            const Expr::Tag& dtTag,
                            const Expr::Tag& volfractag,
                            const Expr::Tag& rhoStarTag,
                            const bool hasMovingGeometry,
                            const bool       userefpressure,
                            const double     refPressureValue,
                            const Uintah::IntVector refPressureLocation,
                            const bool       use3dlaplacian,
                            const bool       enforceSolvability,
                            const bool       isConstDensity,
                            Uintah::SolverParameters& sparams,
                            Uintah::SolverInterface& solver )
 : ExpressionBuilder(result),
   fxt_( fxtag ),
   fyt_( fytag ),
   fzt_( fztag ),
   psrct_( pSourceTag ),
   dtt_( dtTag ),
   volfract_ ( volfractag  ),
   rhoStarTag_ ( rhoStarTag ),
   hasMovingGeometry_(hasMovingGeometry),
   userefpressure_( userefpressure ),
   refpressurevalue_( refPressureValue ),
   refpressurelocation_( refPressureLocation ),
   use3dlaplacian_( use3dlaplacian ),
   enforceSolvability_(enforceSolvability),
   isConstDensity_(isConstDensity),
   sparams_( sparams ),
   solver_( solver )
{}

//--------------------------------------------------------------------

Expr::ExpressionBase*
Pressure::Builder::build() const
{
  const Expr::TagList& ptags = get_tags();
  return new Pressure( ptags[0].name(), ptags[1].name(), fxt_, fyt_, fzt_,
                       psrct_, dtt_,volfract_, rhoStarTag_, hasMovingGeometry_, userefpressure_,
                       refpressurevalue_, refpressurelocation_, use3dlaplacian_,
                       enforceSolvability_, isConstDensity_, sparams_, solver_ );
}

} // namespace WasatchCore
