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

#include "PoissonExpression.h"

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
    
  PoissonExpression::PoissonExpression( const Expr::Tag& phiTag,
                                        const Expr::Tag& phiRHSLocalTag,
                                        const Expr::Tag& phiRHSTag,
                                        const bool       useRefPhi,
                                        const double     refPhiValue,
                                        const SCIRun::IntVector refPhiLocation,
                                        const bool       use3DLaplacian,
                                        const Uintah::SolverParameters& solverParams,
                                        Uintah::SolverInterface& solver )
  : Expr::Expression<SVolField>(),
    phit_   (phiTag),
    phirhslocalt_(phiRHSLocalTag),
    phirhst_( phiRHSTag ),
    doX_( true ),
    doY_( true ),
    doZ_( true ),
    
    didAllocateMatrix_(false),
    
    materialID_(0),
    rkStage_(1),
    
    useRefPhi_( useRefPhi ),
    refPhiValue_( refPhiValue ),
    refPhiLocation_( refPhiLocation ),
    
    use3DLaplacian_( use3DLaplacian ),
    
    solverParams_( solverParams ),
    solver_( solver ),
    
    // note that this does not provide any ghost entries in the matrix...
    matrixLabel_( Uintah::VarLabel::create( phit_.name() + "_matrix", Uintah::CCVariable<Uintah::Stencil4>::getTypeDescription() ) ),
    phiLabel_( Uintah::VarLabel::create( phit_.name(),
                                         Wasatch::get_uintah_field_type_descriptor<SVolField>(),
                                         Wasatch::get_uintah_ghost_descriptor<SVolField>() ) ),
    phirhsLabel_( Uintah::VarLabel::create( phirhslocalt_.name(),
                                            Wasatch::get_uintah_field_type_descriptor<SVolField>(),
                                            Wasatch::get_uintah_ghost_descriptor<SVolField>() ) )
  {}

  //--------------------------------------------------------------------
  
  PoissonExpression::~PoissonExpression()
  {
    Uintah::VarLabel::destroy( phiLabel_      );
    Uintah::VarLabel::destroy( phirhsLabel_   );
    Uintah::VarLabel::destroy( matrixLabel_   );
  }
  
  //--------------------------------------------------------------------
  
  void
  PoissonExpression::schedule_solver( const Uintah::LevelP& level,
                                      Uintah::SchedulerP sched,
                                      const Uintah::MaterialSet* const materials,
                                      const int RKStage, const bool isDoingInitialization )
  {
    solver_.scheduleSolve( level, sched, materials, matrixLabel_, Uintah::Task::NewDW,
                           phiLabel_, true,
                           phirhsLabel_, Uintah::Task::NewDW,
                           isDoingInitialization ? 0 : phiLabel_, RKStage == 1 ? Uintah::Task::OldDW : Uintah::Task::NewDW,
                           &solverParams_ );
  }
  
  //--------------------------------------------------------------------
  
  void
  PoissonExpression::schedule_set_poisson_bcs( const Uintah::LevelP& level,
                                               Uintah::SchedulerP sched,
                                               const Uintah::MaterialSet* const materials,
                                               const int RKStage )
  {
    // hack in a task to apply boundary condition on the poisson variable after the linear solve
    Uintah::Task* task = scinew Uintah::Task("Poisson Equation: process poisson bcs", this,
                                             &PoissonExpression::process_bcs);
    const Uintah::Ghost::GhostType gt = get_uintah_ghost_type<SVolField>();
    const int ng = get_n_ghost<SVolField>();
    task->requires(Uintah::Task::NewDW,phiLabel_, gt, ng);
    //task->modifies(phiLabel_);
    Uintah::LoadBalancer* lb = sched->getLoadBalancer();
    sched->addTask(task, lb->getPerProcessorPatchSet(level), materials);
  }
  
  //--------------------------------------------------------------------
  
  void
  PoissonExpression::declare_uintah_vars( Uintah::Task& task,
                                          const Uintah::PatchSubset* const patches,
                                          const Uintah::MaterialSubset* const materials,
                                          const int RKStage )
  {
    if( RKStage == 1 ) task.computes( matrixLabel_, patches, Uintah::Task::ThisLevel, materials, Uintah::Task::NormalDomain );
    else               task.modifies( matrixLabel_, patches, Uintah::Task::ThisLevel, materials, Uintah::Task::NormalDomain );
  }
  
  //--------------------------------------------------------------------
  
  void
  PoissonExpression::bind_uintah_vars( Uintah::DataWarehouse* const dw,
                                       const Uintah::Patch* const patch,
                                       const int material,
                                       const int RKStage )
  {
    materialID_ = material;
    if (didAllocateMatrix_) {
      if (RKStage==1 ) dw->put( matrix_, matrixLabel_, materialID_, patch );
      else             dw->getModifiable( matrix_, matrixLabel_, materialID_, patch );
    } else {
      dw->allocateAndPut( matrix_, matrixLabel_, materialID_, patch );
      setup_matrix();
      didAllocateMatrix_=true;
    }
  }
  
  //--------------------------------------------------------------------
  
  void
  PoissonExpression::advertise_dependents( Expr::ExprDeps& exprDeps )
  {
    if(phirhst_ != Expr::Tag()) exprDeps.requires_expression( phirhst_ );
  }
  
  //--------------------------------------------------------------------
  
  void
  PoissonExpression::bind_fields( const Expr::FieldManagerList& fml )
  {
    const Expr::FieldManager<SVolField>& svfm = fml.field_manager<SVolField>();
    if(phirhst_ != Expr::Tag()) phirhs_ = &svfm.field_ref( phirhst_ );
  }
  
  //--------------------------------------------------------------------
    
  void
  PoissonExpression::setup_matrix()
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
    
    // When boundary conditions are present, modify the coefficient matrix coefficients at the boundary
    if ( patch_->hasBoundaryFaces() )update_poisson_matrix((this->names())[0], matrix_, patch_, materialID_);

    // if the user specified a reference value, then modify the appropriate matrix coefficients
    if ( useRefPhi_ ) set_ref_poisson_coefs(matrix_, patch_, refPhiLocation_ );
  }
  
  //--------------------------------------------------------------------
  
  void
  PoissonExpression::evaluate()
  {
    using namespace SpatialOps;
    namespace SS = SpatialOps::structured;

    typedef std::vector<SVolField*> SVolFieldVec;
    SVolFieldVec& results = this->get_value_vec();

    SVolField& phi = *results[0];
    if (rkStage_ == 1) phi <<= 0.0;

    SVolField& rhs = *results[1];
    rhs <<= 0.0;
    rhs <<= - *phirhs_;    
    
    // update poisson rhs for reference value
    if (useRefPhi_) set_ref_poisson_rhs( rhs, patch_, refPhiValue_, refPhiLocation_ );
    
    if(patch_->hasBoundaryFaces()) update_poisson_rhs(phit_, matrix_, phi, rhs, patch_, materialID_);
  }
  
  //--------------------------------------------------------------------
  
  void
  PoissonExpression::process_bcs ( const Uintah::ProcessorGroup* const pg,
                                   const Uintah::PatchSubset* const patches,
                                   const Uintah::MaterialSubset* const materials,
                                   Uintah::DataWarehouse* const oldDW,
                                   Uintah::DataWarehouse* const newDW )
  {
    using namespace SpatialOps;
    typedef SelectUintahFieldType<SVolField>::const_type UintahField;
    UintahField poissonField_;
    
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
        if ( patch->hasBoundaryFaces() ) {
          newDW->get( poissonField_, phiLabel_, material, patch, gt, ng);
          SVolField* const phi = wrap_uintah_field_as_spatialops<SVolField>(poissonField_,patch);
          process_poisson_bcs(phit_, *phi, patch, material);
          delete phi;
        }
      }
    }
  }
  
  //--------------------------------------------------------------------
  
  PoissonExpression::Builder::Builder( const Expr::TagList& results,
                                       const Expr::Tag& phiRHSTag,
                                       const bool       useRefPhi,
                                       const double     refPhiValue,
                                       const SCIRun::IntVector refPhiLocation,
                                       const bool       use3dlaplacian,
                                       const Uintah::SolverParameters& sparams,
                                       Uintah::SolverInterface& solver )
  : ExpressionBuilder(results),
  phirhst_( phiRHSTag ),
  userefphi_ ( useRefPhi ),
  refphivalue_ ( refPhiValue ),
  refphilocation_ ( refPhiLocation ),
  use3dlaplacian_( use3dlaplacian ),
  sparams_( sparams ),
  solver_( solver )
  {}
  
  //--------------------------------------------------------------------
  
  Expr::ExpressionBase*
  PoissonExpression::Builder::build() const
  {
    const Expr::TagList& phitags = get_computed_field_tags();
    //const Expr::Tag& phitag = get_computed_field_tag();
    return new PoissonExpression( phitags[0], phitags[1], phirhst_, userefphi_,
                        refphivalue_, refphilocation_, use3dlaplacian_,
                        sparams_, solver_ );
  }
  Expr::TagList PoissonExpression::poissonTagList = Expr::TagList();
} // namespace Wasatch
