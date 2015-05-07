/*
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
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

#include <CCA/Components/Wasatch/Expressions/RadiationSource.h>

//-- Wasatch Includes --//
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/BCHelperTools.h>
#include <CCA/Components/Wasatch/TagNames.h>

//-- Uintah Includes --//
#include <CCA/Ports/SolverInterface.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Material.h>

#include <Core/Disclosure/TypeDescription.h>
#include <Core/Grid/Variables/VarTypes.h>  // delt_vartype
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <CCA/Ports/LoadBalancer.h>

//-- SpatialOps Includes --//
#include <spatialops/OperatorDatabase.h>

namespace Wasatch {
  
  //==================================================================
  
  Expr::Tag RadiationSource_tag()
  {
    return Expr::Tag( "RadiationSource", Expr::STATE_NONE );
  }
  
  //==================================================================
  
  RadiationSource::RadiationSource( const std::string& radiationSourceName,
                                   const Expr::Tag& temperatureTag,
                                   const Expr::Tag& absorptionTag,
                                   const Expr::Tag& celltypeTag,
                                   Uintah::Ray* rmcrt,
                                   const Uintah::ProblemSpecP& radiationSpec,
                                   Uintah::SimulationStateP sharedState,
                                   Uintah::GridP grid)
  : Expr::Expression<SVolField>(),
  
  // note that this does not provide any ghost entries in the matrix...
  temperatureLabel_( Uintah::VarLabel::create( temperatureTag.name(),
                                              Wasatch::get_uintah_field_type_descriptor<SVolField>() ) ),
  absorptionLabel_ ( Uintah::VarLabel::create( absorptionTag.name(),
                                              Wasatch::get_uintah_field_type_descriptor<SVolField>() ) ),
  celltypeLabel_   ( Uintah::VarLabel::create( celltypeTag.name(),
                                              Wasatch::get_uintah_field_type_descriptor<int>() ) ),
  divqLabel_       ( Uintah::VarLabel::create( radiationSourceName,
                                              Wasatch::get_uintah_field_type_descriptor<SVolField>() ) ),
  rmcrt_(rmcrt)
  {
     temperature_ = create_field_request<SVolField>(temperatureTag);
     absCoef_ = create_field_request<SVolField>(absorptionTag);
    // cellType_ = create_field_request<FieldT>(celltypeTag );        
    
    rmcrt_->registerVarLabels( 0,
                              absorptionLabel_,
                              temperatureLabel_,
                              celltypeLabel_,
                              divqLabel_ );
    
    rmcrt_->problemSetup(radiationSpec, radiationSpec, grid, sharedState);
    
    rmcrt_->BC_bulletproofing( radiationSpec );
  }
  
  //--------------------------------------------------------------------
  
  void
  RadiationSource::schedule_setup_bndflux( const Uintah::LevelP& level,
                                          Uintah::SchedulerP sched,
                                          const Uintah::MaterialSet* const materials )
  {
    // hack in a task to apply boundary condition on the pressure after the pressure solve
    Uintah::Task* task = scinew Uintah::Task( "RadiationSource: setup bndflux", this,
                                             &RadiationSource::setup_bndflux );
    
    task->computes(Uintah::VarLabel::find("boundFlux"));
    sched->addTask(task, level->eachPatch(), materials);
  }

  //--------------------------------------------------------------------
  
  void
  RadiationSource::setup_bndflux ( const Uintah::ProcessorGroup* const pg,
                                  const Uintah::PatchSubset* const patches,
                                  const Uintah::MaterialSubset* const materials,
                                  Uintah::DataWarehouse* const oldDW,
                                  Uintah::DataWarehouse* const newDW )
  {
    for (int p=0; p < patches->size(); p++){
      const Uintah::Patch* patch = patches->get(p);
      Uintah::CCVariable<Uintah::Stencil7> boundFlux;
      newDW->allocateAndPut( boundFlux, Uintah::VarLabel::find("boundFlux"), 0, patch );
      
      for (Uintah::CellIterator iter = patch->getExtraCellIterator(); !iter.done(); iter++){
        Uintah::IntVector origin = *iter;
        boundFlux[origin].initialize(0.0);
      }
      
    }
  }
  
  //--------------------------------------------------------------------
  
  RadiationSource::~RadiationSource()
  {
    Uintah::VarLabel::destroy( temperatureLabel_ );
    Uintah::VarLabel::destroy( absorptionLabel_  );
    Uintah::VarLabel::destroy( celltypeLabel_    );
    Uintah::VarLabel::destroy( divqLabel_        );
  }
  
  //--------------------------------------------------------------------
  
  void
  RadiationSource::schedule_ray_tracing( const Uintah::LevelP& level,
                                        Uintah::SchedulerP sched,
                                        const Uintah::MaterialSet* const materials,
                                        const int RKStage )
  {
    using namespace Uintah;
    
    GridP grid = level->getGrid();
    const bool modifiesDivQ      = true;
    const bool includeExtraCells = false;  // domain for sigmaT4 computation
    const int radiationCalcFreq = 5;
    // only sched on RK step 0 and on arches level
    if ( RKStage > 1 ) {
      return;
    }
    
    schedule_setup_bndflux(level, sched, materials);
    
    int maxLevels = grid->numLevels();
    {
      const LevelP& fineLevel = grid->getLevel(0);
      Uintah::Task::WhichDW tempDW   = Task::NewDW;
      Uintah::Task::WhichDW abskgDW  = Task::NewDW;
      
      // if needed convert absorptionLabel: double -> float
      rmcrt_->sched_DoubleToFloat( fineLevel, sched, abskgDW, radiationCalcFreq );
      
      rmcrt_->sched_sigmaT4( fineLevel,  sched, tempDW, 1, includeExtraCells );
      
      for (int l = 0; l < maxLevels; l++) {
        const LevelP& level = grid->getLevel(l);
        const bool modifies_abskg   = false;
        const bool modifies_sigmaT4 = false;
        
        rmcrt_->sched_CoarsenAll( level, sched, modifies_abskg, modifies_sigmaT4, radiationCalcFreq );
        
        if(level->hasFinerLevel() || maxLevels == 1){
          const Task::WhichDW sigmaT4DW  = Task::NewDW;
          const Task::WhichDW celltypeDW = Task::NewDW;
          
          rmcrt_->sched_setBoundaryConditions( level, sched, tempDW, radiationCalcFreq, false );
          rmcrt_->sched_rayTrace( level, sched, abskgDW, sigmaT4DW, celltypeDW, modifiesDivQ, radiationCalcFreq );
        }
      }
      
      // push divQ  to the coarser levels
      for (int l = 0; l < maxLevels; l++) {
        const LevelP& level = grid->getLevel(l);
        const PatchSet* patches = level->eachPatch();
        rmcrt_->sched_Refine_Q( sched,  patches, materials, radiationCalcFreq );
      }
    }
    
  }
  
  
  //--------------------------------------------------------------------
  
  void
  RadiationSource::declare_uintah_vars( Uintah::Task& task,
                                       const Uintah::PatchSubset* const patches,
                                       const Uintah::MaterialSubset* const materials,
                                       const int RKStage )
  {}
  
  //--------------------------------------------------------------------
  
  void
  RadiationSource::bind_uintah_vars( Uintah::DataWarehouse* const dw,
                                    const Uintah::Patch* const patch,
                                    const int material,
                                    const int RKStage )
  {}
  
  //--------------------------------------------------------------------
  
  void
  RadiationSource::bind_operators( const SpatialOps::OperatorDatabase& opDB )
  {}
  
  //--------------------------------------------------------------------
  
  void
  RadiationSource::evaluate()
  {
    using namespace SpatialOps;
    
    typedef std::vector<SVolField*> SVolFieldVec;
    SVolFieldVec& results = this->get_value_vec();
    SVolField& radvolq = *results[1];
    radvolq <<= 0.0;
    SVolField& radvrflux = *results[2];
    radvrflux <<= 0.0;
  }
  
  //--------------------------------------------------------------------
  
  RadiationSource::Builder::Builder( const Expr::TagList& results,
                                    const Expr::Tag& temperatureTag,
                                    const Expr::Tag& absorptionTag,
                                    const Expr::Tag& celltypeTag,
                                    Uintah::Ray* rmcrt,
                                    Uintah::ProblemSpecP& radiationSpec,
                                    Uintah::SimulationStateP& sharedState,
                                    Uintah::GridP& grid)
  : ExpressionBuilder  ( results        ),
  temperatureTag_    ( temperatureTag ),
  absorptionTag_     ( absorptionTag  ),
  celltypeTag_       ( celltypeTag    ),
  rmcrt_             ( rmcrt          ),
  radiationSpec_     ( radiationSpec  ),
  sharedState_       ( sharedState    ),
  grid_              ( grid           )
  {}
  
  //--------------------------------------------------------------------
  
  Expr::ExpressionBase*
  RadiationSource::Builder::build() const
  {
    const Expr::TagList radTags = get_tags();
    return new RadiationSource( radTags[0].name(), temperatureTag_, absorptionTag_, celltypeTag_, rmcrt_, radiationSpec_, sharedState_, grid_ );
  }
  
} // namespace Wasatch
