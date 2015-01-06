#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/BowmanNOx.h>

#include <sci_defs/uintah_defs.h>
#ifdef WASATCH_IN_ARCHES
//-- SpatialOps includes --//
#include <spatialops/Nebo.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>
#endif

//===========================================================================

using namespace std;
using namespace Uintah; 

BowmanNOx::BowmanNOx( std::string src_name, ArchesLabel* field_labels,
                            vector<std::string> req_label_names, std::string type ) 
: SourceTermBase(src_name, field_labels->d_sharedState, req_label_names, type), _field_labels(field_labels)
{ 


  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() ); 

  _source_grid_type = CC_SRC; 

  _MW_O2 = 32.00; 
  _MW_N2 = 28.00; 

}

BowmanNOx::~BowmanNOx()
{
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
BowmanNOx::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb; 

  db->getWithDefault("A", _A, 6.0e16); 
  db->getWithDefault("E_R", _E_R, 69090); 
  db->getWithDefault("o2_label", _o2_name, "O2"); 
  db->getWithDefault("n2_label", _n2_name, "N2"); 
  db->getWithDefault("density_label", _rho_name, "density"); 
  db->getWithDefault("temperature_label", _temperature_name, "temperature"); 

  _field_labels->add_species( _o2_name ); 
  _field_labels->add_species( _n2_name ); 
  _field_labels->add_species( _rho_name ); 
  _field_labels->add_species( _temperature_name ); 


}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
BowmanNOx::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "BowmanNOx::eval";
  Task* tsk = scinew Task(taskname, this, &BowmanNOx::computeSource, timeSubStep);

  Ghost::GhostType  gType;
  int nGhosts;
  
#ifdef WASATCH_IN_ARCHES
  gType = Ghost::AroundCells;
  nGhosts = Wasatch::get_n_ghost<SVolField>();
#else
  gType = Ghost::None;
  nGhosts = 0;
#endif

  if (timeSubStep == 0) {
    tsk->computes(_src_label);
  } else {
#ifdef WASATCH_IN_ARCHES
    
    tsk->modifiesWithScratchGhost( _src_label,
                                  level->eachPatch()->getUnion(), Uintah::Task::ThisLevel,
                                  _shared_state->allArchesMaterials()->getUnion(), Uintah::Task::NormalDomain,
                                  gType, nGhosts);
#else
    tsk->modifies(_src_label);
#endif
  }

  // resolve some labels: 
  _n2_label = VarLabel::find( _n2_name ); 
  _o2_label  = VarLabel::find( _o2_name  ); 
  _rho_label = VarLabel::find( _rho_name ); 
  _temperature_label = VarLabel::find( _temperature_name ); 


  tsk->requires( Task::OldDW, _n2_label, gType, nGhosts );
  tsk->requires( Task::OldDW, _o2_label, gType, nGhosts );
  tsk->requires( Task::OldDW, _rho_label, gType, nGhosts );
  tsk->requires( Task::OldDW, _temperature_label, gType, nGhosts );
  tsk->requires( Task::OldDW, _field_labels->d_volFractionLabel, gType, nGhosts ); 

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials()); 

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
BowmanNOx::computeSource( const ProcessorGroup* pc, 
                   const PatchSubset* patches, 
                   const MaterialSubset* matls, 
                   DataWarehouse* old_dw, 
                   DataWarehouse* new_dw, 
                   int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> rate; 
    Ghost::GhostType  gType;
    int nGhosts;

#ifdef WASATCH_IN_ARCHES
    gType = Ghost::AroundCells;
    nGhosts = Wasatch::get_n_ghost<SVolField>();
#else
    gType = Ghost::None;
    nGhosts = 0;
#endif

    if ( new_dw->exists(_src_label, matlIndex, patch ) ){
      new_dw->getModifiable( rate, _src_label, matlIndex, patch, gType, nGhosts );
    } else { 
      new_dw->allocateAndPut( rate, _src_label, matlIndex, patch, gType, nGhosts );
      rate.initialize(0.0); 
    } 

    constCCVariable<double> N2; 
    constCCVariable<double> O2; 
    constCCVariable<double> rho; 
    constCCVariable<double> T; 
    constCCVariable<double> vol_fraction;

    old_dw->get( N2 , _n2_label          , matlIndex , patch , gType , nGhosts );
    old_dw->get( O2  , _o2_label          , matlIndex , patch , gType , nGhosts );
    old_dw->get( rho , _rho_label         , matlIndex , patch , gType , nGhosts );
    old_dw->get( T   , _temperature_label , matlIndex , patch , gType , nGhosts );
    old_dw->get( vol_fraction, _field_labels->d_volFractionLabel, matlIndex, patch, gType, nGhosts );


#   ifdef WASATCH_IN_ARCHES
    using namespace Wasatch;
    using namespace SpatialOps;
    using SpatialOps::operator *;
    typedef SpatFldPtr<SVolField> SVolPtr;

    // SVolField = CCVariable<double>
    const Wasatch::AllocInfo ainfo( old_dw, new_dw, matlIndex, patch, pc );
    const SpatialOps::GhostData gd( nGhosts );
    SVolPtr rate_        = wrap_uintah_field_as_spatialops<SVolField>(rate,ainfo, gd );
    const SVolPtr N2_    = wrap_uintah_field_as_spatialops<SVolField>(N2,  ainfo, gd );
    const SVolPtr O2_    = wrap_uintah_field_as_spatialops<SVolField>(O2,  ainfo, gd );
    const SVolPtr rho_   = wrap_uintah_field_as_spatialops<SVolField>(rho, ainfo, gd );
    const SVolPtr T_     = wrap_uintah_field_as_spatialops<SVolField>(T,   ainfo, gd );
    const SVolPtr vfrac_ = wrap_uintah_field_as_spatialops<SVolField>(vol_fraction,ainfo,gd);
    
    *rate_ <<= 30000.0 * _A / ( sqrt(*T_) ) * exp(-_E_R/ *T_) * (1.0e-3/_MW_N2 * *N2_ * *rho_) * sqrt(1.0e-3/_MW_O2 * *O2_ * *rho_);
    *rate_ <<= cond( *rate_ < 1.0e-16, 0.0)
                   ( *rate_ );
#   else
//    delt_vartype DT;
//    old_dw->get(DT, _field_labels->d_sharedState->get_delt_label()); 
//    double dt = DT;

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 

      //convert to mol/cm^3
      double n2  = N2[c] * rho[c] / _MW_N2 * 1.0e-3; 
      double o2  = O2[c] * rho[c] / _MW_O2 * 1.0e-3; 

      double T_pow = sqrt( T[c] );
      double o2_pow = sqrt( o2  );

      double my_exp = -1.0 * _E_R / T[c]; 

//      rate[c] = T[c];
      rate[c] = _A / T_pow * exp( my_exp ) * n2 * o2_pow * 30000;

//      rate[c] = 30000.0 * _A / sqrt(T[c]) * exp( -_E_R / T[c] ) * ( 1.0e-3/_MW_N2 * N2[c] * rho[c]) * sqrt(1.0e-3/ _MW_O2 * O2[c] * rho[c]);
      
      if ( rate[c] < 1.0e-16 ){
        rate[c] = 0.0;
      } 
    }
#   endif
  }
}
//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
BowmanNOx::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "BowmanNOx::initialize"; 

  Task* tsk = scinew Task(taskname, this, &BowmanNOx::initialize);

  tsk->computes(_src_label);

  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter, _shared_state->allArchesMaterials()->getUnion()); 
  }

  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials());

}
void 
BowmanNOx::initialize( const ProcessorGroup* pc, 
                       const PatchSubset* patches, 
                       const MaterialSubset* matls, 
                       DataWarehouse* old_dw, 
                       DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _shared_state->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> src;
    new_dw->allocateAndPut( src, _src_label, matlIndex, patch ); 

    src.initialize(0.0); 

    for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
      const VarLabel* tempVL = *iter; 
      CCVariable<double> tempVar; 
      new_dw->allocateAndPut(tempVar, tempVL, matlIndex, patch ); 
      tempVar.initialize(0.0); 
    }
  }
}



