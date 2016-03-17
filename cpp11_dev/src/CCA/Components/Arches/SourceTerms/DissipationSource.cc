#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/SourceTerms/DissipationSource.h>

using namespace std;
using namespace Uintah; 

DissipationSource::DissipationSource( std::string src_name, SimulationStateP& shared_state,
                                      vector<std::string> req_label_names, std::string type ) 
: SourceTermBase(src_name, shared_state, req_label_names, type)
{
  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() ); 
}

DissipationSource::~DissipationSource()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
DissipationSource::problemSetup(const ProblemSpecP& inputdb)
{
  ProblemSpecP db = inputdb; 
  
  db->getWithDefault("density_label", _density, "density");  
  db->getWithDefault("turb_visc_label", _mu_t, "turb_viscosity");
  db->require("mixture_fraction_label",_mixfrac);
  db->require("gradmixfrac_label", _grad_mixfrac2);  
  db->require("D", _D);
  
  _source_grid_type = CC_SRC; 
  
}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term 
//---------------------------------------------------------------------------
void 
DissipationSource::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "DissipationSource::eval";
  Task* tsk = scinew Task(taskname, this, &DissipationSource::computeSource, timeSubStep);
  
  if (timeSubStep == 0 ) { 
    tsk->computes(_src_label);
  } else {
    tsk->modifies(_src_label); 
  }
  
  _densityLabel = VarLabel::find(_density);
  _mixFracLabel = VarLabel::find(_mixfrac);
  _gradMixFrac2Label = VarLabel::find(_grad_mixfrac2);
  _turbViscLabel = VarLabel::find(_mu_t);
  _ccVelocityLabel = VarLabel::find("CCVelocity");
  _volfrac_label = VarLabel::find("volFraction");
  
  if (timeSubStep == 0) {
    tsk->requires( Task::OldDW, _densityLabel, Ghost::None, 0 ); 
    tsk->requires( Task::OldDW, _mixFracLabel, Ghost::AroundCells, 1 );
    tsk->requires( Task::OldDW, _gradMixFrac2Label, Ghost::None, 0 );
    tsk->requires( Task::OldDW, _turbViscLabel, Ghost::None, 0 );
    tsk->requires( Task::OldDW, _volfrac_label, Ghost::AroundCells, 1 );
    tsk->requires( Task::OldDW, _ccVelocityLabel, Ghost::AroundCells, 1 );
  } else {
    tsk->requires( Task::NewDW, _densityLabel, Ghost::None, 0 ); 
    tsk->requires( Task::NewDW, _mixFracLabel, Ghost::AroundCells, 1 );
    tsk->requires( Task::NewDW, _gradMixFrac2Label, Ghost::None, 0 );
    tsk->requires( Task::NewDW, _turbViscLabel, Ghost::None, 0 );
    tsk->requires( Task::NewDW, _volfrac_label, Ghost::AroundCells, 1 );
    tsk->requires( Task::NewDW, _ccVelocityLabel, Ghost::AroundCells, 1 );
  }
  
  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials() ); 
}
//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
DissipationSource::computeSource( const ProcessorGroup* pc, 
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
    
    CCVariable<double> rateSrc; 
    constCCVariable<double> den;    // mixture density
    constCCVariable<double> mf;     // mixture fraction
    constCCVariable<double> gradmf; //grad mixfrac squared
    constCCVariable<double> mu_t;   //turb viscosity
    constCCVariable<Vector> CCV;    //cell center velocity
    constCCVariable<double> vf;     //volume fraction
    
    if (timeSubStep == 0) {
      new_dw->allocateAndPut( rateSrc, _src_label, matlIndex, patch );
      old_dw->get( den, _densityLabel, matlIndex, patch, Ghost::None , 0);
      old_dw->get( mf, _mixFracLabel, matlIndex, patch, Ghost::AroundCells , 1);
      old_dw->get( gradmf, _gradMixFrac2Label, matlIndex, patch, Ghost::None , 0);
      old_dw->get( mu_t, _turbViscLabel, matlIndex, patch, Ghost::None , 0);
      old_dw->get( vf, _volfrac_label, matlIndex, patch, Ghost::AroundCells, 1);
      old_dw->get( CCV, _ccVelocityLabel, matlIndex, patch, Ghost::AroundCells , 1);
      rateSrc.initialize(0.0);
    } else {
      new_dw->getModifiable( rateSrc, _src_label, matlIndex, patch );
      new_dw->get( den, _densityLabel, matlIndex, patch, Ghost::None , 0);
      new_dw->get( mf, _mixFracLabel, matlIndex, patch, Ghost::AroundCells , 1);
      new_dw->get( gradmf, _gradMixFrac2Label, matlIndex, patch, Ghost::None , 0);
      new_dw->get( mu_t, _turbViscLabel, matlIndex, patch, Ghost::None , 0);
      new_dw->get( vf, _volfrac_label, matlIndex, patch, Ghost::AroundCells, 1);
      new_dw->get( CCV, _ccVelocityLabel, matlIndex, patch, Ghost::AroundCells , 1);
    }
    
    Vector Dx = patch->dCell(); 
    double filterWidth = 3.0 * Dx.x(); //get this from dw?
    double small = 1e-10;
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 
      IntVector cxm = c - IntVector(1,0,0); 
      IntVector cxp = c + IntVector(1,0,0); 
      IntVector cym = c - IntVector(0,1,0); 
      IntVector cyp = c + IntVector(0,1,0); 
      IntVector czm = c - IntVector(0,0,1); 
      IntVector czp = c + IntVector(0,0,1); 
      
      //indicies for mixed 2nd derivatives
      IntVector cxmym = c - IntVector(1,1,0);
      IntVector cxmyp = c + IntVector(-1,1,0);
      IntVector cxpym = c + IntVector(1,-1,0);
      IntVector cxpyp = c + IntVector(1,1,0);
      
      IntVector cxmzm = c - IntVector(1,0,1);
      IntVector cxmzp = c + IntVector(-1,0,1);
      IntVector cxpzm = c + IntVector(1,0,-1);
      IntVector cxpzp = c + IntVector(1,0,1);
      
      IntVector cymzm = c - IntVector(0,1,1);
      IntVector cymzp = c + IntVector(0,-1,1);
      IntVector cypzm = c + IntVector(0,1,-1);
      IntVector cypzp = c + IntVector(0,1,1);
      
      //handle embbeded geometry with dz = 0 in wall dir
      double mfxp, mfxm, mfyp, mfym, mfzp, mfzm;
      mfxp = (vf[cxp] > 0.0) ? mf[cxp] : mf[c];
      mfxm = (vf[cxm] > 0.0) ? mf[cxm] : mf[c];
      mfyp = (vf[cyp] > 0.0) ? mf[cyp] : mf[c];
      mfym = (vf[cym] > 0.0) ? mf[cym] : mf[c];
      mfzp = (vf[czp] > 0.0) ? mf[czp] : mf[c];
      mfzm = (vf[czm] > 0.0) ? mf[czm] : mf[c];
      
      //resolved production term
      double resProd = - 2.0 * den[c] *
                       ( (CCV[cxp][0] - CCV[cxm][0])/Dx.x() * (mfxp-mfxm)/Dx.x() * (mfxp-mfxm)/Dx.x()/8.0 +
                       (CCV[cyp][0] - CCV[cym][0])/Dx.y() * (mfxp-mfxm)/Dx.x() * (mfyp-mfym)/Dx.y()/8.0 +
                       (CCV[czp][0] - CCV[czm][0])/Dx.z() * (mfxp-mfxm)/Dx.x() * (mfzp-mfzm)/Dx.z()/8.0 +
                       (CCV[cxp][1] - CCV[cxm][1])/Dx.x() * (mfyp-mfym)/Dx.y() * (mfxp-mfxm)/Dx.x()/8.0 +
                       (CCV[cyp][1] - CCV[cym][1])/Dx.y() * (mfyp-mfym)/Dx.y() * (mfyp-mfym)/Dx.y()/8.0 +
                       (CCV[czp][1] - CCV[czm][1])/Dx.z() * (mfyp-mfym)/Dx.y() * (mfzp-mfzm)/Dx.z()/8.0 +
                       (CCV[cxp][2] - CCV[cxm][2])/Dx.x() * (mfzp-mfzm)/Dx.z() * (mfxp-mfxm)/Dx.x()/8.0 +
                       (CCV[cyp][2] - CCV[cym][2])/Dx.y() * (mfzp-mfzm)/Dx.z() * (mfyp-mfym)/Dx.y()/8.0 + 
                       (CCV[czp][2] - CCV[czm][2])/Dx.z() * (mfzp-mfzm)/Dx.z() * (mfzp-mfzm)/Dx.z()/8.0 );
  
      double gradZ2 = 0.5/Dx.x() * ( mfxp - mfxm ) * 0.5/Dx.x() * ( mfxp - mfxm ) + 
                      0.5/Dx.y() * ( mfyp - mfym ) * 0.5/Dx.y() * ( mfyp - mfym ) +
                      0.5/Dx.z() * ( mfzp - mfzm ) * 0.5/Dx.z() * ( mfzp - mfzm );
      
      //unresolved prod
      double unresProd = den[c] * 32.0 * mu_t[c] / filterWidth / filterWidth * ( gradmf[c] - gradZ2 );
      
      //resolved dissipation
      //set up corner indicies
      double mfxpyp, mfxpym, mfxmyp, mfxmym, mfxpzp, mfxpzm, mfxmzp, mfxmzm, mfypzp, mfypzm, mfymzp, mfymzm;
      mfxpyp = ( vf[cxpyp] > 0.0) ?  mf[cxpyp] : mf[c];
      mfxpym = ( vf[cxpym] > 0.0) ?  mf[cxpym] : mf[c];
      mfxmyp = ( vf[cxmyp] > 0.0) ?  mf[cxmyp] : mf[c];
      mfxmym = ( vf[cxmym] > 0.0) ?  mf[cxmym] : mf[c];
      mfxpzp = ( vf[cxpzp] > 0.0) ?  mf[cxpzp] : mf[c];
      mfxpzm = ( vf[cxpzm] > 0.0) ?  mf[cxpzm] : mf[c];
      mfxmzp = ( vf[cxmzp] > 0.0) ?  mf[cxmzp] : mf[c];
      mfxmzm = ( vf[cxmzm] > 0.0) ?  mf[cxmzm] : mf[c];
      mfypzp = ( vf[cypzp] > 0.0) ?  mf[cypzp] : mf[c];
      mfypzm = ( vf[cypzm] > 0.0) ?  mf[cypzm] : mf[c];
      mfymzp = ( vf[cymzp] > 0.0) ?  mf[cymzp] : mf[c];
      mfymzm = ( vf[cymzm] > 0.0) ?  mf[cymzm] : mf[c];
 
     double resDiss = -2.0 * den[c] * _D *
                       pow( (mfxp - 2.0 * mf[c] + mfxm)/(Dx.x() * Dx.x() ) +
                            (mfxpyp - mfxpym - mfxmyp + mfxmym)/(4.0 * Dx.x() * Dx.y() ) +
                            (mfxpzp - mfxpzm - mfxmzp + mfxmzm)/(4.0 * Dx.x() * Dx.z() ) +
                            (mfxpyp - mfxpym - mfxmyp + mfxmym)/(4.0 * Dx.x() * Dx.y() ) +
                            (mfyp - 2.0 * mf[c] + mfym)/(Dx.y() * Dx.y() ) + 
                            (mfypzp - mfypzm - mfymzp + mfymzm)/(4.0 * Dx.y() * Dx.z() ) +
                            (mfxpzp - mfxpzm - mfxmzp + mfxmzm)/(4.0 * Dx.x() * Dx.z() ) +
                            (mfypzp - mfypzm - mfymzp + mfymzm)/(4.0 * Dx.y() * Dx.z() ) +
                            (mfzp - 2.0 * mf[c] + mfzm)/(Dx.z() * Dx.z() ), 2.0 );
      
      
      //unresolved dissipation
      double unresDiss;
      if ( gradZ2 > small) {
        unresDiss = - 12.0 / filterWidth / filterWidth / gradZ2 * den[c] * _D * ( gradmf[c] - gradZ2 ) * ( gradmf[c] - gradZ2 );
      } else {
        unresDiss = 0.0;
      }
      
      if (vf[c] > 0.0 && gradZ2 > small ) {
        rateSrc[c] = resProd + unresProd + resDiss + unresDiss;
      } else {
        rateSrc[c] = 0.0;
      }

      
    }
    
  }
}

//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
DissipationSource::sched_initialize( const LevelP& level, SchedulerP& sched )
{
  string taskname = "DissipationSource::initialize"; 
  
  Task* tsk = scinew Task(taskname, this, &DissipationSource::initialize);
  
  tsk->computes(_src_label);
  
  for (std::vector<const VarLabel*>::iterator iter = _extra_local_labels.begin(); iter != _extra_local_labels.end(); iter++){
    tsk->computes(*iter); 
  }
  
  sched->addTask(tsk, level->eachPatch(), _shared_state->allArchesMaterials() );
}
//---------------------------------------------------------------------------
// Method: initialization
//---------------------------------------------------------------------------
void 
DissipationSource::initialize( const ProcessorGroup* pc, 
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
  }
}
