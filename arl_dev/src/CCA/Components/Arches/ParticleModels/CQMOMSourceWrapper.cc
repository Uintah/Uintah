#include <CCA/Components/Arches/ParticleModels/CQMOMSourceWrapper.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>

using namespace std;
using namespace Uintah;

CQMOMSourceWrapper::CQMOMSourceWrapper( ArchesLabel* fieldLabels, std::string sourceName, std::vector<int> momentIndex, int nIC )
{
  d_fieldLabels = fieldLabels;
  d_momentIndex = momentIndex;
  _nIC = nIC;
  d_modelLabel = VarLabel::create(sourceName, CCVariable<double>::getTypeDescription());
}

CQMOMSourceWrapper::~CQMOMSourceWrapper()
{
  VarLabel::destroy(d_modelLabel);
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
CQMOMSourceWrapper::problemSetup(const ProblemSpecP& inputdb)
{
  ProblemSpecP db = inputdb;
  
  ProblemSpecP db_root = db->getRootNode();
  ProblemSpecP cqmom_db = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("CQMOM");

  cqmom_db->get("NumberInternalCoordinates",M);
  std::vector<int> N_i;
  cqmom_db->get("QuadratureNodes",N_i);
  _N = 1;
  for (int i = 0; i < M; i++) {
    _N *= N_i[i];
  }
  
  inputdb->getAttribute("label",model_name);
  
  for ( int i = 0; i < _N; i++ ) {
    string thisNodeSource;
    string node;
    std::stringstream index;
    index << i;
    node = index.str();
    
    thisNodeSource = model_name + "_" + node;
    const VarLabel * tempLabel;
    tempLabel = VarLabel::find( thisNodeSource );
    //      cout << "found varlabel for " << thisNodeSource << " " << tempLabel << endl; //check that this is finding right varlabels made by model
    d_nodeSources.push_back(tempLabel);
  }
}

//---------------------------------------------------------------------------
// Method: Schedule the intialization of the variables.
//---------------------------------------------------------------------------
void
CQMOMSourceWrapper::sched_initializeVariables( const LevelP& level, SchedulerP& sched )
{
  string taskname = "CQMOMSourceWrapper::initializeVariables";
  Task* tsk = scinew Task(taskname, this, &CQMOMSourceWrapper::initializeVariables);

  //New
  tsk->computes(d_modelLabel);
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually initialize the variables.
//---------------------------------------------------------------------------
void
CQMOMSourceWrapper::initializeVariables( const ProcessorGroup* pc,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){
    
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    
    CCVariable<double> model;

    new_dw->allocateAndPut( model, d_modelLabel, matlIndex, patch );
    model.initialize(0.0);
    
  }
}

//---------------------------------------------------------------------------
// Method: Schedule building the source term
//---------------------------------------------------------------------------
void
CQMOMSourceWrapper::sched_buildSourceTerm( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "CQMOMSourceWrapper::buildSourceTerm";
  Task* tsk = scinew Task(taskname, this, &CQMOMSourceWrapper::buildSourceTerm);

  //----NEW----
  tsk->modifies(d_modelLabel);

  //loop over requires for weights and abscissas needed
  for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
    const VarLabel* tempLabel = iW->second;
    if (timeSubStep == 0 ) {
      tsk->requires( Task::OldDW, tempLabel, Ghost::None, 0 );
    } else {
      tsk->requires( Task::NewDW, tempLabel, Ghost::None, 0 );
    }
  }
  for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
    const VarLabel* tempLabel = iA->second;
    if (timeSubStep == 0 ) {
      tsk->requires( Task::OldDW, tempLabel, Ghost::None, 0 );
    } else {
      tsk->requires( Task::NewDW, tempLabel, Ghost::None, 0 );
    }
  }

  //loop over all the d\phi/dt sources
  for ( int i = 0; i < _N; i++ ) {
    const VarLabel* tempLabel = d_nodeSources[i];
    tsk->requires( Task::NewDW, tempLabel, Ghost::None, 0 );
  }
  
  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());
}
//---------------------------------------------------------------------------
// Method: Actually build the source term
//---------------------------------------------------------------------------
void
CQMOMSourceWrapper::buildSourceTerm( const ProcessorGroup* pc,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++) {
    Ghost::GhostType  gn  = Ghost::None;
    
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    
    CCVariable<double> model;
    new_dw->getModifiable( model, d_modelLabel, matlIndex, patch );

    vector<constCCVarWrapper> cqmomWeights;
    vector<constCCVarWrapper> cqmomAbscissas;
    vector<constCCVarWrapper> nodeSource;
 
    //get weights and abscissas from dw
    for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
      const VarLabel* tempLabel = iW->second;
      constCCVarWrapper tempWrapper;
      if (new_dw->exists( tempLabel, matlIndex, patch) ) {
        new_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gn, 0 );
      } else {
        old_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gn, 0 );
      }
      cqmomWeights.push_back(tempWrapper);
    }
        
    for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
      const VarLabel* tempLabel = iA->second;
      constCCVarWrapper tempWrapper;
      if (new_dw->exists( tempLabel, matlIndex, patch) ) {
        new_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gn, 0 );
      } else {
        old_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gn, 0 );
      }
      cqmomAbscissas.push_back(tempWrapper);
    }

    //get the list of source terms for each of the nodes
    for ( int i = 0; i < _N; i++ ) {
      const VarLabel* tempLabel = d_nodeSources[i];
      constCCVarWrapper tempWrapper;
      new_dw->get( tempWrapper.data, tempLabel, matlIndex, patch, gn, 0 );
      nodeSource.push_back( tempWrapper );
    }

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      
      std::vector<double> temp_weights ( _N, 0.0 );
      std::vector<double> temp_abscissas ( _N * M, 0.0 );
      std::vector<double> temp_sources ( _N, 0.0 );
      
      //put all the weigths, abscissas and sources in temp vectors
      int ii = 0;
      for (vector<constCCVarWrapper>::iterator iter = cqmomWeights.begin(); iter!= cqmomWeights.end(); ++iter) {
        double temp_value = (iter->data)[c];
        temp_weights[ii] = temp_value;
        ii++;
      }
      ii = 0;
      for (vector<constCCVarWrapper>::iterator iter = cqmomAbscissas.begin(); iter!= cqmomAbscissas.end(); ++iter) {
        double temp_value = (iter->data)[c];
        temp_abscissas[ii] = temp_value;
        ii++;
      }
      ii = 0;
      for (vector<constCCVarWrapper>::iterator iter = nodeSource.begin(); iter!= nodeSource.end(); ++iter) {
        double temp_value = (iter->data)[c];
        temp_sources[ii] = temp_value;
        ii++;
      }

      double d_small = 1e-10;
      double sum = 0.0;
      for ( int i = 0; i < _N; i++ ) { //loop each quad node
        double product = 1.0;
        for ( int m = 0; m < M; m++ ) { //loop each internal coordinate
//          std::cout << "i[" << i << "] m[" << m << "]" << std::endl;
//          std::cout << temp_abscissas[i + _N * m] << std::endl;
//          std::cout << d_momentIndex[m] << std::endl;
          if ( m != _nIC ) {
            product *= pow( temp_abscissas[i + _N * m], d_momentIndex[m] );
          } else {
            if ( d_momentIndex[m]-1 < 0 && temp_abscissas[i + _N * m] < d_small ) {
              //ignore 0 abscisssa or very small ones from making infs here
            } else {
              product *= pow( temp_abscissas[i + _N * m], d_momentIndex[m]-1 );
            }
          }
        }
        product *= temp_weights[i] * temp_sources[i];
        sum += product;
      }
      model[c] = d_momentIndex[_nIC] * sum;
    } //cell loop
  } //patch loop
}

