/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <CCA/Components/Arches/ParticleModels/CQMOMSourceWrapper.h>

#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Ports/Scheduler.h>

#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/Parallel.h>

using namespace std;
using namespace Uintah;

CQMOMSourceWrapper::CQMOMSourceWrapper( ArchesLabel* fieldLabels )
{
  d_fieldLabels = fieldLabels;
  d_addSources = false;
}

CQMOMSourceWrapper::~CQMOMSourceWrapper()
{
  for (int i = 0; i < nMoments*nSources; i++) {
    VarLabel::destroy( d_sourceLabels[i] );
  }
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
  ProblemSpecP models_db = db_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ParticleModels");

  cqmom_db->get("NumberInternalCoordinates",M);
  std::vector<int> N_i;
  cqmom_db->get("QuadratureNodes",N_i);
  _N = 1;
  for (int i = 0; i < M; i++) {
    _N *= N_i[i];
  }
  
  //get moment indexes
  nMoments = 0;
  for ( ProblemSpecP db_moments = db->findBlock("Moment"); db_moments != nullptr; db_moments = db_moments->findNextBlock("Moment") ) {
    //make list of moment indexes
    vector<int> temp_moment_index;
    db_moments->get("m", temp_moment_index);
    momentIndexes.push_back(temp_moment_index);
    nMoments++; // keep track of total number of moments
  }
  
  //loop through each source found in ParticleModels tags
  nSources = 0;
  if ( models_db ) {
    d_addSources = true;
    for (ProblemSpecP m_db = models_db->findBlock("model"); m_db != nullptr; m_db = m_db->findNextBlock("model")) {
      //parse the model blocks for var label
      std::string model_name;
      std::string source_label;
      std::string ic_name;
      
      m_db->getAttribute("label",model_name);
      if ( m_db->findBlock("IC") ) {
        m_db->get("IC",ic_name);
        proc0cout << "Model name: " << model_name << endl;

        int m = 0;
        for ( ProblemSpecP db_name = cqmom_db->findBlock("InternalCoordinate"); db_name != nullptr; db_name = db_name->findNextBlock("InternalCoordinate") ) {
          std::string var_name;
          db_name->getAttribute("name",var_name);
          if ( var_name == ic_name) {
            nIC.push_back(m);
            break;
          }
          m++;
          if ( m >= M ) { // occurs if IC not found
            string err_msg = "Error: could not find internal coordinate '" + ic_name + "' in list of internal coordinates specified by CQMOM spec";
            throw ProblemSetupException(err_msg,__FILE__,__LINE__);
          }
        }
      
        //store list of all needed node sources
        for ( int i = 0; i < _N; i++ ) {
          string thisNodeSource;
          string node;
          std::stringstream index;
          index << i;
          node = index.str();
        
          thisNodeSource = model_name + "_" + node;
          const VarLabel * tempLabel;
          tempLabel = VarLabel::find( thisNodeSource );
          d_nodeSources.push_back(tempLabel);
        }
      
        //create & store var labels for this source for all moment eqns
        for ( int i = 0; i < nMoments; i ++ ) {
          vector<int> temp_moment_index = momentIndexes[i];
        
          //base moment name
          string eqn_name = "m_";
          for (int n = 0; n < M ; n++) {
            string node;
            std::stringstream out;
            out << temp_moment_index[n];
            node = out.str();
            eqn_name += node;
          }
        
          //create varlabel
          string source_name;
          source_name = eqn_name + "_" + model_name + "_src";
          const VarLabel* tempLabel;
          proc0cout << "Creating source label: " << source_name << endl;
          tempLabel = VarLabel::create( source_name, CCVariable<double>::getTypeDescription() );
          d_sourceLabels.push_back(tempLabel);
        }
        nSources++; //count total number of sources
      } else {
        proc0cout << model_name << " parsed as a particle property" << endl;
      }
    }
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
  for ( int i = 0; i < nMoments*nSources; i++ ) {
    tsk->computes(d_sourceLabels[i]);
  }
  
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
    
    for ( int i = 0; i < nMoments*nSources; i++ ) {
      CCVariable<double> src;
      new_dw->allocateAndPut( src, d_sourceLabels[i], matlIndex, patch );
      src.initialize(0.0);
    }
    
  }
}

//---------------------------------------------------------------------------
// Method: Schedule building the source term
//---------------------------------------------------------------------------
void
CQMOMSourceWrapper::sched_buildSourceTerm( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  string taskname = "CQMOMSourceWrapper::buildSourceTerm";
  Task* tsk = scinew Task(taskname, this, &CQMOMSourceWrapper::buildSourceTerm, timeSubStep);

  //----NEW----
  for ( int i = 0; i < nMoments*nSources; i++ ) {
    tsk->modifies(d_sourceLabels[i]);
  }

  volfrac_label = VarLabel::find( "volFraction" );
  tsk->requires( Task::OldDW, volfrac_label, Ghost::None, 0);
  
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
  Task::WhichDW which_dw; 
  if ( timeSubStep == 0 ){ 
    which_dw = Task::OldDW; 
  } else { 
    which_dw = Task::NewDW; 
  }
  for ( int i = 0; i < _N * nSources; i++ ) {
    const VarLabel* tempLabel = d_nodeSources[i];
    tsk->requires( which_dw, tempLabel, Ghost::None, 0 );
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
                                     DataWarehouse* new_dw, 
                                     const int timeSubStep )
{
  //patch loop
  for (int p=0; p < patches->size(); p++) {
    Ghost::GhostType  gn  = Ghost::None;
    
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex();
    
    constCCVariable<double> volFrac;
    old_dw->get( volFrac, volfrac_label, matlIndex, patch, gn, 0);

    //allocate/modify all the source terms
    vector<CCVariable<double>* > srcs;
    for ( int i = 0; i < nMoments*nSources; i++ ) {
      CCVariable<double>* tempCCVar = scinew CCVariable<double>;
      if (new_dw->exists(d_sourceLabels[i], matlIndex, patch) ) {
        new_dw->getModifiable( *tempCCVar, d_sourceLabels[i], matlIndex, patch );
      } else {
        new_dw->allocateAndPut( *tempCCVar, d_sourceLabels[i], matlIndex, patch );
      }
      srcs.push_back( tempCCVar );
    }
 
    std::vector <constCCVariable<double> > cqmomWeights( _N );
    std::vector <constCCVariable<double> > cqmomAbscissas( _N * M);
    std::vector <constCCVariable<double> > nodeSource( _N * nSources);
    
    //get weights and abscissas from dw
    int j = 0;
    for (ArchesLabel::WeightMap::iterator iW = d_fieldLabels->CQMOMWeights.begin(); iW != d_fieldLabels->CQMOMWeights.end(); ++iW) {
      const VarLabel* tempLabel = iW->second;
      if (new_dw->exists( tempLabel, matlIndex, patch) ) {
        new_dw->get( cqmomWeights[j], tempLabel, matlIndex, patch, gn, 0 );
      } else {
        old_dw->get( cqmomWeights[j], tempLabel, matlIndex, patch, gn, 0 );
      }
      j++;
    }
    
    j = 0;
    for (ArchesLabel::AbscissaMap::iterator iA = d_fieldLabels->CQMOMAbscissas.begin(); iA != d_fieldLabels->CQMOMAbscissas.end(); ++iA) {
      const VarLabel* tempLabel = iA->second;
      if (new_dw->exists( tempLabel, matlIndex, patch) ) {
        new_dw->get( cqmomAbscissas[j], tempLabel, matlIndex, patch, gn, 0 );
      } else {
        old_dw->get( cqmomAbscissas[j], tempLabel, matlIndex, patch, gn, 0 );
      }
      j++;
    }
    
    DataWarehouse* which_dw;
    if ( timeSubStep == 0 ){
      which_dw = old_dw;
    } else {
      which_dw = new_dw;
    }
    for ( int i = 0; i < _N * nSources; i++ ) {
      const VarLabel* tempLabel = d_nodeSources[i];
      which_dw->get( nodeSource[i], tempLabel, matlIndex, patch, gn, 0);
    }

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      
      std::vector<double> temp_weights ( _N, 0.0 );
      std::vector<double> temp_abscissas ( _N * M, 0.0 );
      std::vector<double> temp_sources ( _N * nSources, 0.0 );
      
      //put all the weigths, abscissas and sources in temp vectors
      for (int i = 0; i < _N; i++) {
        temp_weights[i] = cqmomWeights[i][c];
      }
      
      for (int i = 0; i < _N * M; i++ ) {
        temp_abscissas[i] = cqmomAbscissas[i][c];
      }
      
      for (int i = 0; i < _N * nSources; i++ ) {
        temp_sources[i] = nodeSource[i][c];
      }
      
      if (volFrac[c] > 0.0 ) { //if not flow cell set src = 0
        for ( int s = 0; s < nSources; s++ ) {
          
          for ( int n = 0; n < nMoments; n++ ) {
            MomentVector temp_moment_index = momentIndexes[n];
        
            if ( temp_moment_index[ nIC[s] ] != 0 ) { //if this is a zeorth moment the coefficient of the term = 0
              double sum = 0.0;
              for ( int i = 0; i < _N; i++ ) { //loop each quad node
                double product = 1.0;
                for (int m = 0; m < M; m++ ) { //loop each internal coordinate
                  if ( m != nIC[s] ) {
                    for ( int j = 0; j < temp_moment_index[m]; j++ ) {
                      product *= temp_abscissas[i + _N * m];
                    }
                  } else {
                    for ( int j = 0; j < temp_moment_index[m]-1; j++ ) {
                      product *= temp_abscissas[i + _N * m];
                    }
                  }
                }
                product *= temp_weights[i] * temp_sources[i + s * _N];
                sum += product;
              }
              (*srcs[n + nMoments * s])[c] = temp_moment_index[nIC[s]] * sum;;
            } else {
              (*srcs[n + nMoments * s])[c] = 0.0;
            }
          }
        }
      } else {
        for (int i = 0; i < nMoments*nSources; i++ ) {
          (*srcs[i])[c] = 0.0;
        }
      }
    } //cell loop
    
    for ( int i = 0; i < nMoments*nSources; i++ ) { //clean up pointers
      delete srcs[i];
    }
  } //patch loop
}
