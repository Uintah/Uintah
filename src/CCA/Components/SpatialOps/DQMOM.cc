#include <CCA/Components/SpatialOps/Fields.h>
#include <CCA/Components/SpatialOps/SourceTerms/SourceTermBase.h>
#include <CCA/Components/SpatialOps/LU.h>
#include <CCA/Components/SpatialOps/CoalModels/ModelFactory.h>
#include <CCA/Components/SpatialOps/SpatialOpsMaterial.h>
#include <CCA/Components/SpatialOps/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/SpatialOps/CoalModels/ModelBase.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Components/SpatialOps/DQMOM.h>
#include <Core/ProblemSpec/ProblemSpec.h>

//===========================================================================

using namespace Uintah;

DQMOM::DQMOM(const Fields* fieldLabels):
d_fieldLabels(fieldLabels)
{}

DQMOM::~DQMOM()
{}
//---------------------------------------------------------------------------
// Method: Problem setup
//---------------------------------------------------------------------------
void DQMOM::problemSetup(const ProblemSpecP& params)
{

  ProblemSpecP db = params; 
  unsigned int moments;
  unsigned int index_length;
  moments = 0;

  // obtain moment index vectors
  vector<int> temp_moment_index;
  for ( ProblemSpecP db_moments = db->findBlock("Moment");
        db_moments != 0; db_moments = db_moments->findNextBlock("Moment") ) {
    temp_moment_index.resize(0);
    db_moments->get("m", temp_moment_index);
    
    // put moment index into vector of moment indexes:
    momentIndexes.push_back(temp_moment_index);
    
    // keep track of total number of moments
    ++moments;

    index_length = temp_moment_index.size();
  }

 
  // This for block is not necessary if the map includes weighted abscissas/internal coordinats 
  // in the same order as is given in the input file (b/c of the moment indexes)
  // Reason: this block puts the labels in the same order as the input file, so the moment indexes match up OK
  
  DQMOMEqnFactory & eqn_factory = DQMOMEqnFactory::self();
  //ModelFactory & model_factory = ModelFactory::self();
  N_ = eqn_factory.get_quad_nodes();
 
  for( unsigned int alpha = 0; alpha < N_; ++alpha ) {
    string wght_name = "w_qn";
    string node;
    stringstream out;
    out << alpha;
    node = out.str();
    wght_name += node;
    // store equation:
    EqnBase& temp_weightEqnE = eqn_factory.retrieve_scalar_eqn( wght_name );
    DQMOMEqn& temp_weightEqnD = dynamic_cast<DQMOMEqn&>(temp_weightEqnE);
    weightEqns.push_back( &temp_weightEqnD );
  }

  N_xi = 0;
  for (ProblemSpecP db_ic = db->findBlock("Ic"); db_ic != 0; db_ic = db_ic->findNextBlock("Ic") ) {
    string ic_name;
    vector<string> modelsList;
    db_ic->getAttribute("label", ic_name);
    for( unsigned int alpha = 0; alpha < N_; ++alpha ) {
      string final_name = ic_name + "_qn";
      string node;
      stringstream out;
      out << alpha;
      node = out.str();
      final_name += node;
      // store equation:
      EqnBase& temp_weightedAbscissaEqnE = eqn_factory.retrieve_scalar_eqn( final_name );
      DQMOMEqn& temp_weightedAbscissaEqnD = dynamic_cast<DQMOMEqn&>(temp_weightedAbscissaEqnE);
      weightedAbscissaEqns.push_back( &temp_weightedAbscissaEqnD );
    }
    N_xi = N_xi + 1;
  }
  
  // Check to make sure number of total moments specified in input file is correct
  if ( moments != (N_xi+1)*N_ ) {
    cout << "ERROR:DQMOM:ProblemSetup: You specified " << moments << " moments, but you need " << (N_xi+1)*N_ << " moments." << endl;
    throw InvalidValue( "ERROR:DQMOM:ProblemSetup: The number of moments specified was incorrect! Need ",__FILE__,__LINE__);
  }

  // Check to make sure number of moment indices matches the number of internal coordinates
  if ( index_length != N_xi ) {
    cout << "ERROR:DQMOM:ProblemSetup: You specified " << index_length << " moment indices, but there are " << N_xi << " internal coordinates." << endl;
    throw InvalidValue( "ERROR:DQMOM:ProblemSetup: The number of moment indices specified was incorrect! Need ",__FILE__,__LINE__);
  }

}

// **********************************************
// sched_solveLinearSystem
// **********************************************
void
DQMOM::sched_solveLinearSystem( const LevelP& level, SchedulerP& sched, int timeSubStep )
{ 
  string taskname = "DQMOM::solveLinearSystem";
  Task* tsk = scinew Task(taskname, this, &DQMOM::solveLinearSystem);

  ModelFactory& model_factory = ModelFactory::self();

  for (vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin(); iEqn != weightEqns.end(); ++iEqn) {
    const VarLabel* tempLabel;
    tempLabel = (*iEqn)->getTransportEqnLabel();
    
    // require weights
    cout << "The linear system requires weight " << *tempLabel << endl;
    tsk->requires( Task::NewDW, tempLabel, Ghost::None, 0 );

    const VarLabel* sourceterm_label = (*iEqn)->getSourceLabel();
    if (timeSubStep == 0) {
      cout << "The linear system computes source term " << *sourceterm_label << endl;
      tsk->computes(sourceterm_label);
    } else {
      cout << "The linear system modifies source term " << *sourceterm_label << endl;
      tsk->modifies(sourceterm_label);
    }
 
  }
  
  for (vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin(); iEqn != weightedAbscissaEqns.end(); ++iEqn) {
    const VarLabel* tempLabel;
    tempLabel = (*iEqn)->getTransportEqnLabel();
    
    // require weighted abscissas
    cout << "The linear system requires weighted abscissa " << *tempLabel << endl;
    tsk->requires(Task::NewDW, tempLabel, Ghost::None, 0);

    // compute or modify source terms
    cout << " current eqn  = " << (*iEqn)->getEqnName() << endl;
    const VarLabel* sourceterm_label = (*iEqn)->getSourceLabel();
    if (timeSubStep == 0) {
      cout << "The linear system computes source term " << *sourceterm_label << endl;
      tsk->computes(sourceterm_label);
    } else {
      cout << "The linear system modifies source term " << *sourceterm_label << endl;
      tsk->modifies(sourceterm_label);
    }
    
    // require model terms
    vector<string> modelsList = (*iEqn)->getModelsList();
    for ( vector<string>::iterator iModels = modelsList.begin(); iModels != modelsList.end(); ++iModels ) {
      cout << "looking for model: " << (*iModels) << endl;
      ModelBase& model_base = model_factory.retrieve_model(*iModels);
      const VarLabel* model_label = model_base.getModelLabel();
      tsk->requires( Task::NewDW, model_label, Ghost::AroundCells, 1 );
    }
  }

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allSpatialOpsMaterials());

}

// **********************************************
// solveLinearSystem
// **********************************************
void
DQMOM::solveLinearSystem( const ProcessorGroup* pc,  
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw )
{
  cout << "Now entering DQMOM linear solver.\n";
  // patch loop
  for (int p=0; p < patches->size(); ++p) {
    const Patch* patch = patches->get(p);

    Ghost::GhostType  gn  = Ghost::None; 
    int matlIndex = 0;

    ModelFactory& model_factory = ModelFactory::self();

    // Cell iterator
    for ( CellIterator iter = patch->getCellIterator__New();
          !iter.done(); ++iter) {
  
      LU A( (N_xi+1)*N_, 1); // bandwidth doesn't matter b/c A is being stored densely
      vector<double> B( (N_xi+1)*N_, 0.0 );
      IntVector c = *iter;

      vector<double> weights;
      vector<double> weightedAbscissas;
      vector<double> models;

      // store weights
      for (vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin(); 
           iEqn != weightEqns.end(); ++iEqn) {
        constCCVariable<double> temp;
        const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();
        new_dw->get( temp, equation_label, matlIndex, patch, gn, 0);
        weights.push_back(temp[c]);
      }

      // store abscissas
      for (vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin();
           iEqn != weightedAbscissaEqns.end(); ++iEqn) {
        const VarLabel* equation_label = (*iEqn)->getTransportEqnLabel();
        constCCVariable<double> temp;
        new_dw->get(temp, equation_label, matlIndex, patch, gn, 0);
        weightedAbscissas.push_back(temp[c]);

        double runningsum = 0;
        vector<string> modelsList = (*iEqn)->getModelsList();
        for ( vector<string>::iterator iModels = modelsList.begin();
              iModels != modelsList.end(); ++iModels) {
          ModelBase& model_base = model_factory.retrieve_model(*iModels);
          const VarLabel* model_label = model_base.getModelLabel();
          constCCVariable<double> tempCCVar;
          new_dw->get(tempCCVar, model_label, matlIndex, patch, gn, 0);
          runningsum = runningsum + tempCCVar[c];
        }

        models.push_back(runningsum);
      }


      // construct AX=B
      for ( unsigned int k = 0; k < momentIndexes.size(); ++k) {
        MomentVector thisMoment = momentIndexes[k];
        
        // weights
        for ( unsigned int alpha = 0; alpha < N_; ++alpha) {
          double prefixA = 1;
          double productA = 1;
          for ( unsigned int i = 0; i < thisMoment.size(); ++i) {
            if (weights[alpha] != 0) {
              // Appendix C, C.9 (A1 matrix)
              prefixA = prefixA - (thisMoment[i]);
              productA = productA*( pow((weightedAbscissas[i*(N_)+alpha]/weights[alpha]),thisMoment[i]) );
            } else {
              prefixA = 0;
              productA = 0;
            }
          }
          A(k,alpha)=prefixA*productA;
        } //end weights matrix

        // weighted abscissas
        double totalsumS = 0;
        for( unsigned int j = 0; j < N_xi; ++j ) {
          double prefixA    = 1;
          double productA   = 1;
          
          double prefixS    = 1;
          double productS   = 1;
          double modelsumS  = 0;
          
          double quadsumS = 0;
          for( unsigned int alpha = 0; alpha < N_; ++alpha ) {
            if (weights[alpha] != 0) {
              // Appendix C, C.11 (A_j+1 matrix)
              prefixA = (thisMoment[j])*( pow((weightedAbscissas[j*(N_)+alpha]/weights[alpha]),(thisMoment[j]-1)) );
              productA = 1;

              // Appendix C, C.16 (S matrix)
              prefixS = -(thisMoment[j])*( pow((weightedAbscissas[j*(N_)+alpha]/weights[alpha]),(thisMoment[j]-1)));
              productS = 1;

              for (unsigned int n = 0; n < N_xi; ++n) {
                if (n != j) {
                  // A_j+1 matrix:
                  productA = productA*( pow( (weightedAbscissas[n*(N_)+alpha]/weights[alpha]), thisMoment[n] ));
                  // S matrix:
                  productS = productS*( pow( (weightedAbscissas[n*(N_)+alpha]/weights[alpha]), thisMoment[n] ));
                }
              }
            } else {
              prefixA = 0;
              productA = 0;
              productS = 0;
            }

            modelsumS = modelsumS - models[j*(N_)+alpha];

            A(k,(j+1)*N_ + alpha)=prefixA*productA;
            
            quadsumS = quadsumS + weights[alpha]*modelsumS*prefixS*productS;
          }//end quad nodes
          totalsumS = totalsumS + quadsumS;
        }//end int coords
        
        B[k] = totalsumS;
      } // end moments

      /*
      // Print out cell-by-cell matrix information
      // (Make sure and change your domain to have a small # of cells!)
      cout << "Cell " << c << endl;
      cout << endl;

      cout << "A matrix:" << endl;
      A.dump();
      cout << endl;

      cout << "B matrix:" << endl;
      for (vector<double>::iterator iB = B.begin(); iB != B.end(); ++iB) {
        cout << (*iB) << endl;
      }
      cout << endl;
      */

      A.decompose();
      A.back_subs( &B[0] );

      /*
      cout << "X matrix:" << endl;
      for (vector<double>::iterator iB = B.begin(); iB != B.end(); ++iB) {
        cout << (*iB) << endl;
      }
      cout << endl;
      */
 
      // set weight/weighted abscissa transport eqn source terms equal to results
      unsigned int z = 0;
      for (vector<DQMOMEqn*>::iterator iEqn = weightEqns.begin();
           iEqn != weightEqns.end(); iEqn++) {
        const VarLabel* source_label = (*iEqn)->getSourceLabel();
        CCVariable<double> tempCCVar;
        if (new_dw->exists(source_label, matlIndex, patch)) {
          new_dw->getModifiable(tempCCVar, source_label, matlIndex, patch);
        } else {
          new_dw->allocateAndPut(tempCCVar, source_label, matlIndex, patch);
        }

        tempCCVar[c] = B[z];
        ++z;
      }
  
      for (vector<DQMOMEqn*>::iterator iEqn = weightedAbscissaEqns.begin();
           iEqn != weightedAbscissaEqns.end(); ++iEqn) {
        const VarLabel* source_label = (*iEqn)->getSourceLabel();
        CCVariable<double> tempCCVar;
        if (new_dw->exists(source_label, matlIndex, patch)) {
          new_dw->getModifiable(tempCCVar, source_label, matlIndex, patch);
        } else {
          new_dw->allocateAndPut(tempCCVar, source_label, matlIndex, patch);
        }

        tempCCVar[c] = B[z];
        ++z;
      }
     
    }//end for cells

  }//end per patch
  cout << "Now leaving DQMOM linear solver.\n";
}
