//---------------------------- DQMOM.cc ----------------------

#include <CCA/Components/SpatialOps/DQMOM.h>
#include <CCA/Components/SpatialOps/SpatialOps.h>
#include <CCA/Components/SpatialOps/Fields.h>
#include <CCA/Components/SpatialOps/LU.h>
#include <CCA/Components/SpatialOps/TransportEqns/EqnFactory.h>
#include <CCA/Components/SpatialOps/TransportEqns/EqnBase.h>
#include <CCA/Components/SpatialOps/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/SpatialOps/CoalModels/ModelFactory.h>
#include <CCA/Components/SpatialOps/CoalModels/ModelBase.h>
#include <CCA/Components/SpatialOps/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/SpatialOps/SourceTerms/SourceTermBase.h>

#include <CCA/Ports/Scheduler.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Thread/Time.h>

#include <math.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

// **********************************************
// Default constructor for DQMOM
// **********************************************
DQMOM::DQMOM( const Fields* fieldLabels ):d_fieldLabels(fieldLabels)
{ 
}

// **********************************************
// Default destructor for DQMOM
// **********************************************
DQMOM::~DQMOM()
{
}

// **********************************************
// problemSetup()
// **********************************************
void
DQMOM::problemSetup( const ProblemSpecP& params )
{
  //<DQMOM>
  //  <internal_coordinate name="blah">
  //    <model>blah1</model>
  //    <model>blah2</model>
  //  </internal_coordinate>
  //  <quadrature_nodes>5</quadrature_nodes>
  // <moment>[0,0,0,1]</moment>
  // <moment>[0,1,0,0]</moment>
  // <moment>[0,2,0,1]</moment>
  //            ...
  //</DQMOM>
  
  ProblemSpecP db = params->findBlock("DQMOM");

  unsigned int moments;
  moments = 0;

  // obtain moment index vectors
  for (ProblemSpecP db_moments = db->findBlock("moment");
       db_moments != 0; db_moments->findNextBlock("moment")) {
    // get moment index vector
    vector<int> temp_moment_index;
    db_moments->get("moment",temp_moment_index);
    
    // put moment index into map of moment indexes
    momentIndexes.push_back(temp_moment_index);
    
    // keep track of total # of moments
    ++moments;
  }

  // throw exception if moments != (N_xi+1)*N

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
    weightEqns.push_back( temp_weightEqnD );
  }

  N_xi = 0;
  for (ProblemSpecP db_ic = db->findBlock("Ic"); db_ic != 0; db_ic = db_ic->findNextBlock("Ic") ) {
    string ic_name;
    vector<string> modelsList;
    db_ic->getAttribute("label", ic_name);
    for( unsigned int alpha = 1; alpha <= N_; ++alpha ) {
      string final_name = ic_name + "_qn";
      string node;
      stringstream out;
      out << alpha;
      node = out.str();
      final_name += node;
      // store equation:
      EqnBase& temp_weightedAbscissaEqnE = eqn_factory.retrieve_scalar_eqn( final_name );
      DQMOMEqn& temp_weightedAbscissaEqnD = dynamic_cast<DQMOMEqn&>(temp_weightedAbscissaEqnE);
      weightedAbscissaEqns.push_back( temp_weightedAbscissaEqnD );
    }
    N_xi = N_xi + 1;
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

  d_timeSubStep = timeSubStep;
  SourceTermFactory& sourceterm_factory = SourceTermFactory::self();
  ModelFactory& model_factory = ModelFactory::self();

  for (vector<DQMOMEqn>::iterator iEqn = weightEqns.begin(); iEqn != weightEqns.end(); ++iEqn) {
    const VarLabel* tempLabel;
    tempLabel = iEqn->getTransportEqnLabel();
    
    // require weights
    cout << "The linear system requires weight " << tempLabel << endl;
    tsk->requires( Task::OldDW, tempLabel, Ghost::AroundCells, 1 );
  }
  
  for (vector<DQMOMEqn>::iterator iEqn = weightedAbscissaEqns.begin(); iEqn != weightedAbscissaEqns.end(); ++iEqn) {
    const VarLabel* tempLabel;
    tempLabel = iEqn->getTransportEqnLabel();
    
    // require weighted abscissas
    cout << "The linear system requires weighted abscissa " << tempLabel << endl;
    tsk->computes(tempLabel);

    if (d_timeSubStep == 0) {
      cout << "The linear system computes weighted abscissa: " << tempLabel << endl;
      tsk->computes(tempLabel);
    } else {
      cout << "The linear system modifies weighted abscissa: " << tempLabel << endl;
      tsk->modifies(tempLabel);
    }
    
    // compute or modify source terms
    SourceTermBase& sourceterm_base = sourceterm_factory.retrieve_source_term( iEqn->getEqnName() );
    const VarLabel* sourceterm_label = sourceterm_base.getSrcLabel();
    if (d_timeSubStep == 0) {
      cout << "The linear system computes source term " << sourceterm_label << endl;
      tsk->computes(sourceterm_label);
    } else {
      cout << "The linear system modifies source term " << sourceterm_label << endl;
      tsk->modifies(sourceterm_label);
    }
    
    // require model terms
    vector<string> modelsList = iEqn->getModelsList();
    for ( vector<string>::iterator iModels = modelsList.begin(); iModels != modelsList.end(); ++iModels ) {
      ModelBase& model_base = model_factory.retrieve_model(*iModels);
      const VarLabel* model_label = model_base.getModelLabel();
      tsk->requires( Task::OldDW, model_label, Ghost::AroundCells, 1 );
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
                          DataWarehouse* new_dw)
{
  vector< constCCVariable<double>* > weightCCVars;
  vector< constCCVariable<double>* > weightedAbscissaCCVars;
  vector< vector< constCCVariable<double>* > > weightedAbscissaModelCCVars;
  vector< CCVariable<double>* > sourceCCVars;
  
  SourceTermFactory& sourceterm_factory = SourceTermFactory::self();
  ModelFactory& model_factory = ModelFactory::self();

  // patch loop
  for (int p=0; p < patches->size(); ++p) {
    const Patch* patch = patches->get(p);

    Ghost::GhostType  gac = Ghost::AroundCells;
    int matlIndex = 0;

    // getModifiable/allocateAndPut calls:

    // loop over all weight equations
    for (vector<DQMOMEqn>::iterator iEqn = weightEqns.begin(); iEqn != weightEqns.end(); ++iEqn) {
      
      // transported variables are already allocated at beginning of timestep/exist in warehouse
      const VarLabel* equation_label = iEqn->getTransportEqnLabel();
      constCCVariable<double> equation_ccvar;
      
      new_dw->get( equation_ccvar, equation_label, matlIndex, patch, gac, 0 );
      weightCCVars.push_back(&equation_ccvar);
      
      SourceTermBase& source_base = sourceterm_factory.retrieve_source_term( iEqn->getEqnName() );
      const VarLabel* source_label = source_base.getSrcLabel();
      CCVariable<double> source_ccvar;
      // source terms are not allocated in new DW at beginning of timestep
      if (new_dw->exists(source_label, matlIndex, patch) ) {
        new_dw->getModifiable( source_ccvar, source_label, matlIndex, patch );
      } else {
        new_dw->allocateAndPut( source_ccvar, source_label, matlIndex, patch );
      }
      sourceCCVars.push_back(&source_ccvar);
    }//end for weight eqns


    // loop over all weighted abscissa eqns
    for (vector<DQMOMEqn>::iterator iEqn = weightedAbscissaEqns.begin(); 
         iEqn != weightedAbscissaEqns.end(); ++iEqn) {
      
      // transported variables already allcoated/exist in warehouse
      const VarLabel* equation_label = iEqn->getTransportEqnLabel();
      constCCVariable<double> equation_ccvar;
      new_dw->get( equation_ccvar, equation_label, matlIndex, patch, gac, 0 );
      weightedAbscissaCCVars.push_back(&equation_ccvar);

      SourceTermBase& source_base = sourceterm_factory.retrieve_source_term( iEqn->getEqnName() );
      const VarLabel* source_label = source_base.getSrcLabel();
      CCVariable<double> source_ccvar;
      if (new_dw->exists(source_label, matlIndex, patch) ) {
        new_dw->getModifiable( source_ccvar, source_label, matlIndex, patch );
      } else {
        new_dw->allocateAndPut( source_ccvar, source_label, matlIndex, patch );
      }
      sourceCCVars.push_back(&source_ccvar);

      vector< constCCVariable<double>* > eqnModels;
      vector<string> modelsList = iEqn->getModelsList();
      for ( vector<string>::iterator iModels = modelsList.begin();
            iModels != modelsList.end(); ++iModels) {
        // model terms have already been allocated/computed/exist in DW
        ModelBase& model_base = model_factory.retrieve_model(*iModels);
        const VarLabel* model_label = model_base.getModelLabel();
        constCCVariable<double> model_ccvar;
        new_dw->get( model_ccvar, model_label, matlIndex, patch, gac, 0 );
        eqnModels.push_back(&model_ccvar);
      }
      weightedAbscissaModelCCVars.push_back(eqnModels);
    }//end for weighted abscissa eqns


    // Cell iterator

    for ( CellIterator iter = patch->getCellIterator__New();
          !iter.done(); ++iter) {
      
      LU A( (N_xi+1)*N_, 1); //bandwidth doesn't matter b/c A is being stored densely
      vector<double> B( (N_xi+1)*N_, 0.0 );
      IntVector c = *iter;
 
      for ( unsigned int k = 1; k <= momentIndexes.size(); ++k) {
        MomentVector thisMoment = momentIndexes[k];
        
        // weights
        for ( unsigned int alpha = 1; alpha <= N_; ++alpha) {
          double prefixA = 1;
          double productA = 1;
          for ( unsigned int i = 1; i <= thisMoment.size(); ++i) {
            // Appendix C, C.9 (A1 matrix)
            prefixA = prefixA - (thisMoment[i]);
            double w_temp_doub = (*weightCCVars[alpha])[c];
            double wa_temp_doub = (*weightedAbscissaCCVars[(i-1)*alpha+alpha])[c];
            productA = productA*( pow((wa_temp_doub/w_temp_doub),thisMoment[i]) );
          }
          A(alpha,k)=prefixA*productA;
        } //end weights matrix

        // weighted abscissas
        double totalsumB = 0;
        for( unsigned int j = 1; j <= N_xi; ++j ) {
          double prefixA = 1;
          double productA = 1;
          double productB = 1;
          double modelsumB = 0;
          for( unsigned int alpha = 1; alpha <= N_; ++alpha ) {
            double w_temp_doub = (*weightCCVars[alpha])[c];
            double wa_temp_doub = (*weightedAbscissaCCVars[(j-1)*alpha+alpha])[c];
            // Appendix C, C.11 (A_j+1 matrix)
            prefixA = (thisMoment[j]-1)*( pow((wa_temp_doub/w_temp_doub),(thisMoment[j]-1)) );
            productA = 1;
            // Appendix C, C.16 (B matrix)
            productB = w_temp_doub;
            productB = productB*( -(thisMoment[j])*( pow((wa_temp_doub/w_temp_doub),(thisMoment[j]-1)) ) );
        
            for (unsigned int n = 1; n <= N_xi; ++n) {
              if (n != j) {
                double w_temp_doub = (*weightCCVars[alpha])[c];
                double wa_temp_doub = (*weightedAbscissaCCVars[(n-1)*alpha+alpha])[c];
                // A_j+1 matrix:
                productA = productA*( pow((wa_temp_doub/w_temp_doub),thisMoment[n]) );
                // B matrix:
                productB = productB*( pow((wa_temp_doub/w_temp_doub),thisMoment[n]) );
              }
            }

            // for given quad node, get model source term
            for (unsigned int z = 1; z <= weightedAbscissaModelCCVars[j].size(); ++z) {
              modelsumB = modelsumB + (*weightedAbscissaModelCCVars[j][z])[c];
            }
             
            A(j*alpha + alpha, k)=prefixA*productA;

          }//end quad nodes
          totalsumB = totalsumB + (productB)*(modelsumB);
        }//end int coords
        B[k] = totalsumB;
      } // end moments

      A.decompose();
      A.back_subs( &B[0] );
      // set sources equal to result
      for (unsigned int z = 1; z <= sourceCCVars.size(); ++z) {
        (*sourceCCVars[z])[c]=B[z];
      }

    }//end for cells

  }//end per patch

}

// **********************************************
// destroyLinearSystem
// **********************************************
void
DQMOM::destroyLinearSystem()
{
}

// **********************************************
// finalizeSolver
// **********************************************
/*
void
DQMOM::finalizeSolver()
{
}
*/

