//---------------------------- DQMOM.cc ----------------------

#include <CCA/Components/SpatialOps/DQMOM.h>

#include <CCA/Components/SpatialOps/SpatialOps.h>
#include <CCA/Components/SpatialOps/Fields.h>
#include <CCA/Components/SpatialOps/LU.h>
#include <CCA/Components/SpatialOps/TransportEqns/EqnFactory.h>
#include <CCA/Components/SpatialOps/TransportEqns/EqnBase.h>
#include <CCA/Components/SpatialOps/TransportEqns/DQMOMEqn.h>

#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Thread/Time.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Exceptions/UintahPetscError.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>

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
DQMOM::problemSetup( const ProblemSpecP& params, DQMOMEqnFactory& eqn_factory )
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

  // This for block is not necessary if the map includes weighted abscissas/internal coordinats 
  // in the same order as is given in the input file (b/c of the moment indexes)
  // Reason: this block puts the labels in the same order as the input file, so the moment indexes match up OK
  //DQMOMEqnFactory& eqn_factory = DQMOMEqnFactory::self();
  //DQMOMEqnFactory& eqn_factory = DQMOMEqnFactory::self();
  N = eqn_factory.get_quad_nodes();
  
  for (int alpha = 0; alpha < N; ++alpha){
    std::string wght_name = "w_qn";
    std::string node;
    std::stringstream out;
    out << alpha;
    node = out.str();
    wght_name += node;
    EqnBase& temp_weightEqnE = eqn_factory.retrieve_scalar_eqn( wght_name );
    DQMOMEqn& temp_weightEqnD = dynamic_cast<DQMOMEqn&>(temp_weightEqnE);
    weightEqns.push_back( temp_weightEqnD );
  }

  N_xi = 0;
  for (ProblemSpecP db_ic = db->findBlock("Ic"); db_ic != 0; db_ic = db_ic->findNextBlock("Ic") ) {
    std::string ic_name;
    db_ic->getAttribute("label", ic_name);
    for (int alpha = 0; alpha < N; alpha++) {
      std::string final_name = ic_name + "_qn";
      std::string node;
      std::stringstream out;
      out << alpha;
      node = out.str();
      final_name += node;
      EqnBase& temp_weightedAbscissaEqnE = eqn_factory.retrieve_scalar_eqn( final_name );
      DQMOMEqn& temp_weightedAbscissaEqnD = dynamic_cast<DQMOMEqn&>(temp_weightedAbscissaEqnE);
      weightedAbscissaEqns.push_back( temp_weightedAbscissaEqnD );
    }
    N_xi = N_xi + 1;
  }

}



// **********************************************
// sched_setLinearSystem()
// **********************************************
/*
void
DQMOM::sched_setLinearSystem()
{
}
*/

// **********************************************
// setLinearSystem()
// **********************************************
/*
void
DQMOM::setLinearSystem()
{
}
*/



// **********************************************
// sched_solveLinearSystem
// **********************************************
void
DQMOM::sched_solveLinearSystem( const LevelP& level, SchedulerP& sched )
{
  string taskname = "DQMOM::solveLinearSystem";
  Task* tsk = scinew Task(taskname, this, &DQMOM::solveLinearSystem);

  for (vector<DQMOMEqn>::iterator iEqn = weightEqns.begin(); iEqn != weightEqns.end(); ++iEqn) {
    const VarLabel* tempLabel;
    tempLabel = iEqn->getTransportEqnLabel();
    cout << "The linear system modifies weight: " << tempLabel << endl;
    tsk->modifies( tempLabel );
  }
  for (vector<DQMOMEqn>::iterator iEqn = weightedAbscissaEqns.begin(); iEqn != weightedAbscissaEqns.end(); ++ iEqn) {
    const VarLabel* tempLabel;
    tempLabel = iEqn->getTransportEqnLabel();
    cout << "The linear system modifies weighted abscissa: " << tempLabel << endl;
    tsk->modifies( tempLabel );
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
/*
  for (vector<MomentVector>::iterator iMoments = momentIndexes.begin(); iMoments != momentIndexes.end(); ++iMoments ) {
    MomentVector thisMoment = iMoments;


    // weights
    for (unsigned int alpha = 1; alpha <= N; ++alpha) {
      int prefix = 1;
      double product = 1;
      for (unsigned int i = 1; i <= thisMoment.size(); ++i) {
        prefix = prefix - (thisMoment[i]);
        product = product*( (  weightedAbscissaEqns[(i-1)*alpha + alpha].getTransportEqnLabel() / weightEqns[alpha].getTransportEqnLabel() )^(thisMoment[i]) );
      }
      // set values here
    } //end weights matrix

    // weighted abscissas
    for (unsigned int j = 1; j <= N_xi; ++j) {
      double prefix = 1;
      double product = 1;
      double productB = 1;
      for (unsigned int alpha = 1; alpha <= N; ++alpha) {
        // A_j+1 matrix:
        prefix = (1 - thisMoment[j])*( weightedAbscissaEqns[(j-1)*alpha + alpha].getTransportEqnLabel() / weightEqns[alpha].getTransportEqnLabel() )^(thisMoment[j] - 1);
        product = 1;
        // B matrix:
        productB = weightEqns[alpha];
        productB = productB*( -(thisMoment[j])*( (weightedAbscissaEqns[(j-1)*alpha + alpha])^(thisMoment[j] - 1) ) );
        
        for (unsigned int n = 1; n <= N_xi; ++n) {
          if (n != j) {
            // A_j+1 matrix:
            product = product*( ( weightedAbscissaEqns[(n-1)*alpha + alpha] / weightEqns[alpha] )^(thisMoment[n]) );
            // B matrix:
            // FIX THIS!!! NEED SOURCES HERE TOO
            productB = productB*( ( weightedAbscissaEqns[(n-1)*alpha + alpha] / weightEqns[alpha] )^(thisMoment[n]) );
          }
        }
        // this is where I stopped
        // set values here
      }
    } //end weighted abscissa + B matrix

  } // end for each moment
*/
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

