/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

//----- PCTProps.cc --------------------------------------------------

// includes for Arches
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/ChemMix/PCTProps.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelFactory.h>

// includes for Uintah
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <Core/IO/UintahZlibUtil.h>


using namespace std;
using namespace Uintah;

//--------------------------------------------------------------------------- 
// Default Constructor 
//--------------------------------------------------------------------------- 
PCTProps::PCTProps( ArchesLabel* labels, const MPMArchesLabel* MAlabels ) :
  MixingRxnModel( labels, MAlabels )
{
  _boundary_condition = scinew BoundaryCondition_new( labels->d_sharedState->getArchesMaterial(0)->getDWIndex() ); 
}

//--------------------------------------------------------------------------- 
// Default Destructor
//--------------------------------------------------------------------------- 
PCTProps::~PCTProps()
{
  delete _boundary_condition; 
}

//--------------------------------------------------------------------------- 
// Problem Setup
//--------------------------------------------------------------------------- 
void
PCTProps::problemSetup( const ProblemSpecP& propertiesParameters )
{
  ProblemSpecP db_pct = db_pct->findBlock( "PCT" ); 

  // insert the input file interface here
  //

  problemSetupCommon( db_pct ); 

}

//--------------------------------------------------------------------------- 
// schedule get State
//--------------------------------------------------------------------------- 
void 
PCTProps::sched_getState( const LevelP& level, 
                          SchedulerP& sched, 
                          const TimeIntegratorLabel* time_labels, 
                          const bool initialize_me,
                          const bool with_energy_exch, 
                          const bool modify_ref_den )

{
  string taskname = "PCTProps::getState"; 
  Ghost::GhostType  gn = Ghost::None;

  Task* tsk = scinew Task(taskname, this, &PCTProps::getState, time_labels, initialize_me, with_energy_exch, modify_ref_den );

  // dependent variables
  if ( initialize_me ) {

    for ( MixingRxnModel::VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ) {
      tsk->computes( i->second ); 
    }

    tsk->computes( d_lab->d_drhodfCPLabel ); // I don't think this is used anywhere...maybe in coldflow? 
    if (d_MAlab)
      tsk->computes( d_lab->d_densityMicroLabel ); 

  } else {

    for ( MixingRxnModel::VarMap::iterator i = d_dvVarMap.begin(); i != d_dvVarMap.end(); ++i ) {
      tsk->modifies( i->second ); 
    }

    tsk->modifies( d_lab->d_drhodfCPLabel ); // I don't think this is used anywhere...maybe in coldflow? 
    if (d_MAlab)
      tsk->modifies( d_lab->d_densityMicroLabel ); 

  }

  // other variables 
  tsk->modifies( d_lab->d_densityCPLabel );  // lame .... fix me
  if ( modify_ref_den ) {
    tsk->computes(time_labels->ref_density); 
  }
  tsk->requires( Task::NewDW, d_lab->d_volFractionLabel, gn, 0 ); 
  tsk->requires( Task::NewDW, d_lab->d_cellTypeLabel, gn, 0 ); 

  // for inert mixing 
  for ( InertMasterMap::iterator iter = d_inertMap.begin(); iter != d_inertMap.end(); iter++ ){ 
    const VarLabel* label = VarLabel::find( iter->first ); 
    tsk->requires( Task::NewDW, label, gn, 0 ); 
  } 

  sched->addTask( tsk, level->eachPatch(), d_lab->d_sharedState->allArchesMaterials() ); 
}

//--------------------------------------------------------------------------- 
// get State
//--------------------------------------------------------------------------- 
void 
PCTProps::getState( const ProcessorGroup* pc, 
                                 const PatchSubset* patches, 
                                 const MaterialSubset* matls, 
                                 DataWarehouse* old_dw, 
                                 DataWarehouse* new_dw, 
                                 const TimeIntegratorLabel* time_labels, 
                                 const bool initialize_me, 
                                 const bool with_energy_exch, 
                                 const bool modify_ref_den )
{
  for (int p=0; p < patches->size(); p++){

    Ghost::GhostType gn = Ghost::None; 
    const Patch* patch = patches->get(p); 
    int archIndex = 0; 
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    constCCVariable<double> eps_vol; 
    constCCVariable<int> cell_type; 
    new_dw->get( eps_vol, d_lab->d_volFractionLabel, matlIndex, patch, gn, 0 ); 
    new_dw->get( cell_type, d_lab->d_cellTypeLabel, matlIndex, patch, gn, 0 ); 


    // dependent variables:
    CCVariable<double> mpmarches_denmicro; 

    if ( initialize_me ) {

      // others: 
      CCVariable<double> drho_df; 
      new_dw->allocateAndPut( drho_df, d_lab->d_drhodfCPLabel, matlIndex, patch ); 
      if (d_MAlab) {
        new_dw->allocateAndPut( mpmarches_denmicro, d_lab->d_densityMicroLabel, matlIndex, patch ); 
        mpmarches_denmicro.initialize(0.0);
      }

      drho_df.initialize(0.0);  // this variable might not be actually used anywhere and may just be polution  

    } else { 

      // others:
      CCVariable<double> drho_dw; 
      new_dw->getModifiable( drho_dw, d_lab->d_drhodfCPLabel, matlIndex, patch ); 
      if (d_MAlab) 
        new_dw->getModifiable( mpmarches_denmicro, d_lab->d_densityMicroLabel, matlIndex, patch ); 
    }

    // for inert mixing 
    StringToCCVar inert_mixture_fractions; 
    inert_mixture_fractions.clear(); 
    for ( InertMasterMap::iterator iter = d_inertMap.begin(); iter != d_inertMap.end(); iter++ ){ 
      const VarLabel* label = VarLabel::find( iter->first ); 
      constCCVariable<double> variable; 
      new_dw->get( variable, label, matlIndex, patch, gn, 0 ); 
      ConstVarContainer container; 
      container.var = variable; 

      inert_mixture_fractions.insert( std::make_pair( iter->first, container) ); 

    } 

    CCVariable<double> arches_density; 
    new_dw->getModifiable( arches_density, d_lab->d_densityCPLabel, matlIndex, patch ); 

    // Go through the patch and populate the requested state variables
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

      IntVector c = *iter; 

      //1) get dependent variable
      //2) multiply by volume fraction 
      //3) mix in any inert material if needed

    }

    // set boundary property values: 
    vector<Patch::FaceType> bf;
    vector<Patch::FaceType>::const_iterator bf_iter;
    patch->getBoundaryFaces(bf);

    // Loop over all boundary faces on this patch
    for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){

      Patch::FaceType face = *bf_iter; 
      IntVector insideCellDir = patch->faceDirection(face);

      int numChildren = patch->getBCDataArray(face)->getNumberChildren(matlIndex);
      for (int child = 0; child < numChildren; child++){

        std::vector<double> iv; 
        Iterator nu;
        Iterator bound_ptr; 

        std::vector<double> bc_values;
          
        //_______________________________________
        //correct for solid wall temperatures
        //NEED TO ADD This
      }
    }

    // reference density modification 
    if ( modify_ref_den ) {

      double den_ref = 0.0;

      if (patch->containsCell(d_ijk_den_ref)) {

        den_ref = arches_density[d_ijk_den_ref];
        cerr << "Modified reference density to: density_ref = " << den_ref << endl;

      }
      new_dw->put(sum_vartype(den_ref),time_labels->ref_density);
    }
  }
}
//---------------------------
double 
PCTProps::getTableValue( std::vector<double> iv, std::string variable )
{
  //get a variable by passing the independent variables
}

//---------------------------
double 
PCTProps::getTableValue( std::vector<double> iv, std::string depend_varname, 
                         MixingRxnModel::StringToCCVar inert_mixture_fractions, 
                         IntVector c )
{ 
  //get a variable by passing the independent variables and the inerts and a specific location 
}
//---------------------------
double 
PCTProps::getTableValue( std::vector<double> iv, std::string depend_varname, 
                                      MixingRxnModel::doubleMap inert_mixture_fractions )
{ 
  //get a variable by passing the indep. vars. and single inert values
}

// --- look into these below this line --------------------------
//
void PCTProps::tableMatching(){ 
  // Match the requested dependent variables with their table index:
}


//ignore below this line----------------------------

//--------------------------------------------------------------------------- 
// Old Table Hack -- to be removed with Properties.cc
//--------------------------------------------------------------------------- 
 void 
PCTProps::oldTableHack( const InletStream& inStream, Stream& outStream, bool calcEnthalpy, const string bc_type )
{
  //NOTHING TO BE DONE HERE
}
//--------------------------------------------------------------------------- 
// schedule Dummy Init
//--------------------------------------------------------------------------- 
  void 
PCTProps::sched_dummyInit( const LevelP& level, 
    SchedulerP& sched )

{
  //NOTHING TO BE DONE HERE
}

//--------------------------------------------------------------------------- 
// Dummy Init
//--------------------------------------------------------------------------- 
void 
PCTProps::dummyInit( const ProcessorGroup* pc, 
    const PatchSubset* patches, 
    const MaterialSubset* matls, 
    DataWarehouse* old_dw, 
    DataWarehouse* new_dw )
{
  //NOTHING TO BE DONE HERE
}
