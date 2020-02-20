/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

//----- PCProperties.cc --------------------------------------------------

// includes for Arches
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>
#include <CCA/Components/Arches/ChemMix/PCProperties.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelBase.h>
#include <CCA/Components/Arches/PropertyModels/PropertyModelFactory.h>

// includes for Uintah
#include <Core/Grid/BoundaryConditions/BCUtils.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>

using namespace std;
using namespace Uintah;

//---------------------------------------------------------------------------
// Default Constructor
//---------------------------------------------------------------------------
PCProperties::PCProperties( MaterialManagerP& materialManager ) :
  MixingRxnModel( materialManager )
{}

//---------------------------------------------------------------------------
// Default Destructor
//---------------------------------------------------------------------------
PCProperties::~PCProperties()
{}

//---------------------------------------------------------------------------
// Problem Setup
//---------------------------------------------------------------------------
void
PCProperties::problemSetup( const ProblemSpecP& db )
{
  ProblemSpecP db_pcProps = db;

  // A call to the common problem setup which
  // sets some common label/name flags
  // (e.g., m_densityLabel, m_volFractionLabel, _temperature_label_name )
  problemSetupCommon( db, this );

}

//---------------------------------------------------------------------------
// schedule get State
//---------------------------------------------------------------------------
  void
PCProperties::sched_getState( const LevelP& level,
    SchedulerP& sched,
    const int time_substep,
    const bool initialize_me,
    const bool modify_ref_den )
{

  string taskname = "PCProperties::getState";
  Task* tsk = scinew Task(taskname, this, &PCProperties::getState, time_substep, initialize_me, modify_ref_den );


  sched->addTask( tsk, level->eachPatch(), m_materialManager->allMaterials( "Arches" ) );

}

//---------------------------------------------------------------------------
// get State
//---------------------------------------------------------------------------
void
PCProperties::getState( const ProcessorGroup* pc,
    const PatchSubset* patches,
    const MaterialSubset* matls,
    DataWarehouse* old_dw,
    DataWarehouse* new_dw,
    const int time_substep,
    const bool initialize_me,
    const bool modify_ref_den )
{
  for (int p=0; p < patches->size(); p++){

    //const Patch* patch = patches->get(p);

  }
}

double
PCProperties::getTableValue( std::vector<double>, std::string ){
  return -99.0; 
}
