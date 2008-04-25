//----- CO2RateSrc.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/CO2RateSrc.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/ArchesMaterial.h>
#include <Packages/Uintah/CCA/Components/Arches/TimeIntegratorLabel.h>
#include <Packages/Uintah/CCA/Components/Arches/CellInformationP.h>
#include <Packages/Uintah/Core/Grid/Variables/PerPatch.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/CCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Exceptions/VariableNotFoundInGrid.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>

using namespace std;
using namespace Uintah;

//****************************************************************************
// Interface constructor for CO2RateSrc
//****************************************************************************
CO2RateSrc::CO2RateSrc(const ArchesLabel* label, 
		       		   const MPMArchesLabel* MAlb,
                       const VarLabel* d_src_label):
                       ExtraScalarSrc(label, MAlb, d_src_label)
{
}

//****************************************************************************
// Destructor
//****************************************************************************
CO2RateSrc::~CO2RateSrc()
{
}

//****************************************************************************
// Problem Setup 
//****************************************************************************
void 
CO2RateSrc::problemSetup(const ProblemSpecP& params)
{

	ProblemSpecP db = params;
	//Get the name of what to look for in the table
	// we "require" this because this source is specifically 
	// designed for a table-read source term.
	db->require("tableName", d_tableName);

	//Initialize
 	setTableIndex(-1);

	//warning
	cout << "** WARNING! **\n";
	cout << "   The CO2Rate Source term requires that carbon_balance_es be set to true! \n";

}
//****************************************************************************
// Schedule source computation
//****************************************************************************
void
CO2RateSrc::sched_addExtraScalarSrc(SchedulerP& sched, 
                                    const PatchSet* patches,
				   				    const MaterialSet* matls,
				   				    const TimeIntegratorLabel* timelabels)
{
  
  string taskname =  "CO2RateSrc::addExtraScalarSrc" +
      	              timelabels->integrator_step_name+
                      d_scalar_nonlin_src_label->getName();
  
  Task* tsk = scinew Task(taskname, this,
      		    &CO2RateSrc::addExtraScalarSrc,
      		    timelabels);

  //variables needed:
  tsk->modifies(d_scalar_nonlin_src_label);
  tsk->requires(Task::NewDW, d_lab->d_co2RateLabel, 
  		Ghost::None, Arches::ZEROGHOSTCELLS);

  //add the task:
  sched->addTask(tsk, patches, matls);
  
}
//****************************************************************************
// Actual source computation 
//****************************************************************************
void 
CO2RateSrc::addExtraScalarSrc(const ProcessorGroup* pc,
				      const PatchSubset* patches,
					  const MaterialSubset*,
				      DataWarehouse*,
				      DataWarehouse* new_dw,
				      const TimeIntegratorLabel* timelabels)
{
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int archIndex = 0; // only one arches material
    int matlIndex = d_lab->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> scalarNonlinSrc;
    new_dw->getModifiable(scalarNonlinSrc, d_scalar_nonlin_src_label,
                          matlIndex, patch);

	//going to estimate volume for all cells.
	//this will need to be fixed when going to stretched meshes
	Vector dx = patch->dCell();
	double vol = dx.x()*dx.y()*dx.z();

    constCCVariable<double> CO2rate;
    new_dw->get(CO2rate, d_lab->d_co2RateLabel, 
	matlIndex, patch, Ghost::None, Arches::ZEROGHOSTCELLS);

	for (CellIterator iter=patch->getCellIterator__New(); !iter.done(); iter++){

		scalarNonlinSrc[*iter] += CO2rate[*iter]*vol*44000; //44000 = conversion from mol/cm^3/s to kg/m^3/s
	
	}
  }
}
