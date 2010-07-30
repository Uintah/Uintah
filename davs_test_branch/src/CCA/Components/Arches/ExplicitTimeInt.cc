#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Ports/Scheduler.h>
//#include <Core/Grid/Task.h>
//#include <Core/Grid/Box.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Components/Arches/ExplicitTimeInt.h>

//===========================================================================

using namespace Uintah;

ExplicitTimeInt::ExplicitTimeInt(const ArchesLabel* fieldLabels):
d_fieldLabels(fieldLabels)
{}

ExplicitTimeInt::~ExplicitTimeInt()
{}
//---------------------------------------------------------------------------
// Method: Problem setup
//---------------------------------------------------------------------------
void ExplicitTimeInt::problemSetup(const ProblemSpecP& params)
{
  ProblemSpecP ex_db = params->findBlock("ExplicitIntegrator");

  ex_db->getAttribute("order", d_time_order); 

  if (d_time_order == "first"){
    
    ssp_alpha[0] = 0.0;
    ssp_alpha[1] = 0.0;
    ssp_alpha[2] = 0.0;

    ssp_beta[0]  = 1.0;
    ssp_beta[1]  = 0.0;
    ssp_beta[2]  = 0.0;

    time_factor[0] = 1.0;
    time_factor[1] = 0.0;
    time_factor[2] = 0.0; 

  }
  else if (d_time_order == "second") {

    ssp_alpha[0]= 0.0;
    ssp_alpha[1]= 0.5;
    ssp_alpha[2]= 0.0;

    ssp_beta[0]  = 1.0;
    ssp_beta[1]  = 0.5;
    ssp_beta[2]  = 0.0;

    time_factor[0] = 1.0;
    time_factor[1] = 1.0;
    time_factor[2] = 0.0; 

  }
  else if (d_time_order == "third") {

    ssp_alpha[0] = 0.0;
    ssp_alpha[1] = 0.75;
    ssp_alpha[2] = 1.0/3.0;

    ssp_beta[0]  = 1.0;
    ssp_beta[1]  = 0.25;
    ssp_beta[2]  = 2.0/3.0;

    time_factor[0] = 1.0;
    time_factor[1] = 0.5;
    time_factor[2] = 1.0; 

  }
  else
	    throw InvalidValue("Explicit time integration order must be one of: first, second, third!  Please fix input file.",__FILE__, __LINE__);		
}
