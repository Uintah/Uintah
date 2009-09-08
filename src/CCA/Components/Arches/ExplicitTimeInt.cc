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

  string time_order; 
  ex_db->getAttribute("order", time_order); 

  if (time_order == "first"){
    
    d_alpha[0] = 0.0;
    d_alpha[1] = 0.0;
    d_alpha[2] = 0.0;

    d_beta[0]  = 1.0;
    d_beta[1]  = 0.0;
    d_beta[2]  = 0.0;

  }
  else if (time_order == "second") {

    d_alpha[0]= 0.0;
    d_alpha[1]= 0.5;
    d_alpha[2]= 0.0;

    d_beta[0]  = 1.0;
    d_beta[1]  = 0.5;
    d_beta[2]  = 0.0;

  }
  else if (time_order == "third") {

    d_alpha[0] = 0.0;
    d_alpha[1] = 0.75;
    d_alpha[2] = 1.0/3.0;

    d_beta[0]  = 1.0;
    d_beta[1]  = 0.25;
    d_beta[2]  = 2.0/3.0;

  }
  else
	    throw InvalidValue("Explicit time integration order must be one of: first, second, third!  Please fix input file.",__FILE__, __LINE__);		
}
