#include <Packages/Uintah/CCA/Components/ICE/TurbulenceFactory.h>
#include <Packages/Uintah/CCA/Components/ICE/SmagorinskyModel.h>
#include <Packages/Uintah/CCA/Components/ICE/DynamicModel.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/Core/Grid/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/SFCZVariable.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Core/Geometry/IntVector.h>
#include <fstream>
#include <iostream>
#include <string>

using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah;

TurbulenceFactory::TurbulenceFactory()
{
}

TurbulenceFactory::~TurbulenceFactory()
{
}

Turbulence* TurbulenceFactory::create(ProblemSpecP& ps,
                                      bool& d_Turb)
{
    ProblemSpecP child = ps->findBlock("turbulence");
   
    if(child){
      d_Turb = true;
      std::string turbulence_model;
      if(!child->getAttribute("model",turbulence_model))
        throw ProblemSetupException("No model for turbulence"); 
    
      if (turbulence_model == "Smagorinsky") 
        return(scinew Smagorinsky_Model(child));    
      else if (turbulence_model == "Germano") 
        return(scinew DynamicModel(child));   
      else
        throw ProblemSetupException("Unknown turbulence model ("+turbulence_model+")");
    }
    return 0;
}
/* ---------------------------------------------------------------------
  Function~  callTurb
  Purpose~ Call turbulent subroutines
  -----------------------------------------------------------------------  */  
void TurbulenceFactory::callTurb(DataWarehouse* new_dw,
                                 const Patch* patch,
                                 const CCVariable<Vector>& vel_CC,
                                 const CCVariable<double>& rho_CC,
                                 const int indx,
                                 ICELabel* lb,
                                 SimulationStateP&  d_sharedState,
                                 Turbulence* d_turbulence,
                                 CCVariable<double>& tot_viscosity)
{
    Ghost::GhostType  gac = Ghost::AroundCells;
  
    constSFCXVariable<double> uvel_FC;
    constSFCYVariable<double> vvel_FC;
    constSFCZVariable<double> wvel_FC;
  
    CCVariable<double> turb_viscosity, turb_viscosity_copy;    
    new_dw->allocateTemporary(turb_viscosity, patch, gac, 1); 
    new_dw->allocateAndPut(turb_viscosity_copy,lb->turb_viscosity_CCLabel,indx, patch);
   
    turb_viscosity.initialize(0.0); 
    
    new_dw->get(uvel_FC,     lb->uvel_FCMELabel,            indx,patch,gac,3);  
    new_dw->get(vvel_FC,     lb->vvel_FCMELabel,            indx,patch,gac,3);  
    new_dw->get(wvel_FC,     lb->wvel_FCMELabel,            indx,patch,gac,3);
    
    d_turbulence->computeTurbViscosity(new_dw,patch,vel_CC,uvel_FC,vvel_FC,
                                       wvel_FC,rho_CC,indx,d_sharedState,turb_viscosity);

    setBC(turb_viscosity,    "zeroNeumann",  patch, d_sharedState, indx);
    // make copy of turb_viscosity for visualization.
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;    
             turb_viscosity_copy[c] = turb_viscosity[c];         
     }
    //__________________________________
    //  At patch boundaries you need to extend
    // the computational footprint by one cell in ghostCells
    CellIterator iter = patch->getCellIterator();
    CellIterator iterPlusGhost = patch->addGhostCell_Iter(iter,1);
  
    for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
      IntVector c = *iter;    
      tot_viscosity[c] += turb_viscosity[c];         
    } 
          

}
