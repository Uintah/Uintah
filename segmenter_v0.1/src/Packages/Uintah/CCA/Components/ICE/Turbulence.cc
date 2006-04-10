
#include <Packages/Uintah/CCA/Components/ICE/Turbulence.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Material.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/Labels/ICELabel.h>
#include <Core/Geometry/IntVector.h>

using namespace Uintah;

Turbulence::Turbulence()
{
}

Turbulence::Turbulence(ProblemSpecP& ps, SimulationStateP& sharedState)
  : d_sharedState(sharedState)
{
  for (ProblemSpecP child = ps->findBlock("FilterScalar"); child != 0;
       child = child->findNextBlock("FilterScalar")) {
    FilterScalar* s = new FilterScalar;
    child->get("name", s->name);

    s->matl = sharedState->parseAndLookupMaterial(child, "material");
    vector<int> m(1);
    m[0] = s->matl->getDWIndex();
    s->matl_set = new MaterialSet();
    s->matl_set->addAll(m);
    s->matl_set->addReference();

    s->scalar = VarLabel::create(s->name, CCVariable<double>::getTypeDescription());
    s->scalarVariance = VarLabel::create(s->name+"-variance", CCVariable<double>::getTypeDescription());
    filterScalars.push_back(s);
  }
}

Turbulence::~Turbulence()
{
  for(int i=0;i<static_cast<int>(filterScalars.size());i++){
    FilterScalar* s = filterScalars[i];
    VarLabel::destroy(s->scalar);
    VarLabel::destroy(s->scalarVariance);
    delete s;
  }
}


/* ---------------------------------------------------------------------
  Function~  callTurb
  Purpose~ Call turbulent subroutines
  -----------------------------------------------------------------------  */  
void Turbulence::callTurb(DataWarehouse* new_dw,
                          const Patch* patch,
                          const CCVariable<Vector>& vel_CC,
                          const CCVariable<double>& rho_CC,
                          const int indx,
                          ICELabel* lb,
                          SimulationStateP&  d_sharedState,
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
    
  computeTurbViscosity(new_dw,patch,vel_CC,uvel_FC,vvel_FC,
                       wvel_FC,rho_CC,indx,d_sharedState,turb_viscosity);
    
  setBC(turb_viscosity, "zeroNeumann",  patch, d_sharedState, indx, new_dw);
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

  double maxvis = 0;
  double maxturb = 0;
  double maxtot = 0;
  for(CellIterator iter = iterPlusGhost; !iter.done(); iter++) {  
    IntVector c = *iter;    
    if(tot_viscosity[c] > maxvis)
      maxvis = tot_viscosity[c];
    tot_viscosity[c] += turb_viscosity[c];         
    if(turb_viscosity[c] > maxturb)
      maxturb = turb_viscosity[c];
    if(tot_viscosity[c] > maxtot)
      maxtot = tot_viscosity[c];
  } 
  //cerr << "Maximum viscosity: " << maxvis << ", max turb=" << maxturb << ", max total=" << maxtot << '\n';
}
