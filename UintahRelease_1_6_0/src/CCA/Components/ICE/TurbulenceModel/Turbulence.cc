/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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


#include <CCA/Components/ICE/TurbulenceModel/Turbulence.h>
#include <CCA/Components/ICE/BoundaryCond.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Labels/ICELabel.h>
#include <Core/Geometry/IntVector.h>

using namespace Uintah;
using namespace std;

Turbulence::Turbulence()
{
}

Turbulence::Turbulence(ProblemSpecP& ps, SimulationStateP& sharedState)
  : d_sharedState(sharedState)
{
  for (ProblemSpecP child = ps->findBlock("FilterScalar"); child != 0;
       child = child->findNextBlock("FilterScalar")) {
    FilterScalar* s = scinew FilterScalar;
    child->get("name", s->name);
    
    
    s->matl = sharedState->parseAndLookupMaterial(child, "material");
    vector<int> m(1);
    m[0] = s->matl->getDWIndex();
    s->matl_set = scinew MaterialSet();
    s->matl_set->addAll(m);
    s->matl_set->addReference();

    s->scalar = VarLabel::create(s->name, CCVariable<double>::getTypeDescription());
    s->scalarVariance = VarLabel::create(s->name+"-variance", CCVariable<double>::getTypeDescription());
    filterScalars.push_back(s);
  }
}
//______________________________________________________________________
//
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
                          constCCVariable<Vector>& vel_CC,
                          constCCVariable<double>& rho_CC,
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
    
  computeTurbViscosity(new_dw, patch, lb, vel_CC,
                       uvel_FC, vvel_FC, wvel_FC, rho_CC, 
                       indx, d_sharedState, turb_viscosity );
    
  setBC(turb_viscosity, "zeroNeumann",  patch, d_sharedState, indx, new_dw);
  

  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells
  double maxvis  = 0;
  double maxturb = 0;
  double maxtot  = 0;
  
  int NGC =1;  // number of ghostCells
  for(CellIterator iter = patch->getCellIterator(NGC); !iter.done(); iter++) { 
    IntVector c = *iter;    
    if(tot_viscosity[c] > maxvis) {
      maxvis = tot_viscosity[c];
    }
    tot_viscosity[c] += turb_viscosity[c];         
    
    if(turb_viscosity[c] > maxturb) {
      maxturb = turb_viscosity[c];
    }
    if(tot_viscosity[c] > maxtot) {
      maxtot = tot_viscosity[c];
    }
  } 
  
  // make copy for visualization.
  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
    IntVector c = *iter;    
    turb_viscosity_copy[c] = tot_viscosity[c];         
  }
}


//______________________________________________________________________
//  Zero Neumann BC:
// This is a special version of setBC since it will be applied on the ghost cells
// of temporary allocated arrays (M_ij, S_ij, ...etc).

template<class T>
void Turbulence::setZeroNeumannBC_CC( const Patch* patch,
                                      CCVariable<T>& var,
                                      const int NGC)
{
  if(patch->hasBoundaryFaces() == false){
    return;
  }  
  
  // Iterate over the faces encompassing the domain
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);
  
  for( vector<Patch::FaceType>::const_iterator faceIter = bf.begin(); faceIter != bf.end(); ++faceIter ){
    Patch::FaceType face = *faceIter;
    
    //__________________________________
    //  find the iterator for this face
    int p_dir = patch->getFaceAxes(face)[0];  // principal direction
    
    //is this a plus face
    bool plusface=face%2;

    ASSERT(patch->getBCType(face)!=Patch::Neighbor);

    IntVector lowPt  = patch->getExtraCellLowIndex(NGC);
    IntVector highPt = patch->getExtraCellHighIndex(NGC);

    // select the face
    if( plusface ){
      //move low point to plus face
      lowPt[p_dir] = patch->getCellHighIndex()[p_dir];
    }
    else{
      //move high point to minus face
      highPt[p_dir] = patch->getCellLowIndex()[p_dir];
    }
    

    //__________________________________
    //  Apply BC    
    CellIterator iter( lowPt, highPt );

    IntVector oneCell = patch->faceDirection( face );
    for(;!iter.done(); iter++){
      IntVector c = *iter;
      IntVector adjCell = c - oneCell;
      var[c] = var[adjCell];
    }
  }
}
//______________________________________________________________________
// Explicit template instantiations:

template void Turbulence::setZeroNeumannBC_CC( const Patch* patch, CCVariable<double>& var, const int NGC);
template void Turbulence::setZeroNeumannBC_CC( const Patch* patch, CCVariable<Vector>& var, const int NGC);


