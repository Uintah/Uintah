/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#include <CCA/Components/ICE/SmagorinskyModel.h>
#include <CCA/Components/ICE/BoundaryCond.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Math/CubeRoot.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Patch.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
static DebugStream cout_doing("ICE_DOING_COUT", false);

Smagorinsky_Model::Smagorinsky_Model(ProblemSpecP& ps,
                                     SimulationStateP& sharedState)
  : Turbulence(ps, sharedState)
{
  //__________________________________
  //typically filter_width=grid spacing(uniform) for implicit filter.
  ps->require("model_constant",d_model_constant);
  ps->require("filter_width",d_filter_width);

  //  ps->require("turb_Pr",d_turbPr);  
  
}

Smagorinsky_Model::Smagorinsky_Model()
{
}

Smagorinsky_Model::~Smagorinsky_Model()
{
}

/* ---------------------------------------------------------------------
  Function~  computeTurbViscosity
  Purpose~ Calculate the turbulent viscosity
  -----------------------------------------------------------------------  */
void Smagorinsky_Model::computeTurbViscosity(DataWarehouse* new_dw,
                                            const Patch* patch,
                                            const CCVariable<Vector>& /*vel_CC*/,
                                            const SFCXVariable<double>& uvel_FC,
                                            const SFCYVariable<double>& vvel_FC,
                                            const SFCZVariable<double>& wvel_FC,
                                            const CCVariable<double>& rho_CC,
                                            const int indx,
                                            SimulationStateP&  d_sharedState,
                                            CCVariable<double>& turb_viscosity)
{
  //__________________________________
  //implicit filter, filter_width=(dx*dy*dz)**(1.0/3.0), 
  //don't use d_filter_width given in input file now, 
  //keep the parameter here for future use
  
  Vector dx = patch->dCell();
  filter_width = pow((dx.x()*dx.y()*dx.z()), 1.0/3.0);
  double term = (d_model_constant * filter_width)
               *(d_model_constant * filter_width);
  
  SCIRun::StaticArray<CCVariable<double> > SIJ(6);
  for (int comp = 0; comp <= 5; comp++) {
    new_dw->allocateTemporary(SIJ[comp], patch,Ghost::AroundCells,2);
    SIJ[comp].initialize(0.0);
  }
   
  computeStrainRate(patch, uvel_FC, vvel_FC, wvel_FC, indx, d_sharedState, new_dw,
                    SIJ);

  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by one cell in ghostCells
  int NGC =1;  // number of ghostCells
  for(CellIterator iter = patch->getCellIterator(NGC); !iter.done(); iter++) {  
    IntVector c = *iter;    
       
    turb_viscosity[c]=rho_CC[c] * term * 
    sqrt(2.0 * (SIJ[0][c]*SIJ[0][c] + SIJ[1][c]*SIJ[1][c] + SIJ[2][c]*SIJ[2][c] + 
         2.0 * (SIJ[3][c]*SIJ[3][c] + SIJ[4][c]*SIJ[4][c] + SIJ[5][c]*SIJ[5][c])));

   }
}


/* ---------------------------------------------------------------------
  Function~  computeStrainRate
  Purpose~ Calculate the grid strain rate
  -----------------------------------------------------------------------  */  
void Smagorinsky_Model::computeStrainRate(const Patch* patch,
                                    const SFCXVariable<double>& uvel_FC,
                                    const SFCYVariable<double>& vvel_FC,
                                    const SFCZVariable<double>& wvel_FC,
                                    const int indx,
                                    SimulationStateP&  d_sharedState,
                                    DataWarehouse* new_dw,
                                    SCIRun::StaticArray<CCVariable<double> >& SIJ)
{
  Vector dx = patch->dCell();
  double delX = dx.x();
  double delY = dx.y();
  double delZ = dx.z();

  //__________________________________
  //  At patch boundaries you need to extend
  // the computational footprint by twe cells in ghostCells
  int NGC =2;  // number of ghostCells
  for(CellIterator iter = patch->getCellIterator(NGC); !iter.done(); iter++) {
    IntVector c = *iter;
    int i = c.x();
    int j = c.y();
    int k = c.z();

    IntVector right = IntVector(i+1,j,k);
    IntVector top   = IntVector(i,j+1,k);
    IntVector front = IntVector(i,j,k+1);             
    //__________________________________
    //calculate the grid strain rate tensor
    SIJ[0][c] = (uvel_FC[right] - uvel_FC[c])/delX;
    SIJ[1][c] = (vvel_FC[top]   - vvel_FC[c])/delY;
    SIJ[2][c] = (wvel_FC[front] - wvel_FC[c])/delZ;
    
    SIJ[3][c] = 0.5 * ((uvel_FC[right] - uvel_FC[c])/delY 
                       + (vvel_FC[top] - vvel_FC[c])/delX);
    SIJ[4][c] = 0.5 * ((uvel_FC[right] - uvel_FC[c])/delZ 
                     + (wvel_FC[front] - wvel_FC[c])/delX);
    SIJ[5][c] = 0.5 * ((vvel_FC[top] - vvel_FC[c])/delZ 
                   + (wvel_FC[front] - wvel_FC[c])/delY);
  }
  
  for (int comp = 0; comp < 6; comp ++ ) {
    setBC(SIJ[comp],"zeroNeumann",patch, d_sharedState, indx, new_dw);
  } 
 
}
  
//__________________________________
//
void Smagorinsky_Model::scheduleComputeVariance(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* /*matls*/)
{
  if(filterScalars.size() > 0){
    for(int i=0;i<static_cast<int>(filterScalars.size());i++){
      FilterScalar* s = filterScalars[i];
      Task* task = scinew Task("Smagorinsky_Model::computeVariance",this, 
                               &Smagorinsky_Model::computeVariance, s);
                               
      task->requires(Task::OldDW, s->scalar, Ghost::AroundCells, 1);
      task->computes(s->scalarVariance);
      sched->addTask(task, patches, s->matl_set);
    }
  }
}
//__________________________________
//
void Smagorinsky_Model::computeVariance(const ProcessorGroup*, 
                                        const PatchSubset* patches,
                                        const MaterialSubset* matls,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw,
                                        FilterScalar* s)
{
  cout_doing << "Doing computeVariance "<< "\t\t\t Smagorinsky_Model" << endl;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    for(int m=0;m<matls->size();m++){
      int matl = matls->get(m);
      constCCVariable<double> f;
      CCVariable<double> fvar;
      old_dw->get(f, s->scalar, matl, patch, Ghost::AroundCells, 1);
      new_dw->allocateAndPut(fvar, s->scalarVariance, matl, patch);
      
      Vector dx = patch->dCell();
      Vector inv_dx(1./dx.x(), 1./dx.y(), 1./dx.z());
      double mixing_length = cbrt(dx.x()*dx.y()*dx.z());
      double scale = mixing_length*mixing_length;

      for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
        const IntVector& c = *iter;
        
        // Compute the difference of the face centered overages,
        //   0.5*(f[c+IntVector(1,0,0)]+f[c]) 
        //  -0.5*(f[c]-f[c-IntVector(1,0,0)])
        // which is
        //   0.5*(f[c+IntVector(1,0,0)]-f[c-IntVector(1,0,0)])
        // do the same for x,y,z
        
        Vector df(0.5*(f[c+IntVector(1,0,0)]-f[c-IntVector(1,0,0)]),
                  0.5*(f[c+IntVector(0,1,0)]-f[c-IntVector(0,1,0)]),
                  0.5*(f[c+IntVector(0,0,1)]-f[c-IntVector(0,0,1)]));
        df *= inv_dx;
        fvar[c] = scale * df.length2();
      }
      setBC(fvar,s->scalarVariance->getName(),patch, d_sharedState, matl, new_dw);
    }
  }
}
