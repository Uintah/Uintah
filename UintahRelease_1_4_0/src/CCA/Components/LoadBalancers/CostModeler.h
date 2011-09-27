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


#ifndef UINTAH_HOMEBREW_CostModeler_H
#define UINTAH_HOMEBREW_CostModeler_H

#include <vector>

#include <Core/Grid/Grid.h>
#include <Core/Grid/Region.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Components/LoadBalancers/CostForecasterBase.h>
#include <CCA/Components/LoadBalancers/ProfileDriver.h>
namespace Uintah {
   /**************************************
     
     CLASS
       CostModeler 
      
       Computes weights of execution using a simple model based on the number
       of cells and particles.
      
     GENERAL INFORMATION
      
       CostModeler.h
      
       Justin Luitjens
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       CostModeler
       DynamicLoadBalancer
      
     DESCRIPTION
       Computes the weights of execution for each patch in the grid based
       on the number of cells and particles it has.  

       The DLB uses the getWeights function to assign weights to regions which 
       can then be used to load balance the calculation.
      
     WARNING
      
     ****************************************/

  class CostModeler : public CostForecasterBase {
    public:
      void getWeights(const Grid* grid, std::vector<std::vector<int> > num_particles, std::vector<std::vector<double> >&costs)
      {
        costs.resize(grid->numLevels());
        //for each level
        for (int l=0; l<grid->numLevels();l++)
        {
          LevelP level=grid->getLevel(l);
          costs[l].resize(level->numPatches());
          for(int p=0; p<level->numPatches();p++)
          {
            const Patch *patch=level->getPatch(p);
            costs[l][p]=d_patchCost+patch->getNumCells()*d_cellCost+(patch->getNumExtraCells()-patch->getNumCells())*d_extraCellCost+num_particles[l][p]*d_particleCost;
          }
        }
      }
      void setCosts(double patchCost, double cellCost, double extraCellCost, double particleCost)
      {
        d_patchCost=patchCost;d_cellCost=cellCost;d_extraCellCost=extraCellCost;d_particleCost=particleCost;
      }

      CostModeler(double patchCost, double cellCost, double extraCellCost, double particleCost) { setCosts(patchCost,cellCost,extraCellCost,particleCost); };
    protected:
      double d_patchCost, d_cellCost, d_extraCellCost, d_particleCost;

  };
} // End namespace Uintah


#endif

