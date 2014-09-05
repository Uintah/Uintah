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


#ifndef UINTAH_HOMEBREW_PATCHFIXER_H
#define UINTAH_HOMEBREW_PATCHFIXER_H

#include <CCA/Components/Regridder/BNRTask.h>
#include <Core/Parallel/ProcessorGroup.h>

#include <vector>


namespace Uintah {

/**************************************

CLASS
   PatchFixer 
   
   This class takes a patchset and fixes it to meet Uintah's constraints
	 neighbor constraints.

GENERAL INFORMATION

   PatchFixer.h

	 Justin Luitjens
   Bryan Worthen
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   PatchFixer

DESCRIPTION
   This class takes a patchset and fixes it to meet Uintah's neighbor 
	 constraints by splitting patches where coarse/fine interfaces exist 
	 along faces of patches.  
  
WARNING
  
****************************************/
  class PatchFixer {
  public:
    PatchFixer(const ProcessorGroup* pg) : d_myworld(pg) 
    {
    };
    ~PatchFixer() {};
    void FixUp(std::vector<Region> &patches);	
  private:
    const ProcessorGroup *d_myworld;
    std::vector<int> lattice_;                   //lattice of ints
    std::vector<int> cellstolattice_[3];         //map from cells to lattice
    std::vector<int> latticetocells_[3];         //map from lattice to cells
    Region bounds_;                    //bounds of the patches
    IntVector csize_,lsize_;		            // size of the cells and lattice
		
    void FixFace(std::vector<Region> &patches,Region patch, int dim, int side);
    void SplitPatch(int index, std::vector<Region> &patches, const Split &split);
    void BuildLattice(const std::vector<Region> &patches);
    void Fill(const Region patch,const int id);
    inline int CellToLattice(int c, int d)
    {
      return cellstolattice_[d][c-bounds_.getLow()[d]];
    };
    inline void CellToLattice( int* c)
    {
      for(int d=0;d<3;d++)
        {
          c[d]=CellToLattice(c[d],d);
        }
    };
    inline int LatticeToCell(int l, int d)
    {
      return latticetocells_[d][l];
    };
    inline void LatticeToCell( int* l)
    {
      for(int d=0;d<3;d++)
        {
          l[d]=LatticeToCell(l[d],d);
        }
    };
    
  };

} // End namespace Uintah

#endif
