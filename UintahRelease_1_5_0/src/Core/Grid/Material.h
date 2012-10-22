/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Grid/SimulationStateP.h>

namespace Uintah {

/**************************************

CLASS
   Material

   Short description...

GENERAL INFORMATION

   Material.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


KEYWORDS
   Material

DESCRIPTION
   Long description...

WARNING

****************************************/

//using ::Grid::Patch;
//using ::Interface::DataWarehouseP;


   class Material {
   public:
     Material(ProblemSpecP& ps);
     Material();
      
     virtual ~Material();
      
     virtual ProblemSpecP outputProblemSpec(ProblemSpecP& ps);

      //////////
      // Return index associated with this material's
      // location in the data warehouse
      int getDWIndex() const;
      
      void setDWIndex(int);

     const MaterialSubset* thisMaterial() const {
       return thismatl;
     }
     
     virtual void registerParticleState(SimulationState* ss);

     bool hasName() const {
       return haveName;
     }
     std::string getName() const {
       return name;
     }
   protected:

      // Index associated with this material's spot in the DW
      int d_dwindex;
      MaterialSubset* thismatl;



   private:

     bool haveName;
     std::string name;
      
     Material(const Material &mat);
     Material& operator=(const Material &mat);
   };
} // End namespace Uintah

#endif // __MATERIAL_H__
