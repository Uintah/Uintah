/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_Ghost_H
#define UINTAH_HOMEBREW_Ghost_H

#include <Core/Util/Assert.h>
#include <Core/Geometry/IntVector.h>
#undef None

#define ALL_CELLS 268435455 /*= MAX_INT/8*/

namespace Uintah {
   /**************************************
     
     CLASS
       Ghost
      
       Short Description...
      
     GENERAL INFORMATION
      
       Ghost.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
      
     KEYWORDS
       Ghost
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/

   class Ghost {
   public:
      enum GhostType {
         None,
         AroundNodes,
         AroundCells,
         AroundFacesX,
         AroundFacesY,
         AroundFacesZ,
         AroundFaces,
         numGhostTypes // 7
      };

     static SCIRun::IntVector getGhostTypeDir(Ghost::GhostType gtype)
     {
       int index = (int)gtype;
       CHECKARRAYBOUNDS(index, 0, numGhostTypes);
       return directions[index];
     }

     static std::string getGhostTypeName(Ghost::GhostType gtype)
     {
       int index = (int)gtype;
       CHECKARRAYBOUNDS(index, 0, numGhostTypes);
       return names[index];
     }
   private:
      Ghost();
      Ghost(const Ghost&);
      Ghost& operator=(const Ghost&);

     static SCIRun::IntVector directions[numGhostTypes];
     static std::string names[numGhostTypes];
#ifdef __GNUG__
   public:
      // Keep gcc quiet, to avoid:
      // warning: all member functions in class `Packages/Uintah::Ghost' are private
      Ghost(Ghost*);
#endif
   };
} // End namespace Uintah

#endif
