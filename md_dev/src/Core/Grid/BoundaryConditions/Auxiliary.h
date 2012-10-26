/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef UINTAH_GRID_AuxiliaryBoundCond_H
#define UINTAH_GRID_AuxiliaryBoundCond_H

#include <Core/Grid/BoundaryConditions/BoundCondBase.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Malloc/Allocator.h>


namespace Uintah {
   
/**************************************

CLASS
   AuxiliaryBoundCond
   
   
GENERAL INFORMATION

   AuxiliaryBoundCond.h

   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   AuxiliaryBoundCond

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  class AuxiliaryBoundCond : public BoundCondBase  {
  public:
    AuxiliaryBoundCond():BoundCondBase("Auxiliary") {};
    AuxiliaryBoundCond(ProblemSpecP&) {d_type = "Auxiliary";};
    virtual ~AuxiliaryBoundCond() {};
    virtual AuxiliaryBoundCond* clone() {
      return scinew AuxiliaryBoundCond(*this);
    };
    virtual string getKind() const {return "auxiliary";};
  private:
#if 0
    AuxiliaryBoundCond(const AuxiliaryBoundCond&);
    AuxiliaryBoundCond& operator=(const AuxiliaryBoundCond&);
#endif
     
   };
} // End namespace Uintah

#endif
