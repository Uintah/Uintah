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


#ifndef UINTAH_GRID_CrackBC_H
#define UINTAH_GRID_CrackBC_H

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

#include <CCA/Components/MPM/PhysicalBC/MPMPhysicalBC.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

using namespace SCIRun;
   
/**************************************

CLASS
   CrackBC
   
  
GENERAL INFORMATION

   CrackBC.h

   Honglai Tan
   Department of Materials Science and Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   CrackBC

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class CrackBC : public MPMPhysicalBC  {
   public:
      CrackBC(ProblemSpecP& ps);
      virtual std::string getType() const;
      virtual void outputProblemSpec(ProblemSpecP& ps);
      
      double   x1() const;
      double   y1() const;
      double   x2() const;
      double   y2() const;
      double   x3() const;
      double   y3() const;
      double   x4() const;
      double   y4() const;

      const Vector&  e1() const;
      const Vector&  e2() const;
      const Vector&  e3() const;
      const Point&   origin() const;
         
   private:
      CrackBC(const CrackBC&);
      CrackBC& operator=(const CrackBC&);
      
      Point  d_origin;
      Vector d_e1,d_e2,d_e3;
      double d_x1,d_y1,d_x2,d_y2,d_x3,d_y3,d_x4,d_y4;
   };
   
} // end namespace Uintah

#endif
