/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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


#ifndef UINTAH_HOMEBREW_CURVED_QUAD_CRACK_H
#define UINTAH_HOMEBREW_CURVED_QUAD_CRACK_H

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Components/MPM/Crack/CrackGeometry.h>


namespace Uintah {

class CurvedQuadCrack : public CrackGeometry
{
  public:
     // Constructors
     CurvedQuadCrack(ProblemSpecP& ps);
     CurvedQuadCrack(const CurvedQuadCrack& copy);

     // Destructor
     virtual ~CurvedQuadCrack();
     virtual void readCrack(ProblemSpecP& ps);
     virtual void outputInitialCrackPlane(int i);
     virtual void discretize(int& nstart0,vector<Point>& cx,
                             vector<IntVector>& ce,vector<int>& SegNodes);

 private:

     int NStraightSides;
     vector<Point> PtsSide2,PtsSide4;
     vector<bool> AtFront;
     int Repetition;
     Vector Offset;
};


}//End namespace Uintah

#endif  /* __CURVED_QUAD_CRACK_H__*/
