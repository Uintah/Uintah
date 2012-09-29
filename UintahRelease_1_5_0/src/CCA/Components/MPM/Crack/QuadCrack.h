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

#ifndef UINTAH_HOMEBREW_QUAD_CRACK_H
#define UINTAH_HOMEBREW_QUAD_CRACK_H

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Components/MPM/Crack/CrackGeometry.h>

namespace Uintah {

class QuadCrack : public CrackGeometry
{
  public:
     // Constructors
     QuadCrack(ProblemSpecP& ps);
     QuadCrack(const QuadCrack& copy);

     // Destructor
     virtual ~QuadCrack();
     virtual void readCrack(ProblemSpecP&);
     virtual void outputInitialCrackPlane(int i);
     virtual void discretize(int& nstart0,std::vector<Point>& cx,
                             std::vector<IntVector>& ce,std::vector<int>& SegNodes);

 protected:
     void GetGlobalCoordinates(const int &, const double &, 
                               const double&, Point&);
 private:
     int N12, N23;
     std::vector<bool> AtFront;
     int Repetition;
     Vector Offset;
};


}//End namespace Uintah

#endif  /* __QUAD_CRACK_H__*/
