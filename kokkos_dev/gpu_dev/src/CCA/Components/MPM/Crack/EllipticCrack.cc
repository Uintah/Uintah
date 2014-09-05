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


#include <CCA/Components/MPM/Crack/EllipticCrack.h>
#include <iostream>

using std::cout;
using std::endl;
using std::vector;

using namespace Uintah;


EllipticCrack::EllipticCrack(ProblemSpecP& ps)
{
  readCrack(ps);
}


EllipticCrack::~EllipticCrack()
{
  // Destructor
  // Do nothing
}


void EllipticCrack::readCrack(ProblemSpecP& ellipse_ps)
{
  
  // Three points on the arc
  Point p; 
  ellipse_ps->require("point1_axis1",p);   vertices.push_back(p);
  ellipse_ps->require("point_axis2", p);   vertices.push_back(p);
  ellipse_ps->require("point2_axis1",p);   vertices.push_back(p);
  
  // Resolution on circumference
  NCells=1; 
  ellipse_ps->require("resolution_circumference",NCells);
  
  // Crack front segment ID, -1 by default which means all segments are crack front
  CrkFrtSegID=-1; 
  ellipse_ps->get("crack_front_segment_ID",CrkFrtSegID);
}

void EllipticCrack::outputInitialCrackPlane(int i)
{

  cout << "  * Ellipse " << i+1 << ": meshed by " << NCells
       << " cells on the circumference." << endl;
  if(CrkFrtSegID==-1)
    cout << "    crack front: on the ellipse circumference" << endl;
  else    
    cout << "    crack front segment ID: " << CrkFrtSegID
         << endl;
  cout << "    end point on axis1: " << vertices[0] << endl;
  cout << "    end point on axis2: " << vertices[1] << endl;
  cout << "    another end point on axis1: " << vertices[2]
       << endl;


}

void EllipticCrack::discretize(int& nstart0,vector<Point>& cx, 
                           vector<IntVector>& ce,vector<int>& SegNodes)
{
}

