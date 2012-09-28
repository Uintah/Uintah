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


/********************************************************************************
    CrackFrontNode.h

    Created by Yajun Guo in Jan 2004.
********************************************************************************/

#ifndef UINTAH_HOMEBREW_CRACKFRONTNODE_H
#define UNITAH_HOMEBREW_CRACKFRONTNODE_H

#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>

namespace Uintah {

using SCIRun::Vector;
using SCIRun::Point;

class CrackFrontNode
{
  public:
     // Constructors
     inline CrackFrontNode();
     inline CrackFrontNode(const int& n);
     inline CrackFrontNode(const CrackFrontNode& copy);

     // Destructor
     inline ~CrackFrontNode();

     // Public parameters
     int node;    // node number 

     int preIdx; // node=node(preIdx) 
     int maxIdx; // max index of the sub-crack on which the node is
     int minIdx; // min index of the sub-crack on which the node is

     int patchID; // patch ID of the node
     short inMat; // flag if the node is inside of material

     Vector v1;   // outward normal of crack-front at the node
     Vector v2;   // out-plane normal of crack-front at the node
     Vector v3;   // tangential normal of crack-front at the node

     Vector  J;   // J-integral at the node
     Vector  K;   // stress-intensity at the node

     short  cp;   // flag if the node will propagate
     Vector da;   // crack incremental at the node
     Point app;   // position after propagation
     int   apn;   // node number after propagation
};

inline CrackFrontNode::CrackFrontNode()
{
  // Default Constructor
  node=-1;
  preIdx=-1;
  maxIdx=-1;
  minIdx=99999999;
  patchID=-1;
  inMat=0;
  v1=Vector(0.,0.,0.);
  v2=Vector(0.,0.,0.);
  v3=Vector(0.,0.,0.);
  J=Vector(-1e99,-1e99,-1e99);
  K=Vector(-1e99,-1e99,-1e99);
  cp=0;
  da=Vector(0.,0.,0.);
  app=Point(-1e99,-1e99,-1e99);
  apn=-1;
}

inline CrackFrontNode::CrackFrontNode(const int& n)
{
  // Constructor, initializing node
  node=n;
}

inline CrackFrontNode::CrackFrontNode(const CrackFrontNode& copy)
{
  node=copy.node;
  preIdx=copy.preIdx;
  maxIdx=copy.maxIdx;
  minIdx=copy.minIdx;
  patchID=copy.patchID;
  inMat=copy.inMat;
  v1=copy.v1;
  v2=copy.v2;
  v3=copy.v3;
  J=copy.J;
  K=copy.K;
  cp=copy.cp;
  da=copy.da;
  app=copy.app;
  apn=copy.apn;
}

inline CrackFrontNode::~CrackFrontNode()
{
  // Destructor
  // Do nothing
}

}//End namespace Uintah

#endif  /* __CRACKFRONTNODE_H__*/
