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
