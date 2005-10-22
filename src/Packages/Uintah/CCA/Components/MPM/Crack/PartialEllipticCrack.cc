#include <Packages/Uintah/CCA/Components/MPM/Crack/PartialEllipticCrack.h>
#include <iostream>

using std::cout;
using std::endl;

using namespace Uintah;

PartialEllipticCrack::PartialEllipticCrack(ProblemSpecP& ps)
{
  readCrack(ps);
}

PartialEllipticCrack::~PartialEllipticCrack()
{
}



void PartialEllipticCrack::readCrack(ProblemSpecP& pellipse_ps)
{
  // Center,two points on major and minor axes
  Point p;
 
  pellipse_ps->require("center",p);       vertices.push_back(p);
  pellipse_ps->require("point_axis1",p);  vertices.push_back(p);
  pellipse_ps->require("point_axis2",p);  vertices.push_back(p);
  
  // Extent of the partial ellipse (quarter or half)
  pellipse_ps->require("extent",Extent);
    
  // Resolution on circumference
  NCells=1; 
  pellipse_ps->require("resolution_circumference",NCells);
  
  // Crack front segment ID, -1 by default which means all segments are crack front
  CrkFrtSegID=-1; 
  pellipse_ps->get("crack_front_segment_ID",CrkFrtSegID);

}


void PartialEllipticCrack::outputInitialCrackPlane(int i)
{
  
  cout << "  * Partial ellipse " << i+1 << " (" << Extent
       << "): meshed by " << NCells
       << " cells on the circumference." << endl;
  if(CrkFrtSegID==-1)
    cout << "    crack front: on the ellipse circumference" << endl;
  else
    cout << "    crack front segment ID: " << CrkFrtSegID
         << endl;
  cout << "    center: " << vertices[0] << endl;
  cout << "    end point on axis1: " << vertices[1] << endl;
  cout << "    end point on axis2: " << vertices[2] << endl;

}

void PartialEllipticCrack::discretize(int& nstart0,vector<Point>& cx, 
                           vector<IntVector>& ce,vector<int>& SegNodes)
{
}

