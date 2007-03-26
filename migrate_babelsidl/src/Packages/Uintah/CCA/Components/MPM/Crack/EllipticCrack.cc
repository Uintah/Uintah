#include <Packages/Uintah/CCA/Components/MPM/Crack/EllipticCrack.h>
#include <iostream>

using std::cout;
using std::endl;
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

