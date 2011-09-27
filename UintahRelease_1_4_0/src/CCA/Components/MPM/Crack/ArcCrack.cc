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


#include <CCA/Components/MPM/Crack/ArcCrack.h>
#include <Core/Geometry/Plane.h>
#include <Core/Math/Matrix3.h>
#include <iostream>

using std::cout;
using std::endl;
using std::vector;

using namespace Uintah;
using namespace SCIRun;


ArcCrack::ArcCrack(ProblemSpecP& ps)
{
  readCrack(ps);
}

ArcCrack::~ArcCrack()
{
  // Destructor
  // Do nothing
}

void ArcCrack::readCrack(ProblemSpecP& arc_ps)
{

  // Three points on the arc
  Point p;    
  arc_ps->require("start_point",p);   vertices.push_back(p);
  arc_ps->require("middle_point",p);  vertices.push_back(p);
  arc_ps->require("end_point",p);     vertices.push_back(p);
  
  // Resolution on circumference
  NCells=1;
  arc_ps->require("resolution_circumference",NCells);
  
  // Crack front segment ID, -1 by default which means all segments are crack front
  CrkFrtSegID=-1; 
  arc_ps->get("crack_front_segment_ID",CrkFrtSegID);

}


void ArcCrack::outputInitialCrackPlane(int i)
{

  cout << "  * Arc " << i+1 << ": meshed by " << NCells
       << " cells on the circumference." << endl;
  if(CrkFrtSegID==-1)
    cout << "   crack front: on the arc" << endl;
  else
    cout << "   crack front segment ID: " << CrkFrtSegID << endl;
  cout << "\n    start, middle and end points of the arc:"  << endl;
  for(int j=0;j<3;j++)
    cout << "    p" << j+1 << ": " << vertices[j] << endl;

}

void ArcCrack::discretize(int& nstart0,vector<Point>& cx, 
                           vector<IntVector>& ce,vector<int>& SegNodes)
{

  // Three points of the arc
  Point p1=vertices[0];
  Point p2=vertices[1];
  Point p3=vertices[2];
  double x1,y1,z1,x2,y2,z2,x3,y3,z3;
  x1=p1.x(); y1=p1.y(); z1=p1.z();
  x2=p2.x(); y2=p2.y(); z2=p2.z();
  x3=p3.x(); y3=p3.y(); z3=p3.z();
  
  // Find center of the arc
  double a1,b1,c1,d1,a2,b2,c2,d2,a3,b3,c3,d3;
  a1=2*(x2-x1); b1=2*(y2-y1); c1=2*(z2-z1);
  d1=x1*x1-x2*x2+y1*y1-y2*y2+z1*z1-z2*z2;
  a2=2*(x3-x1); b2=2*(y3-y1); c2=2*(z3-z1);
  d2=x1*x1-x3*x3+y1*y1-y3*y3+z1*z1-z3*z3;
  Plane arc_plane(p1,p2,p3);

  a3=arc_plane.normal().x();
  b3=arc_plane.normal().y();
  c3=arc_plane.normal().z();
  d3=arc_plane.eval_point(p1);
  
  double delt,deltx,delty,deltz;
  delt  = Matrix3(a1,b1,c1,a2,b2,c2,a3,b3,c3).Determinant();
  deltx = Matrix3(-d1,b1,c1,-d2,b2,c2,-d3,b3,c3).Determinant();
  delty = Matrix3(a1,-d1,c1,a2,-d2,c2,a3,-d3,c3).Determinant();
  deltz = Matrix3(a1,b1,-d1,a2,b2,-d2,a3,b3,-d3).Determinant();
  double x0,y0,z0;
  x0=deltx/delt;  y0=delty/delt;  z0=deltz/delt;
  Point origin=Point(x0,y0,z0);
  double radius=sqrt((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)+(z1-z0)*(z1-z0));
  
  // Define local coordinates
  Vector v2,v3;
  double temp=sqrt(a3*a3+b3*b3+c3*c3);
  v3=Vector(a3/temp,b3/temp,c3/temp);
  
  // TwoPtsDirCos(origin,p1);
  Vector v1=(origin.asVector() - p1.asVector()); 
  v1.normalize();

  Vector v31=Cross(v3,v1);
  v2=v31/v31.length();
  double lx,mx,nx,ly,my,ny;
  lx=v1.x();  mx=v1.y();  nx=v1.z();
  ly=v2.x();  my=v2.y();  ny=v2.z();
  
  // Angle of the arc
  double angleOfArc;
  double PI=3.141592654;
  double x3prime,y3prime;
  x3prime=lx*(x3-x0)+mx*(y3-y0)+nx*(z3-z0);
  y3prime=ly*(x3-x0)+my*(y3-y0)+ny*(z3-z0);
  double cosTheta=x3prime/radius;
  double sinTheta=y3prime/radius;
  double thetaQ=fabs(asin(y3prime/radius));
  if(sinTheta>=0.) {
    if(cosTheta>=0) angleOfArc=thetaQ;
    else angleOfArc=PI-thetaQ;
  }
  else {
    if(cosTheta<=0.) angleOfArc=PI+thetaQ;
    else angleOfArc=2*PI-thetaQ;
  }
  
  // Generate crack nodes
  cx.push_back(origin);
  for(int j=0;j<=NCells;j++) {
    double thetai=angleOfArc*j/NCells;
    double xiprime=radius*cos(thetai);
    double yiprime=radius*sin(thetai);
    double xi=lx*xiprime+ly*yiprime+x0;
    double yi=mx*xiprime+my*yiprime+y0;
    double zi=nx*xiprime+ny*yiprime+z0;
    cx.push_back(Point(xi,yi,zi));
  } 
  
  // Generate crack elements
  for(int j=1;j<=NCells;j++) {
    int n1=nstart0;
    int n2=nstart0+j;
    int n3=nstart0+(j+1);
    ce.push_back(IntVector(n1,n2,n3));
    // Crack front nodes
    if(CrkFrtSegID==-1 || CrkFrtSegID==j) {
      SegNodes.push_back(n2);
      SegNodes.push_back(n3);
    }
  }
  nstart0+=NCells+2;
}

