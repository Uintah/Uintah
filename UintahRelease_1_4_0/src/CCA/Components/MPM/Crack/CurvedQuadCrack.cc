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


#include <CCA/Components/MPM/Crack/CurvedQuadCrack.h>
#include <iostream>

using std::cout;
using std::endl;
using std::vector;
using std::string;

using namespace Uintah;

CurvedQuadCrack::CurvedQuadCrack(ProblemSpecP& ps)
{
  readCrack(ps);
}


CurvedQuadCrack::~CurvedQuadCrack()
{
  // Destructor
  // Do nothing
}


void CurvedQuadCrack::readCrack(ProblemSpecP& cquad_ps)
{
  // Four vertices of the curved quad
  Point p;
  cquad_ps->require("p1",p);  vertices.push_back(p);
  cquad_ps->require("p2",p);  vertices.push_back(p);
  cquad_ps->require("p3",p);  vertices.push_back(p);
  cquad_ps->require("p4",p);  vertices.push_back(p);

  
  // Mesh resolution on two opposite straight sides
  NStraightSides=1;
  cquad_ps->get("resolution_straight_sides",NStraightSides);
  
  // Characteristic points on two opposite cuvered sides
  ProblemSpecP side2_ps=cquad_ps->findBlock("points_curved_side2"); 
  for(ProblemSpecP pt_ps=side2_ps->findBlock("point"); pt_ps!=0; 
      pt_ps=pt_ps->findNextBlock("point")) {  
    pt_ps->get("val",p); 
    PtsSide2.push_back(p);
  }
  
  ProblemSpecP side4_ps=cquad_ps->findBlock("points_curved_side4");
  for(ProblemSpecP pt_ps=side4_ps->findBlock("point"); pt_ps!=0;
      pt_ps=pt_ps->findNextBlock("point")) {
    pt_ps->get("val",p); 
    PtsSide4.push_back(p);
  }
  
  if(PtsSide4.size()!=PtsSide2.size()) {
    cout << "Error: The points on curved side 2 and side 4 "
         << "should appear in pairs." << endl;  
  }
  
  // Crack front
  string cfsides("");
  cquad_ps->get("crack_front_sides",cfsides);
  if(cfsides.length()==4) {
    for(string::const_iterator iter=cfsides.begin();
        iter!=cfsides.end(); iter++) {
      if(*iter=='Y' || *iter=='y') 
        AtFront.push_back(true);
      else
        AtFront.push_back(false);
    }
  }
  else if(cfsides.length()==0) {
    for(int i=0; i<4; i++)
      AtFront.push_back(false);
  }

  // Repetition information
  Repetition = 1;
  cquad_ps->get("repetition",Repetition);

  
  Vector offset=Vector(0.,0.,0.);
  if(Repetition>1) 
    cquad_ps->require("offset",offset);
  Offset = offset;
}


void CurvedQuadCrack::outputInitialCrackPlane(int i)
{
  
  cout << "  * Curved quad " << i+1 << ":" << endl;
  cout << "    Four vertices:" << endl; 
  // four vertices
  for(int j=0;j<4;j++) 
    cout << "      p" << j+1 << ": " << vertices[j] << endl;
  // resolution on straight sides 1 & 3
  cout << "    Resolution on straight sides (sides p1-p2 and p3-p4):"
       << NStraightSides << endl; 
  // points on curved egde 2
  cout << "    Points on curved side 2 (p2-p3): " << endl;
  for(int j=0; j< 3; j++)
    cout << "      p" << j+1 << ": " << PtsSide2[j] << endl;
  // points on curved side 3
  cout << "    Points on curved side 4 (p1-p4): " << endl;
  for(int j=0; j< 3; j++)
    cout << "      p" << j+1 << ": " << PtsSide4[j] << endl; 
  // crack-front sides
  for(int j=0;j<4;j++) {
    if(AtFront[j]) {
      int j2=(j+2<5 ? j+2 : 1);
      cout << "    Side " << j+1 << " (p" << j+1 << "-" << "p" << j2
           << ") is a crack front." << endl;
    }
  }

  // repetition information
  if(Repetition>1) {
    cout << "    The quad is repeated by " << Repetition
         << " times with the offset " << Offset << "." << endl;
  }   
        
}

void CurvedQuadCrack::discretize(int& nstart0,vector<Point>& cx,
                                 vector<IntVector>& ce,vector<int>& SegNodes)
{

  int i,j,ni,nj,n1,n2,n3;
  int nstart1,nstart2,nstart3;
  Point p1,p2,p3,p4,pt;

  // Four vertices of the curved quad
  p1=vertices[0];
  p2=vertices[1];
  p3=vertices[2];
  p4=vertices[3];
  
  // Mesh resolutions on curved sides (ni) & straight sides (nj)
  ni=NStraightSides;
  nj=PtsSide2.size()+1;
  
  // Nodes on curved sides 2 (p2-p3) & 4 (p1-p4) - "j" direction
  Point* p_s2=new Point[2*nj+1];
  Point* p_s4=new Point[2*nj+1];
  p_s2[0]=p2;   p_s2[2*nj]=p3;
  p_s4[0]=p1;   p_s4[2*nj]=p4;
  for(int l=2; l<2*nj; l+=2) {
    p_s2[l]=PtsSide2[l/2-1];
    p_s4[l]=PtsSide4[l/2-1];
  }     
  for(int l=1; l<2*nj; l+=2) {
    p_s2[l]=p_s2[l-1]+(p_s2[l+1]-p_s2[l-1])/2.;
    p_s4[l]=p_s4[l-1]+(p_s4[l+1]-p_s4[l-1])/2.; 
  }     
  
  // Generate crack nodes
  for(j=0; j<=nj; j++) {
    for(i=0; i<=ni; i++) { 
      pt=p_s4[2*j]+(p_s2[2*j]-p_s4[2*j])*(float)i/ni;
      cx.push_back(pt);
    }     
    if(j!=nj) {
      for(i=0; i<ni; i++) { 
        int jj=2*j+1;
        pt=p_s4[jj]+(p_s2[jj]-p_s4[jj])*(float)(2*i+1)/(2*ni);
        cx.push_back(pt);
      }
    }  
  } 
  delete [] p_s2;
  delete [] p_s4;
  
  // Generate crack elements
  for(j=0; j<nj; j++) {
    nstart1=nstart0+(2*ni+1)*j;
    nstart2=nstart1+(ni+1);
    nstart3=nstart2+ni;
    for(i=0; i<ni; i++) {
      // the 1st element
      n1=nstart2+i;  n2=nstart1+i;  n3=nstart1+(i+1);
      ce.push_back(IntVector(n1,n2,n3));
      // the 2nd element
      n1=nstart2+i;  n2=nstart3+i;  n3=nstart1+i;
      ce.push_back(IntVector(n1,n2,n3));
      // the 3rd element
      n1=nstart2+i;  n2=nstart1+(i+1);  n3=nstart3+(i+1);
      ce.push_back(IntVector(n1,n2,n3));
      // the 4th element
      n1=nstart2+i;  n2=nstart3+(i+1);  n3=nstart3+i;
      ce.push_back(IntVector(n1,n2,n3));
    }  // End of loop over j
  }  // End of loop over i
  nstart0+=((2*ni+1)*nj+ni+1);
  
  // Collect crack-front nodes
  int seg0=0; 
  for(j=0; j<4; j++) {
    if(!AtFront[j]) { seg0=j+1;  break; }
  }
  
  for(int l=0; l<4; l++) { // Loop over sides of the quad
    // Find the side index of the crack front "j"           
    j=seg0+l;   
    if(j>3) j-=4;
    if(AtFront[j]) { 
      // pt1-pt2 is crack-front      
      int j1 = (j!=3 ? j+1 : 0);
      Point pt1=vertices[j];
      Point pt2=vertices[j1];
      for(i=0; i<(int)ce.size(); i++) {
        int ii=i;
        if(j>1) ii= (int) ce.size()-(i+1);
        n1=ce[ii].x();
        n2=ce[ii].y();
        n3=ce[ii].z();
        for(int s=0; s<3; s++) { // Loop over sides of the element
          int sn=n1,en=n2; 
          if(s==1) {sn=n2; en=n3;}
          if(s==2) {sn=n3; en=n1;}
          if(twoLinesCoincide(pt1,pt2,cx[sn],cx[en])) {
            SegNodes.push_back(sn);
            SegNodes.push_back(en);
          }
        }
      } // End of loop over i
    }
  } // End of loop over l                 
  
  
}

