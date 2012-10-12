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

#include <CCA/Components/MPM/Crack/QuadCrack.h>
#include <iostream>

using std::cout;
using std::endl;
using std::vector;
using std::string;

using namespace Uintah;

QuadCrack::QuadCrack(ProblemSpecP& ps)
{
  readCrack(ps);
}


QuadCrack::~QuadCrack()
{
  // Destructor
  // Do nothing
}


void QuadCrack::readCrack(ProblemSpecP& quad_ps)
{
  Point p;
  quad_ps->require("p1",p);   vertices.push_back(p);
  quad_ps->require("p2",p);   vertices.push_back(p);
  quad_ps->require("p3",p);   vertices.push_back(p);
  quad_ps->require("p4",p);   vertices.push_back(p);

  if (!quad_ps->get("p5",p)) {
    p = vertices[0]+.5*(vertices[1]-vertices[0]);
    vertices.push_back(p);
  }
  if (!quad_ps->get("p6",p)) {
    p = vertices[1]+.5*(vertices[2]-vertices[1]);
    vertices.push_back(p);  
  }
  if (!quad_ps->get("p7",p)) {
    p = vertices[2]+.5*(vertices[3]-vertices[2]);
    vertices.push_back(p);
  }
  if (!quad_ps->get("p6",p)) {
    p = vertices[0]+.5*(vertices[3]-vertices[0]);
    vertices.push_back(p);
  }
  
  // Mesh resolutions  
  N12=1,N23=1;
  quad_ps->get("resolution_p1_p2",N12);
  quad_ps->get("resolution_p2_p3",N23);

  
  // Crack front
  string cfsides("");
  quad_ps->get("crack_front_sides",cfsides);
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
  int n=1; 
  quad_ps->get("repetition",n);
  Repetition = n;
  
  Vector offset=Vector(0.,0.,0.);
  if(n>1) 
    quad_ps->require("offset",offset);
  Offset = offset;    
}


void QuadCrack::outputInitialCrackPlane(int i)
{

  cout << "  * Quad " << i+1 << ": meshed by [" << N12
       << ", " << N23 << ", " << N12
       << ", " << N23 << "]" << endl;
  for(int j=0;j< (int)vertices.size();j++)
    cout << "    p" << j+1 << ": " << vertices[j] << endl;
  for(int j=0;j<4;j++) {
    if(AtFront[j]) {
      int j2=(j+2<5 ? j+2 : 1);
      cout << "    Side " << j+1 << " (p" << j+1 << "-" << "p" << j2
           << ") is a crack front." << endl;
    }
  }
  
  if (Repetition > 1)
    cout << "    The quad is repeated by " << Repetition
         << " times with the offset " << Offset << "." << endl;
  
}

void QuadCrack::discretize(int& nstart0,vector<Point>& cx, 
                           vector<IntVector>& ce,vector<int>& SegNodes)
{

  int i,j,ni,nj,n1,n2,n3,num;
  int nstart1,nstart2,nstart3;
  double ksi,eta;
  Point p1,p2,p3,p4,pt;

  for (int l = 0; l < Repetition; l++) {
    // Mesh resolutions of the quad
    ni=N12;
    nj=N23;
    
    // total number of nodes of the quad
    num=(ni+1)*(nj+1)+ni*nj;
    
    
    // Flag if node i is on edge j, initialized by false 
    vector<vector<bool> > nodeOnEdge;
    for(i=0; i<num; i++) {
      for(j=0; j<4; j++) nodeOnEdge[i].push_back(false);
    }       
    
    // Generate crack nodes
    int count = -1;
    for(j=0; j<=nj; j++) {
      for(i=0; i<=ni; i++) {
        // Detect edge nodes
        count++;
        if (j==0) nodeOnEdge[count][0]=true;
        if (i==ni) nodeOnEdge[count][1]=true;
        if (j==nj) nodeOnEdge[count][2]=true;
        if (i==0) nodeOnEdge[count][3]=true;
        // Intrinsic coordinates
        ksi=-1.0+(float)(2*i)/ni;
        eta=-1.0+(float)(2*j)/nj;       
        GetGlobalCoordinates(l,ksi,eta,pt);
        cx.push_back(pt);
      }
      if(j!=nj) {
        for(i=0; i<ni; i++) {
          count++;
          // intrinsic coordinates
          ksi=-1.0+(float)(2*i+1)/ni;
          eta=-1.0+(float)(2*j+1)/nj;     
          // Global coordinates          
          GetGlobalCoordinates(l,ksi,eta,pt);
          cx.push_back(pt);
        }
      } 
    } 
    
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
      } 
    }  
    
    // Collect crack-front nodes
    for(j=0; j<4; j++) {
      if(AtFront[j]) {
        for (i = 0; i < (int)ce.size(); i++) {
          n1 = ce[i].x();
          n2 = ce[i].y();
          n3 = ce[i].z();
          if(n1<nstart0 || n2<nstart0 || n3<nstart0) continue;
          for (int s = 0; s < 3; s++) {
            int sn = n1, en=n2;
            if (s==1) {sn = n2; en=n3;}
            if (s==2) { sn= n3; en=n1;}
            if (nodeOnEdge[sn-nstart0][j] && nodeOnEdge[en-nstart0][j]) {
              SegNodes.push_back(sn);
              SegNodes.push_back(en);
            }
          }
        }
      }  
    }
    nstart0+=num;
  }
}

void QuadCrack::GetGlobalCoordinates(const int& l, const double& x,
                                     const double &y, Point& pt)
{
  // (x,y): intrinsic coordinates of point "pt".
  
  // Shape functions of the serendipity eight-noded quadrilateral element
  double sf[8];          
  sf[0]=(1.-x)*(1.-y)*(-1.-x-y)/4.;
  sf[1]=(1.+x)*(1.-y)*(-1.+x-y)/4.;
  sf[2]=(1.+x)*(1.+y)*(-1.+x+y)/4.;
  sf[3]=(1.-x)*(1.+y)*(-1.-x+y)/4.;
  sf[4]=(1.-x*x)*(1.-y)/2.;
  sf[5]=(1.+x)*(1.-y*y)/2.;
  sf[6]=(1.-x*x)*(1.+y)/2.;
  sf[7]=(1.-x)*(1.-y*y)/2.;
  
  // Global coordinates of (x,y)
  double px=0., py=0., pz=0.; 
  for(int j=0; j<8; j++) {
    px+=sf[j]*(vertices[j].x()+l*Offset.x());
    py+=sf[j]*(vertices[j].y()+l*Offset.y());
    pz+=sf[j]*(vertices[j].z()+l*Offset.z());
  }  
  pt=Point(px,py,pz);
  
}

