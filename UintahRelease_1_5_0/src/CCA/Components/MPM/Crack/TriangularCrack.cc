/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/MPM/Crack/TriangularCrack.h>
#include <string>
#include <iostream>

using std::cout;
using std::endl;
using std::vector;
using std::string;

using namespace Uintah;


TriangularCrack::TriangularCrack(ProblemSpecP& ps)
{
  readCrack(ps);
}


TriangularCrack::~TriangularCrack()
{
  // Destructor
  // Do nothing
}

void TriangularCrack::readCrack(ProblemSpecP& tri_ps)
{

  // Three vertices of the triangle
  Point p1,p2,p3,p4,p5,p6;
  tri_ps->require("p1",p1);    vertices.push_back(p1);
  tri_ps->require("p2",p2);    vertices.push_back(p2);
  tri_ps->require("p3",p3);    vertices.push_back(p3); 

  if(!tri_ps->get("p4",p4)) p4=p1+0.5*(p2-p1);  vertices.push_back(p4);
  if(!tri_ps->get("p5",p5)) p5=p2+0.5*(p3-p2);  vertices.push_back(p5);
  if(!tri_ps->get("p6",p6)) p6=p3+0.5*(p1-p3);  vertices.push_back(p6);
  
  // Mesh resolution 
  NCells=1; 
  tri_ps->get("resolution",NCells);
  
  // Crack front
  string cfsides("");
  tri_ps->get("crack_front_sides",cfsides);
  if(cfsides.length()==3) {
    for(string::const_iterator iter=cfsides.begin();
        iter!=cfsides.end(); iter++) {
      if( *iter=='Y' || *iter=='n') 
        AtFront.push_back(true);
      else
        AtFront.push_back(false);
    }
  }
  else if(cfsides.length()==0) {
    for(int i=0; i<3; i++) 
      AtFront.push_back(false);
  }
  
  tri_ps->get("repetition",Repetition);
  
  if (Repetition > 1) 
    tri_ps->require("offset",Offset);
}


void TriangularCrack::outputInitialCrackPlane(int i)
{
  
  cout << "  * Triangle " << i+1 << ": meshed by [" << NCells
       << ", " << NCells << ", " << NCells   << "]" << endl;
  for(int j=0;j<3;j++)
    cout << "    p" << j+1 << ": " << vertices[j] << endl;
  for(int j=0;j<3;j++) {
    if(AtFront[j]) {
      int j2=(j+2<4 ? j+2 : 1);
      cout << "    side " << j+1 << " (p" << j+1 << "-" << "p" << j2
           << ") is a crack front." << endl;
    }
  }
  
}

void TriangularCrack::discretize(int& nstart0,vector<Point>& cx, 
                                 vector<IntVector>& ce,vector<int>& SegNodes)
{
  int i,j;
  int nstart1,nstart2,n1,n2,n3;
  Point p1,p2,p3,pt;

  // Three vertices of the triangle
  p1=vertices[0];
  p2=vertices[1];
  p3=vertices[2];
  
  // Mesh resolution of the triangle
  int neq=NCells;
  
  // Create temprary arraies
  Point* side12=new Point[neq+1];
  Point* side13=new Point[neq+1];
  
  // Generate crack nodes 
  for(j=0; j<=neq; j++) {
    side12[j]=p1+(p2-p1)*(float)j/neq;
    side13[j]=p1+(p3-p1)*(float)j/neq;
  }
  for(j=0; j<=neq; j++) {
    for(i=0; i<=j; i++) {
      double w=0.0;
      if(j!=0) w=(float)i/j;
      pt=side12[j]+(side13[j]-side12[j])*w;
      cx.push_back(pt);
    } 
  } 
  delete [] side12;
  delete [] side13;    
  
  // Generate crack elements 
  for(j=0; j<neq; j++) {
    nstart1=nstart0+j*(j+1)/2;
    nstart2=nstart0+(j+1)*(j+2)/2;
    for(i=0; i<j; i++) {
      // left element
      n1=nstart1+i;  n2=nstart2+i;  n3=nstart2+(i+1);
      ce.push_back(IntVector(n1,n2,n3));
      // right element
      n1=nstart1+i;  n2=nstart2+(i+1);  n3=nstart1+(i+1);
      ce.push_back(IntVector(n1,n2,n3));
    } 
    n1=nstart0+(j+1)*(j+2)/2-1;
    n2=nstart0+(j+2)*(j+3)/2-2;
    n3=nstart0+(j+2)*(j+3)/2-1;
    ce.push_back(IntVector(n1,n2,n3));
  } 
  nstart0+=(neq+1)*(neq+2)/2;
  
  // Collect crack-front nodes
  int seg0=0;
  for(j=0; j<3; j++) {
    if(!AtFront[j]) { seg0=j+1; break; }
  }
  
  for(int l=0; l<3; l++) { // Loop over sides of the triangle
    j=seg0+l;
    if(j>2) j-=3;
    if(AtFront[j]) {
      int j1 = (j!=2 ? j+1 : 0);
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

