/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  Delaunay.cc
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#include "Delaunay.h"
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <Core/CCA/spec/cca_sidl.h>
using namespace std;

Delaunay::Delaunay()
{
	next=0;
	//add four vertices as the bounding box vertices
	//addNode(vector2d(-1,-1));
	//addNode(vector2d( 3,-1));
	//addNode(vector2d(-1, 3));
	//addNode(vector2d( 3, 3));

}

//this constructor is not used
Delaunay::Delaunay(const SSIDL::array1<double> &nodes1d, 
		   const SSIDL::array1<int> &boundaries1d)
{
  std::vector<int> pts;
  for(unsigned int i=0; i<boundaries1d.size();i++){
    int index=boundaries1d[i]+4;
    if(pts.size()>0 && index==pts[0]){
      addBoundary(pts);
      pts.clear();
    }
    else
      pts.push_back(index);
    //cerr<<index<<endl;
  }

  double inf=1e300;
  xmin=inf;
  xmax=-inf;
  ymin=inf;
  ymax=-inf;
  //reserve the first 4 nodes for bounding box vertices
  nodes.push_back(vector2d(0,0));
  nodes.push_back(vector2d(0,0));
  nodes.push_back(vector2d(0,0));
  nodes.push_back(vector2d(0,0));
  
  for(unsigned int i=0; i<nodes1d.size(); i+=2){
    vector2d v(nodes1d[i],nodes1d[i+1]);
    nodes.push_back(v);
    if(v.x < xmin) xmin=v.x;
    if(v.x > xmax) xmax=v.x;
    if(v.y < ymin) ymin=v.y;
    if(v.y > ymax) ymax=v.y;
  }
  //cerr<<"#of nodes="<<nodes.size()<<endl;
  double dx=xmax-xmin;
  double dy=ymax-ymin;
  nodes[0]=vector2d(xmin-dx, ymin-dy);
  nodes[1]=vector2d(xmin-dx, ymax+dy);
  nodes[2]=vector2d(xmax+dx, ymax+dy);
  nodes[3]=vector2d(xmax+dx, ymin-dy);

  next=0;
}

//add one node
void Delaunay::readNodes(istream &is)
{
  double inf=1e300;
  xmin=inf;
  xmax=-inf;
  ymin=inf;
  ymax=-inf;
  //reserve the first 4 nodes for bounding box vertices
  nodes.push_back(vector2d(0,0));
  nodes.push_back(vector2d(0,0));
  nodes.push_back(vector2d(0,0));
  nodes.push_back(vector2d(0,0));
  
  while(!is.eof()){
    vector2d v;
    is>>v;
    nodes.push_back(v);
    if(v.x < xmin) xmin=v.x;
    if(v.x > xmax) xmax=v.x;
    if(v.y < ymin) ymin=v.y;
    if(v.y > ymax) ymax=v.y;
  }
  //cerr<<"#of nodes="<<nodes.size()<<endl;
  double dx=xmax-xmin;
  double dy=ymax-ymin;
  nodes[0]=vector2d(xmin-dx, ymin-dy);
  nodes[1]=vector2d(xmin-dx, ymax+dy);
  nodes[2]=vector2d(xmax+dx, ymax+dy);
  nodes[3]=vector2d(xmax+dx, ymin-dy);
		   
}

//add boundary
void Delaunay::addBoundary(const std::vector<int> &bpts)
{
  Boundary b;
  b.outer= boundaries.size()==0;
  b.index=bpts;
  boundaries.push_back(b);
}

//test if triangle itri contains node ip
bool Delaunay::nodeInTriangle(int ip, int itri)
{
	int i1=triangles[itri].index[0];
	int i2=triangles[itri].index[1];
	int i3=triangles[itri].index[2];

	vector2d a=nodes[i2]-nodes[i1];
	vector2d b=nodes[i3]-nodes[i1];
	vector2d c=nodes[ip]-nodes[i1];

	double A=a%a;
	double C=a%b;
	double B=b%b;
	double C1=a%c;
	double C2=b%c;
	
	double p=(C*C1-A*C2)/(C*C-B*A);
	double q=(C*C2-B*C1)/(C*C-B*A);

	//cerr<<"p,q="<<p<<" "<<q<<endl;
	return p>=0 && q>=0 && p+q<=1;		
}

//test if circle itri contains node ip
bool Delaunay::nodeInCircle(int ip, int itri)
{
	double d2=(circles[itri].center-nodes[ip]).squaredLength();
	return d2<circles[itri].radius*circles[itri].radius;
}


//this method implements the main steps of delauney algrithm
bool Delaunay::triangulation()
{
  if(nodes.size()<3) return false;
  if(next==0){
    triangles.push_back(Triangle(0,1,2));
    triangles.push_back(Triangle(2,3,0));
    circles.push_back(getCircle(0));
    circles.push_back(getCircle(1));
    next+=4;
  }

  while(next<nodes.size()){
  //if(next<nodes.size()){
    int T=-1;
    std::vector<int> C;
    for(unsigned i=0; i<triangles.size(); i++){
      if(nodeInCircle(next,i)){
	C.push_back(i);
	if(nodeInTriangle(next,i)){
	  T=i;
	}
      }
    }
     /*
    std::vector<int> Cb=C;
    
    //cerr<<"node in triangle"<<endl;
    for(int je=0; je<3; je++)
      checkBoundary(C,T,je,false);
    
        //cerr<<"C.size :"<<C.size()<< " ->";
    
    for(int i=C.size()-1; i>=0; i--){
      if(C[i]==-1){
	//cerr<<"erased triangle="<<triangles[Cb[i]].index[0]-4<<" "<<
	//  triangles[Cb[i]].index[1]-4<<" "<<triangles[Cb[i]].index[2]-4<<endl;
	C.erase(C.begin()+i);
	
      }
      }
    //cerr<<C.size()<<endl;
     */
      


    std::vector<Edge> E; //external edges of C
    for(unsigned int i=0; i<C.size(); i++){
      for(int ie=0; ie<3; ie++){
	Edge e1=triangles[C[i]].getEdge(ie);
	bool isExternal=true;
	for(unsigned int j=0; j<C.size() && isExternal; j++){
	  if(i==j)continue;	
	  for(int je=0; je<3; je++){
	    Edge e2=triangles[C[j]].getEdge(je);
	    if(e1==e2){
	      //cerr<<"common edge found"<<endl;	
	      isExternal=false;
	      break;
	    }
	  }
	}
	if(isExternal) E.push_back(e1);
      }
    }
      
    //remove the triangles whose circumcircles contains the new point
    for(int i=C.size()-1; i>=0; i--){	
      triangles.erase(triangles.begin()+C[i]);
      circles.erase(circles.begin()+C[i]);
    }
      
    //connect the new point to each external edge
    for(unsigned i=0;i<E.size(); i++){
				//cerr<<"E(i)="<<E[i].index[0]<<" "<<E[i].index[1]<<endl;
      triangles.push_back(Triangle(E[i].index[0],E[i].index[1],next));				
      circles.push_back(getCircle( triangles.size()-1 ));			
    }
     next++;
  }
  

  /*  //output the mesh
  ofstream fs("output.mes");
  for(unsigned i=0; i<triangles.size();i++){
    if(triangles[i].index[0]>=4 && triangles[i].index[1]>=4 && triangles[i].index[0]>=4){	
      fs<<triangles[i].index[0]-4<<"\t"<<triangles[i].index[1]-4<<"\t"<<triangles[i].index[2]-4<<endl;
    }	
  }	
  */
  

  std::vector<Edge> miss;
  for(unsigned int ib=0; ib<boundaries.size(); ib++){
    for(unsigned int i=1; i<boundaries[ib].index.size(); i++){
      bool missedEdge=true;
      Edge e(boundaries[ib].index[i-1],boundaries[ib].index[i]);
      
      for(unsigned int it=0;it<triangles.size();it++){
	if(triangles[it].contains(e)){
	  missedEdge=false;
	  break;
	}
      }
      if(missedEdge){
	for(unsigned int k=0; k<miss.size(); k++){
	  if(e==miss[k]){
	    missedEdge=false;
	    break;
	  }
	}
      }
      if(missedEdge){
	miss.push_back(e);
      }
    }
  }

  //cerr<<"missed boundary edges are:"<<endl;
  for(unsigned int i=0; i<miss.size();i++)
    //cerr<<miss[i].index[0]-4<<"--"<<miss[i].index[1]-4<<endl;

  for(unsigned int mi=0; mi<miss.size(); mi++){

  std::vector<Edge> allEdges;
  for(unsigned int i=0; i<triangles.size(); i++){
    for(int ie=0;ie<3;ie++){
      Edge e=triangles[i].getEdge(ie);
      bool newEdge=!isBoundary(e);
      if(e.index[0]<4 || e.index[1]<4)newEdge=false;
      if(newEdge){
	for(unsigned k=0; k<allEdges.size();k++){
	  if(e==allEdges[k]){
	    newEdge=false;
	    break;
	  }
	}
	if(newEdge) allEdges.push_back(e);
      }
      
    }
  }

  std::vector<Edge> intEdges;
  std::vector<double> dist;
  //cerr<<"all non-boundary edges are:"<<endl;
  //for(unsigned int i=0; i<allEdges.size();i++)
  //  cerr<<allEdges[i].index[0]-4<<"--"<<allEdges[i].index[1]-4<<endl;
  
 
  for(int i=allEdges.size()-1; i>=0;i--){
     double r=intersected(miss[mi],allEdges[i]);
    if(r>0){
      intEdges.push_back(allEdges[i]);
      dist.push_back(r);
    }
  }
  //cerr<<"all intersected edges are:"<<endl;
  // for(unsigned int i=0; i<intEdges.size();i++)
  //  cerr<<intEdges[i].index[0]-4<<"--"<<intEdges[i].index[1]-4<<endl;

  double min=1e300;
  int id=0;
 

  while( intEdges.size()>0){

    for(unsigned int i=0; i<intEdges.size(); i++){
      if(min>dist[i]){
	id=i;
	min=dist[i];
      }
    }
    Edge diag=intEdges[id]; 
    intEdges.erase(intEdges.begin()+id);

    //find 2 triangles of common edge diag
    int tri[2];
    int cnt=0;
    for(unsigned int i=0; i<triangles.size(); i++){
      if(triangles[i].contains(diag)){
	tri[cnt]=i;
	if(++cnt==2) break;
      }
    }

    int vt[2];
    for(int i=0; i<2; i++){
      for(int j=0; j<3; j++){
	if(triangles[tri[i]].index[j] != diag.index[0] &&
	   triangles[tri[i]].index[j] != diag.index[1]){
	  vt[i]=triangles[tri[i]].index[j];
	  break;
	}
      }
    }
    //cerr<<"vt[]="<<vt[0]-4<<" "<<vt[1]-4<<endl;
    //cerr<<"diag="<<diag.index[0]-4<<" "<<diag.index[1]-4<<endl;

    triangles[tri[0]]=Triangle(vt[0], diag.index[0], vt[1]);
    triangles[tri[1]]=Triangle(vt[1], diag.index[1], vt[1]);
  }

  } //mi

  triangles[0].type=0;
  setColor(0); //setColor for all triangles;
  //erase triangles with zero area or with  color=0
  for(int i=triangles.size()-1; i>=0; i--){
    double x[3];
    double y[3];
    for(int k=0;k<3;k++){
      x[k]=nodes[triangles[i].index[k] ].x;
      y[k]=nodes[triangles[i].index[k] ].y;
    }
    double A2=(x[1]*y[2]-x[2]*y[1])-(x[0]*y[2]-x[2]*y[0])+(x[0]*y[1]-x[1]*y[0]);

    //cerr<<"i="<<i<<endl;
    if(triangles[i].type==0 || fabs(A2)<1e-10 ){
      //cerr<<"erased"<<" A2="<<fabs(A2)<<endl;
      triangles.erase(triangles.begin()+i);
      circles.erase(circles.begin()+i);
    }
  }

  
  return true;
}


//return the circumcircle of triangle itri
Circle Delaunay::getCircle(int itri)
{
	Circle circle;

	vector2d v1=nodes[triangles[itri].index[0]];
	vector2d v2=nodes[triangles[itri].index[1]];
	vector2d v3=nodes[triangles[itri].index[2]];

	//compute the the center
	double L1=v1.squaredLength();
	double L2=v2.squaredLength();
	double L3=v3.squaredLength();

	double N1=(L2-L1)*(v3.y-v1.y)-(L3-L1)*(v2.y-v1.y);
	double N2=(L3-L1)*(v2.x-v1.x)-(L2-L1)*(v3.x-v1.x);
	double D =(v2.x-v1.x)*(v3.y-v1.y)-(v3.x-v1.x)*(v2.y-v1.y);

	double x=N1/(2*D);
	double y=N2/(2*D);
	circle.center=vector2d(x,y);

	//compute the radius
	double a=(v1-v2).length();
	double b=(v2-v3).length();
	double c=(v3-v1).length();
	double s=(a+b+c)/2;
	double r=a*b*c/(4*sqrt(s*(s-a)*(s-b)*(s-c)));
	circle.radius=r;
	return circle;
}

std::vector<vector2d> Delaunay::getNodes()
{
	return nodes;
}

std::vector<Triangle> Delaunay::getTriangles()
{
	return triangles;
}

std::vector<Circle> Delaunay::getCircles()
{
	return circles;
}

std::vector<Boundary> Delaunay::getBoundaries()
{
  return boundaries;
}

double Delaunay::minX()
{
  return xmin;
}

double Delaunay::minY()
{
  return ymin;
}

double Delaunay::width()
{
  return xmax-xmin;
}

double Delaunay::height()
{
  return ymax-ymin;
}

bool Delaunay::isBoundary(Edge e)
{
  for(unsigned int i=0; i<boundaries.size(); i++)
    if(boundaries[i].contains(e))return true;
  return false;
}

void Delaunay::checkBoundary(std::vector<int> &C, int T, int ie, bool remove)
{

  //check/remove boundary edges beyond ie-th edge of T 
  Edge e=triangles[T].getEdge(ie);
  remove|=isBoundary(e);
  for(unsigned int t=0; t<C.size(); t++){
    if(C[t]!=-1 && triangles[C[t]].contains(e) && C[t]!=T){
      for(int i=0; i<3; i++){
	if(!triangles[T].contains(triangles[C[t]].getEdge(i)) ){
	  checkBoundary(C, C[t],i, remove) ;  //check/remove outer triangles
	}
      }
      if(remove) C[t]=-1; //remove C[t];
    }  
  }
}

double Delaunay::intersected(Edge e1, Edge e2)
{
  vector2d a=nodes[e1.index[0]];
  vector2d v1=nodes[e1.index[1]]-a;

  vector2d b=nodes[e2.index[0]];
  vector2d v2=nodes[e2.index[1]]-b;
  
  vector2d c=b-a;

  double a1=v1.x;
  double a2=v1.y;
  double b1=-v2.x;
  double b2=-v2.y;
  double c1=c.x;
  double c2=c.y;

  double d=a1*b2-a2*b1;

  double eps=1e-100;

  if(fabs(d)<eps) return -1;
  
  double r1= (a1*c2-a2*c1)/d;
  double r2= (c1*b2-c2*b1)/d;

  //cerr<<"r1 r2="<<r1<<" "<<r2<<endl;

  if(r1 < 1-eps && r1>eps && 
     r2 < 1-eps && r2>eps) return r1;

  return -1;

}

void Delaunay::setColor(int startTri)
{
  int neighbour[3]={-1,-1,-1};
  Edge e[3];
  for(int i=0; i<3; i++) e[i]=triangles[startTri].getEdge(i);

  int cnt=0;
  for(unsigned int i=0; i<triangles.size(); i++){
    if(triangles[i].type!=-1) continue;
    for(int j=0; j<3; j++){
      if(neighbour[j]==-1 && triangles[i].contains(e[j]) ){
	neighbour[j]=i;
	if(isBoundary(e[j])){
	  triangles[i].type=!triangles[startTri].type;
	}
	else{
	  triangles[i].type=triangles[startTri].type;
	}
	cnt++;
      }
    }
    if(cnt>=3) break;  
  }
  
  for(int i=0; i<3; i++){
    if(neighbour[i]!=-1) setColor(neighbour[i]);
  }
}


