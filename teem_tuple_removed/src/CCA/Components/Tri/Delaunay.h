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
 *  Delaunay.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

//class  Edge, Triangle, Circle, Delauney

#ifndef delaunay_h
#define delaunay_h

#include "vector2d.h"
#include <Core/CCA/spec/cca_sidl.h>


class Edge{
public:
  Edge(){};
  Edge(int i0, int i1) {index[0]=i0; index[1]=i1;}
  //test if two edges are identical
  bool operator==(Edge e){
    return (this->index[0]==e.index[0] && this->index[1]==e.index[1]) ||
      (this->index[0]==e.index[1] && this->index[1]==e.index[0]);
  }
  int index[2];
};


class Triangle{
public:
	Triangle(int i0, int i1, int i2) {
	  index[0]=i0; index[1]=i1; index[2]=i2; 
	  type=-1; 
	}
	Edge getEdge(int ie){return Edge(index[ie],index[(ie+1)%3]); }
	bool contains(Edge e){
	  for(int i=0;i<3;i++) if(getEdge(i)==e) return true;
	  return false;
	}
	int type;  //-1 unkown 0 ghost 1 solid
 	int index[3];

};



class Circle{
public:
	vector2d center;
	double radius;	
};

class Boundary{
 public:
  bool outer;
  std::vector<int> index;
  bool contains(Edge e){

    for(unsigned int i=1; i<index.size();i++){
      if(e==Edge(index[i],index[i-1]) ) return true;
    }  
    if(index.size()>2 && e==Edge(index[0], index[index.size()-1])) return true;
    return false;
  } 
  

};


class Delaunay{
public:
  Delaunay(const SSIDL::array1<double> &nodes1d, const SSIDL::array1<int> &boundaries1d);
  Delaunay();

  void readNodes(std::istream &s);	//read nodes from a stream
  std::vector<vector2d> getNodes();
  std::vector<Circle> getCircles();
  std::vector<Triangle> getTriangles();
  bool triangulation();
  std::vector<Boundary> getBoundaries();
  void addBoundary(const std::vector<int> &bpts);
  void toggleShowCircles();
  double width();
  double height();
  double minX();
  double minY();
  bool isBoundary(Edge e);

  //this method removes some elements so that the boundary edge 
  //will be the external edge
  void checkBoundary(std::vector<int> &C, int T, int ie, bool remove);
  double intersected(Edge e1, Edge e2);
  void setColor(int startTri);
private:
  bool nodeInTriangle(int ip, int itri);
  bool nodeInCircle(int ip, int itri);
  Circle getCircle(int itri);
  std::vector<vector2d> nodes;
  std::vector<Triangle> triangles;	// both triangles and
  std::vector<Circle> circles;  //   circles have the same inices
  std::vector<Boundary> boundaries; //boundaries 

  unsigned int next; //index of next node to be inserted into the mesh
  double xmin, xmax, ymin, ymax;
};



#endif
