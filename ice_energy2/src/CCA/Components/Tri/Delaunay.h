/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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
