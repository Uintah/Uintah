// UnstructuredGeom.cc - Geometries that live in a unstructured space
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/UnstructuredGeom.h>

namespace SCICore{
namespace Datatypes{


NodeSimp::NodeSimp(){
}


NodeSimp::NodeSimp(const Point& ip):
p(ip){
}

NodeSimp::~NodeSimp(){
}

void NodeSimp::draw(double radius, const MaterialHandle& matl,
		    GeomGroup* group){
  group->add(scinew GeomMaterial(scinew GeomSphere(p, radius), matl));
}
  
//////////////////////////////////////

EdgeSimp::EdgeSimp(){
}

EdgeSimp::EdgeSimp(int a, int b){
  nodes[0] = a;
  nodes[1] = b;
}

EdgeSimp::~EdgeSimp()
{}

  
bool EdgeSimp::operator==(const EdgeSimp& iedge) const{
  // Compare iedge to this
  return ((iedge.nodes[0] == this->nodes[0] &&
	   iedge.nodes[1] == this->nodes[1]) ||
	  (iedge.nodes[0] == this->nodes[1] &&
	   iedge.nodes[1] == this->nodes[0]));
}

bool EdgeSimp::operator<(const EdgeSimp& iedge) const{
  return (this->nodes[0] < iedge.nodes[0] ||
	  (this->nodes[0] == iedge.nodes[0] &&
	   this->nodes[1] < iedge.nodes[1]));

}

/////////////////////////////////////////////

FaceSimp::FaceSimp(){
}

FaceSimp::FaceSimp(int a, int b, int c){
  nodes[0] = a;
  nodes[1] = b;
  nodes[2] = c;
}

FaceSimp::~FaceSimp()
{}

////////////////////////////////////////////

TetSimp::TetSimp(){
}

TetSimp::TetSimp(int a, int b, int c, int d){
  nodes[0] = a;
  nodes[1] = b;
  nodes[2] = c;
  nodes[3] = d;
}

TetSimp::~TetSimp()
{}

  
bool TetSimp::draw(const vector<NodeSimp>& inodes, GeomTrianglesP* group){
  if(!group){
    return false;
  }
  Point p1(inodes[nodes[0]].p);
  Point p2(inodes[nodes[1]].p);
  Point p3(inodes[nodes[2]].p);
  Point p4(inodes[nodes[3]].p);
  group->add(p1, p2, p3);
  group->add(p1, p2, p4);
  group->add(p1, p3, p4);
  group->add(p2, p3, p4);
  return true;
}

}
}
