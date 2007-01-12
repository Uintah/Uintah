#include <iostream>
#include <fstream>
#include <vector>

#include <Packages/Wangxl/Core/Datatypes/Mesh/BEdge.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/BFace.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/DVertex.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/Triple.h>
#include <Packages/Wangxl/Core/ThirdParty/triangle.h>

namespace Wangxl {

using namespace SCIRun;

using std::vector;

BFace::BFace() { 
    d_bedges[0] = d_bedges[1] = d_bedges[2] = 0;
    d_v[0] = d_v[1] = d_v[2] = 0;
}

BFace::BFace(DVertex* v0, DVertex* v1, DVertex* v2) {
    d_v[0] = v0;
    d_v[1] = v1;
    d_v[2] = v2;
    d_bedges[0] = d_bedges[1] = d_bedges[2] = 0;
    set_2D();
}

void BFace::add_edge(BEdge* be) {
    if ( d_bedges[0] == 0 ) d_bedges[0] = be;
    else if ( d_bedges[1] == 0 ) d_bedges[1] = be;
    else if ( d_bedges[2] == 0 ) d_bedges[2] = be;
    else {      
      std::cout << " Error!!!!!!!!!!!, Invalid surface mesh" << std::endl; assert(false);
    }
}

// add a new vertex to the middle of this boundary face
void BFace::add_vertex(DVertex* v) {
    d_vertices.push_back(v);
}

bool BFace::is_split() {
    for ( int i = 0; i < 3; i++ ) if ( d_bedges[i]->is_split() ) return true;
    if ( !d_vertices.empty() ) return true;
    return false;
}

bool BFace::is_root() {
    for ( int i = 0; i < 3; i++ ) if ( !d_bedges[i]->is_root() ) return false;
    return true;
}

void BFace::get_new_faces(vector< triple<DVertex*, DVertex*, DVertex*> >& nfaces, vector<int>& neighbors) {
    vector<DVertex*> vertices, bvertices;
    vector<DVertex*>::const_iterator it;
    vector<DVertex*>::reverse_iterator rit;
    std::ofstream offFile("data/face.off");
    offFile << "OFF" << std::endl;
    struct triangulateio in,out;
    int i;
    double x, y;

    // cllect all vertices on the boundary edges
    for ( i = 0; i < 3; i++ ) {
      bvertices.clear();
      d_bedges[i]->get_vertices(bvertices);
      for ( it = bvertices.begin(); it != bvertices.end(); it++ ) std::cout << std::hex << *it << " ";
      std::cout << std::endl;
      if ( vertices.empty() ) vertices = bvertices;
      else {
	if ( bvertices.front() == vertices.back() ) {
	  for ( it = bvertices.begin()+1; it != bvertices.end(); it++ )
	    vertices.push_back(*it);
	}
	else if ( bvertices.back() == vertices.back() ) {
	  for ( rit = bvertices.rbegin()+1; rit != bvertices.rend(); rit++ )
	    vertices.push_back(*rit);
	}
	else if ( bvertices.front() == vertices.front() ) {
	  for ( it = bvertices.begin()+1; it != bvertices.end(); it++ )
	    vertices.insert(vertices.begin(),*it);
	}
	else if ( bvertices.back() == vertices.front() ) {
	  for ( rit = bvertices.rbegin()+1; rit != bvertices.rend(); rit++ )
	    vertices.insert(vertices.begin(),*rit);
	}
	else {
	  std::cout << " Error!!!!!!!!!!!!! in 2D triangulation" << std::endl;
	  assert(false);
	}
      }
    }
    vertices.pop_back(); // remove the last vertices which is all readay at the front
    // add vertices not on the boundary edges
    in.numberofsegments = vertices.size();
    for ( it = d_vertices.begin(); it != d_vertices.end(); it++ ) vertices.push_back(*it);
    /*    for ( it = vertices.begin(); it != vertices.end(); it++ ) {
      Point p = (*it)->point();
      std::cout << std::hex << *it << " " << std::dec << p.x() << " " << p.y() << " " << p.z() << std::endl;
      }*/
    std::cout << std::endl;


    // prepare for the 2D triangulation of this face
    in.numberofpoints = vertices.size();
    in.numberofpointattributes = 0;
    in.numberofholes = 0;
    in.numberofregions = 0;
    in.pointlist = (REAL*)malloc(in.numberofpoints*2*sizeof(REAL));
    in.segmentlist = (int*)malloc(in.numberofpoints*2*sizeof(int));
    in.pointmarkerlist = NULL;
    in.segmentmarkerlist = NULL;
    for ( i = 0; i < in.numberofpoints; i++ ) {
      get_2D(vertices[i]->point(), x, y);
      in.pointlist[2*i] = x;
      in.pointlist[2*i+1] = y;
      if ( i < in.numberofsegments ) {
	in.segmentlist[2*i] = i;
	in.segmentlist[2*i+1] = (i+1)%in.numberofsegments;
      }
    }
    out.trianglelist = (int*)NULL;
    out.segmentlist = (int*)NULL; 
    out.segmentmarkerlist = (int*)NULL;
    out.neighborlist = (int*)NULL;
    triangulate("pzNn", &in, &out,(struct triangulateio*)NULL); 
    //   for ( i = 0; i <= out.numberoftriangles; i++)
   //cout << "Triangle " << dec << i << "'s neighbors " << out.neighborlist[i*3] << " " << out.neighborlist[i*3+1] << " " << out.neighborlist[i*3+2] << endl;
   //cout << endl;
    offFile << in.numberofpoints << " " << out.numberoftriangles << " 0" << std::endl;
    for ( i = 0; i < in.numberofpoints; i++ )
      offFile << vertices[i]->point().x() << " " << vertices[i]->point().y() << " " << vertices[i]->point().z() << std::endl;
   for ( i = 0; i < out.numberoftriangles; i++) {
     neighbors.push_back(out.neighborlist[i*3]);
     neighbors.push_back(out.neighborlist[i*3+1]);
     neighbors.push_back(out.neighborlist[i*3+2]);
     offFile << "3 " << out.trianglelist[i*out.numberofcorners+0] << " " << out.trianglelist[i*out.numberofcorners+1] << " " << out.trianglelist[i*out.numberofcorners+2] << std::endl;
      nfaces.push_back(make_triple( vertices[out.trianglelist[i*out.numberofcorners+0]], vertices[out.trianglelist[i*out.numberofcorners+1]], vertices[out.trianglelist[i*out.numberofcorners+2]]));
   }
}

void BFace::get_split_edges(BEdge* bedges[3]) {
  Point mp = get_split_point();
  Point p0, p1;
  int i, j = 0;
  double a;
  for ( i = 0; i < 3; i++ ) {
    p0 = d_bedges[i]->source()->point();
    p1 = d_bedges[i]->target()->point();
    a =  Dot(p0-mp,p1-mp);
    if ( Dot(p0-mp,p1-mp) <= 0.0 ) bedges[j++] = d_bedges[i];
  }
}

void BFace::set_2D() {
  Point p, q, r;
  Vector u, v, w, uv, tmp;
  p = d_v[0]->point();
  q = d_v[1]->point();
  r = d_v[2]->point();
  u = q-p;
  tmp = r-p;
  w = Cross(u,tmp);
  w.normalize();
  u.normalize();
  v = Cross(w,u);
  trans[0][0] = u.x(); trans[0][1] = v.x(); trans[0][2] = w.x();
  trans[1][0] = u.y(); trans[1][1] = v.y(); trans[1][2] = w.y();
  trans[2][0] = u.z(); trans[2][1] = v.z(); trans[2][2] = w.z();
  trans[3][0] = -Dot(u,p); trans[3][1] = -Dot(v,p); trans[3][2] = -Dot(w,p);
}

void BFace::get_2D(const Point& p, double& x, double& y) {
    x = p.x() * trans[0][0] + p.y() * trans[1][0] + p.z() * trans[2][0] + trans[3][0];
    y = p.x() * trans[0][1] + p.y() * trans[1][1] + p.z() * trans[2][1] + trans[3][1];
    double z = p.x() * trans[0][2] + p.y() * trans[1][2] + p.z() * trans[2][2] + trans[3][2];
}

Point BFace::get_split_point() {
  double d0,d1,d2;
  double c,c0,c1,c2;
  Point p0 = d_v[0]->point();
  Point p1 = d_v[1]->point();
  Point p2 = d_v[2]->point();
  d0 = Dot(p2-p0,p1-p0);
  d1 = Dot(p2-p1,p0-p1);
  d2 = Dot(p0-p2,p1-p2);
  c0 = d1*d2; c1 = d2*d0; c2 = d0*d1; c = c0+c1+c2;
  return Point(((p0*(c1+c2)).asVector()+(p1*(c2+c0)).asVector()+(p2*(c0+c1)).asVector())/(2*c));
  /*  double perimeter = (p0-p1).length() +(p1-p2).length() +(p2-p0).length();
      return Point((p0.asVector()*(p1-p2).length() +  p1.asVector()*(p2-p0).length() +  p2.asVector()*(p0-p1).length())/perimeter);*/

}

}



