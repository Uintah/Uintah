//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : Brick.cc
//    Author : Milan Ikits
//    Date   : Wed Jul 14 16:03:05 2004

#include <cmath>
#include <Packages/Volume/Core/Datatypes/Brick.h>
#include <Packages/Volume/Core/Util/Utils.h>

#include <iostream>
using std::cerr;
using std::endl;

using namespace SCIRun;

namespace Volume {

Brick::Brick (int nx, int ny, int nz, int nc, int* nb, int ox, int oy, int oz,
              int mx, int my, int mz, const BBox& bbox, const BBox& tbox)
  : nx_(nx), ny_(ny), nz_(nz), nc_(nc), ox_(ox), oy_(oy), oz_(oz),
    mx_(mx), my_(my), mz_(mz), bbox_(bbox), tbox_(tbox), dirty_(true)
{
  for(int c=0; c<nc_; c++) {
    nb_[c] = nb[c];
  }

  /* The cube is numbered in the following way 
     
       2________6        y
      /|        |        |  
     / |       /|        |
    /  |      / |        |
   /   0_____/__4        |
  3---------7   /        |_________ x
  |  /      |  /         /
  | /       | /         /
  |/        |/         /
  1_________5         /
                     z  
  */

  // set up vertices
  Point pmin(bbox_.min());
  Point pmax(bbox_.max());
  corner_[0] = pmin;
  corner_[1] = Point(pmin.x(), pmin.y(), pmax.z());
  corner_[2] = Point(pmin.x(), pmax.y(), pmin.z());
  corner_[3] = Point(pmin.x(), pmax.y(), pmax.z());
  corner_[4] = Point(pmax.x(), pmin.y(), pmin.z());
  corner_[5] = Point(pmax.x(), pmin.y(), pmax.z());
  corner_[6] = Point(pmax.x(), pmax.y(), pmin.z());
  corner_[7] = pmax;
  // set up texture coordinates
  Point texture[8];
  Point tmin(tbox_.min());
  Point tmax(tbox_.max());
  texture[0] = Point(tmin.x(), tmin.y(), tmin.z());
  texture[1] = Point(tmin.x(), tmin.y(), tmax.z());
  texture[2] = Point(tmin.x(), tmax.y(), tmin.z());
  texture[3] = Point(tmin.x(), tmax.y(), tmax.z());
  texture[4] = Point(tmax.x(), tmin.y(), tmin.z());
  texture[5] = Point(tmax.x(), tmin.y(), tmax.z());
  texture[6] = Point(tmax.x(), tmax.y(), tmin.z());
  texture[7] = Point(tmax.x(), tmax.y(), tmax.z());
  // set up edges
  edge_[0] = Ray(corner_[0], corner_[2] - corner_[0]);
  edge_[1] = Ray(corner_[2], corner_[6] - corner_[2]);
  edge_[2] = Ray(corner_[4], corner_[6] - corner_[4]);
  edge_[3] = Ray(corner_[0], corner_[4] - corner_[0]);
  edge_[4] = Ray(corner_[1], corner_[3] - corner_[1]);
  edge_[5] = Ray(corner_[3], corner_[7] - corner_[3]);
  edge_[6] = Ray(corner_[5], corner_[7] - corner_[5]);
  edge_[7] = Ray(corner_[1], corner_[5] - corner_[1]);
  edge_[8] = Ray(corner_[0], corner_[1] - corner_[0]);
  edge_[9] = Ray(corner_[2], corner_[3] - corner_[2]);
  edge_[10] = Ray(corner_[6], corner_[7] - corner_[6]);
  edge_[11] = Ray(corner_[4], corner_[5] - corner_[4]);
  // set up texture coordinate edges
  tex_edge_[0] = Ray(texture[0], texture[2] - texture[0]);
  tex_edge_[1] = Ray(texture[2], texture[6] - texture[2]);
  tex_edge_[2] = Ray(texture[4], texture[6] - texture[4]);
  tex_edge_[3] = Ray(texture[0], texture[4] - texture[0]);
  tex_edge_[4] = Ray(texture[1], texture[3] - texture[1]);
  tex_edge_[5] = Ray(texture[3], texture[7] - texture[3]);
  tex_edge_[6] = Ray(texture[5], texture[7] - texture[5]);
  tex_edge_[7] = Ray(texture[1], texture[5] - texture[1]);
  tex_edge_[8] = Ray(texture[0], texture[1] - texture[0]);
  tex_edge_[9] = Ray(texture[2], texture[3] - texture[2]);
  tex_edge_[10] = Ray(texture[6], texture[7] - texture[6]);
  tex_edge_[11] = Ray(texture[4], texture[5] - texture[4]);
}

Brick::~Brick()
{}

// compute polygon of edge plane intersections
void
Brick::compute_polygon(const Ray& view, double t,
                       Array1<float>& vertex, Array1<float>& texcoord,
                       Array1<int>& size) const
{
  compute_polygons(view, t, t, 1.0, vertex, texcoord, size);
}

void
Brick::compute_polygons(const Ray& view, double dt,
                        Array1<float>& vertex, Array1<float>& texcoord,
                        Array1<int>& size) const
{
  Point corner[8];
  corner[0] = bbox_.min();
  corner[1] = Point(bbox_.min().x(), bbox_.min().y(), bbox_.max().z());
  corner[2] = Point(bbox_.min().x(), bbox_.max().y(), bbox_.min().z());
  corner[3] = Point(bbox_.min().x(), bbox_.max().y(), bbox_.max().z());
  corner[4] = Point(bbox_.max().x(), bbox_.min().y(), bbox_.min().z());
  corner[5] = Point(bbox_.max().x(), bbox_.min().y(), bbox_.max().z());
  corner[6] = Point(bbox_.max().x(), bbox_.max().y(), bbox_.min().z());
  corner[7] = bbox_.max();
  double t[8];
  for(int i=0; i<8; i++) {
    t[i] = Dot(corner[i]-view.origin(), view.direction());
  }
  Sort(t, 8);
  double tmin = (floor(t[0]/dt) + 1)*dt;
  double tmax = floor(t[7]/dt)*dt;
  compute_polygons(view, tmin, tmax, dt, vertex, texcoord, size);
}

// compute polygon list of edge plane intersections
void
Brick::compute_polygons(const Ray& view, double tmin, double tmax, double dt,
                        Array1<float>& vertex, Array1<float>& texcoord,
                        Array1<int>& size) const
{
  Vector vv[6], tt[6]; // temp storage for vertices and texcoords
  double t = tmax; // start at tmax
  int k = 0, degree = 0;

  
  // find up and right vectors
  Vector vdir = view.direction();
  Vector up;
  Vector right;
  switch(MinIndex(std::abs(vdir.x()),
                  std::abs(vdir.y()),
                  std::abs(vdir.z()))) {
  case 0:
    up.x(0.0); up.y(-vdir.z()); up.z(vdir.y());
    break;
  case 1:
    up.x(-vdir.z()); up.y(0.0); up.z(vdir.x());
    break;
  case 2:
    up.x(-vdir.y()); up.y(vdir.x()); up.z(0.0);
    break;
  }
  up.normalize();
  right = Cross(vdir, up);
  // we compute polys back to front
  while(t >= tmin) {
    // find intersections
    degree = 0;
    for(int j=0; j<12; j++) {
      double u;
      bool isec =
        edge_[j].planeIntersectParameter(-view.direction(), view.parameter(t), u);
      if(isec && u >= 0.0 && u <= 1.0) {
        vv[degree] = (Vector)(edge_[j].parameter(u));
        tt[degree] = (Vector)(tex_edge_[j].parameter(u));
        degree++;
      }
    }
    // 
    if(degree > 3) {
      // compute centroids
      Vector vc(0.0, 0.0, 0.0), tc(0.0, 0.0, 0.0);
      for(int j=0; j<degree; j++) {
        vc += vv[j]; tc += tt[j];
      }
      vc /= (double)degree; tc /= (double)degree;
      // sort vertices
      int idx[6];
      double pa[6];
      for(int i=0; i<degree; i++) {
        double vx = Dot(vv[i] - vc, right);
        double vy = Dot(vv[i] - vc, up);
        // compute pseudo-angle
        pa[i] = vy / (std::abs(vx) + std::abs(vy));
        if (vx < 0.0) pa[i] = 2.0 - pa[i];
        else if (vy < 0.0) pa[i] = 4.0 + pa[i];
        // init idx
        idx[i] = i;
      }
      Sort(pa, idx, degree);
      // output polygon
      for(int j=0; j<degree; j++) {
        vertex.add(vv[idx[j]].x());
        vertex.add(vv[idx[j]].y());
        vertex.add(vv[idx[j]].z());
        texcoord.add(tt[idx[j]].x());
        texcoord.add(tt[idx[j]].y());
        texcoord.add(tt[idx[j]].z());
      }
    } else if (degree == 3) {
      // output a single triangle
      for(int j=0; j<degree; j++) {
        vertex.add(vv[j].x());
        vertex.add(vv[j].y());
        vertex.add(vv[j].z());
        texcoord.add(tt[j].x());
        texcoord.add(tt[j].y());
        texcoord.add(tt[j].z());
      }
    }
    // else we don't care
    if(degree >= 3) {
      k += degree;
      size.add(degree);
    }
    // decrement ray parameter
    t -= dt;
  }
}

} // end namespace Volume
