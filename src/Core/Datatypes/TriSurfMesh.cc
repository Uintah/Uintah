/*
 *  TriSurfMesh.cc: Tetrahedral mesh with new design.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#include <Core/Datatypes/TriSurfMesh.h>


namespace SCIRun {

PersistentTypeID TriSurfMesh::type_id("TriSurfMesh", "MeshBase", NULL);


const string
TriSurfMesh::type_name(int)
{
  return "TriSurfMesh";
}



TriSurfMesh::TriSurfMesh()
{
}

TriSurfMesh::TriSurfMesh(const TriSurfMesh &copy)
  : points_(copy.points_),
    faces_(copy.faces_),
    neighbors_(copy.neighbors_)
{
}

TriSurfMesh::~TriSurfMesh()
{
}


BBox
TriSurfMesh::get_bounding_box() const
{
  BBox result;

  for (int i = 0; i < points_.size(); i++)
  {
    result.extend(points_[i]);
  }

  return result;
}


TriSurfMesh::node_iterator
TriSurfMesh::node_begin() const
{
  return 0;
}

TriSurfMesh::node_iterator
TriSurfMesh::node_end() const
{
  return points_.size();
}

TriSurfMesh::edge_iterator
TriSurfMesh::edge_begin() const
{
  return 0;
}

TriSurfMesh::edge_iterator
TriSurfMesh::edge_end() const
{
  return faces_.size();
}

TriSurfMesh::face_iterator
TriSurfMesh::face_begin() const
{
  return 0;
}

TriSurfMesh::face_iterator
TriSurfMesh::face_end() const
{
  return faces_.size() / 3;
}

TriSurfMesh::cell_iterator
TriSurfMesh::cell_begin() const
{
  return 0;
}

TriSurfMesh::cell_iterator
TriSurfMesh::cell_end() const
{
  return 0;
}


void
TriSurfMesh::get_nodes(node_array &array, edge_index idx) const
{
  static int table[6][2] =
  {
    {0, 1},
    {1, 2},
    {2, 0},
  };

  const int off = idx % 3;
  const int node = idx - off;

  array.push_back(faces_[node + table[off][0]]);
  array.push_back(faces_[node + table[off][1]]);
}


void
TriSurfMesh::get_nodes(node_array &array, face_index idx) const
{
  array.push_back(faces_[idx * 3 + 0]);
  array.push_back(faces_[idx * 3 + 1]);
  array.push_back(faces_[idx * 3 + 2]);
}


void
TriSurfMesh::get_edges(edge_array &array, face_index idx) const
{
  array.push_back(idx * 3 + 0);
  array.push_back(idx * 3 + 1);
  array.push_back(idx * 3 + 2);
}


void
TriSurfMesh::get_neighbor(face_index &neighbor, edge_index idx) const
{
  neighbor = neighbors_[idx];
}


static double
distance2(const Point &p0, const Point &p1)
{
  const double dx = p0.x() - p1.x();
  const double dy = p0.y() - p1.y();
  const double dz = p0.z() - p1.z();
  return dx * dx + dy * dy + dz * dz;
}


void
TriSurfMesh::locate(node_iterator &loc, const Point &p)
{
  node_iterator ni = node_begin();
  loc = ni;
  if (ni == node_end())
  {
    return;
  }

  double min_dist = distance2(p, points_[*ni]);
  loc = ni;
  ++ni;

  while (ni != node_end())
  {
    const double dist = distance2(p, points_[*ni]);
    if (dist < min_dist)
    {
      loc = ni;
    }
    ++ni;
  }
}


void
TriSurfMesh::locate(edge_iterator &loc, const Point &)
{
  loc = edge_end();
}


void
TriSurfMesh::locate(face_iterator &loc, const Point &)
{
  loc = face_end();
}


void
TriSurfMesh::locate(cell_iterator &loc, const Point &)
{
  loc = cell_end();
}



void
TriSurfMesh::unlocate(Point &result, const Point &p)
{
  result = p;
}


void
TriSurfMesh::get_point(Point &result, node_index index) const
{
  result = points_[index];
}


bool
TriSurfMesh::inside4_p(int i, const Point &p)
{
  // TODO: This has not been tested.
  // TODO: Looks like too much code to check sign of 4 plane/point tests.

  const Point &p0 = points_[faces_[i+0]];
  const Point &p1 = points_[faces_[i+1]];
  const Point &p2 = points_[faces_[i+2]];
  const Point &p3 = points_[faces_[i+3]];
  const double x0 = p0.x();
  const double y0 = p0.y();
  const double z0 = p0.z();
  const double x1 = p1.x();
  const double y1 = p1.y();
  const double z1 = p1.z();
  const double x2 = p2.x();
  const double y2 = p2.y();
  const double z2 = p2.z();
  const double x3 = p3.x();
  const double y3 = p3.y();
  const double z3 = p3.z();

  const double a0 = + x1*(y2*z3-y3*z2) + x2*(y3*z1-y1*z3) + x3*(y1*z2-y2*z1);
  const double a1 = - x2*(y3*z0-y0*z3) - x3*(y0*z2-y2*z0) - x0*(y2*z3-y3*z2);
  const double a2 = + x3*(y0*z1-y1*z0) + x0*(y1*z3-y3*z1) + x1*(y3*z0-y0*z3);
  const double a3 = - x0*(y1*z2-y2*z1) - x1*(y2*z0-y0*z2) - x2*(y0*z1-y1*z0);
  const double iV6 = 1.0 / (a0+a1+a2+a3);

  const double b0 = - (y2*z3-y3*z2) - (y3*z1-y1*z3) - (y1*z2-y2*z1);
  const double c0 = + (x2*z3-x3*z2) + (x3*z1-x1*z3) + (x1*z2-x2*z1);
  const double d0 = - (x2*y3-x3*y2) - (x3*y1-x1*y3) - (x1*y2-x2*y1);
  const double s0 = iV6 * (a0 + b0*p.x() + c0*p.y() + d0*p.z());
  if (s0 < -1.e-6)
    return false;

  const double b1 = + (y3*z0-y0*z3) + (y0*z2-y2*z0) + (y2*z3-y3*z2);
  const double c1 = - (x3*z0-x0*z3) - (x0*z2-x2*z0) - (x2*z3-x3*z2);
  const double d1 = + (x3*y0-x0*y3) + (x0*y2-x2*y0) + (x2*y3-x3*y2);
  const double s1 = iV6 * (a1 + b1*p.x() + c1*p.y() + d1*p.z());
  if (s1 < -1.e-6)
    return false;

  const double b2 = - (y0*z1-y1*z0) - (y1*z3-y3*z1) - (y3*z0-y0*z3);
  const double c2 = + (x0*z1-x1*z0) + (x1*z3-x3*z1) + (x3*z0-x0*z3);
  const double d2 = - (x0*y1-x1*y0) - (x1*y3-x3*y1) - (x3*y0-x0*y3);
  const double s2 = iV6 * (a2 + b2*p.x() + c2*p.y() + d2*p.z());
  if (s2 < -1.e-6)
    return false;

  const double b3 = +(y1*z2-y2*z1) + (y2*z0-y0*z2) + (y0*z1-y1*z0);
  const double c3 = -(x1*z2-x2*z1) - (x2*z0-x0*z2) - (x0*z1-x1*z0);
  const double d3 = +(x1*y2-x2*y1) + (x2*y0-x0*y2) + (x0*y1-x1*y0);
  const double s3 = iV6 * (a3 + b3*p.x() + c3*p.y() + d3*p.z());
  if (s3 < -1.e-6)
    return false;

  return true;
}



TriSurfMesh::node_index
TriSurfMesh::add_find_point(const Point &p, double err)
{
  node_iterator i;
  locate(i, p);
  if (i != node_end() || distance2(points_[*i], p) < err)
  {
    return *i;
  }
  else
  {
    points_.add(p);
    return points_.size() - 1;
  }
}


void
TriSurfMesh::add_triangle(node_index a, node_index b, node_index c,
		      bool cw_p)
{
  if (cw_p)
  {
    faces_.add(a);
    faces_.add(b);
    faces_.add(c);
  }
  else
  {
    faces_.add(c);
    faces_.add(b);
    faces_.add(a);
  }
}


void
TriSurfMesh::add_triangle(const Point &p0, const Point &p1, const Point &p2,
		      bool cw_p)
{
  add_triangle(add_find_point(p0),
	       add_find_point(p1),
	       add_find_point(p2), cw_p);
}


#define TRISURFMESH_VERSION 1

void
TriSurfMesh::io(Piostream &stream)
{
  stream.begin_class(type_id.type.c_str(), TRISURFMESH_VERSION);

  Pio(stream, points_);
  Pio(stream, faces_);
  Pio(stream, neighbors_);

  stream.end_class();
}



} // namespace SCIRun
