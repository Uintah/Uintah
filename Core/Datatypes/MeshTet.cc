/*
 *  MeshTet.cc: Tetrahedral mesh with new design.
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

#include <Core/Datatypes/MeshTet.h>


namespace SCIRun {

PersistentTypeID MeshTet::type_id("MeshTet", "Datatype", NULL);


MeshTet::MeshTet()
{
}

MeshTet::MeshTet(const MeshTet &copy)
  : points_(copy.points_),
    tets_(copy.tets_),
    neighbors_(copy.neighbors_)
{
}

MeshTet::~MeshTet()
{
}


BBox
MeshTet::get_bounding_box() const
{
  BBox result;

  for (int i = 0; i < points_.size(); i++)
  {
    result.extend(points_[i]);
  }

  return result;
}


MeshTet::node_iterator
MeshTet::node_begin() const
{
  return 0;
}

MeshTet::node_iterator
MeshTet::node_end() const
{
  return points_.size();
}

MeshTet::edge_iterator
MeshTet::edge_begin() const
{
  return NULL;
}

MeshTet::edge_iterator
MeshTet::edge_end() const
{
  return NULL;
}

MeshTet::face_iterator
MeshTet::face_begin() const
{
  return NULL;
}

MeshTet::face_iterator
MeshTet::face_end() const
{
  return NULL;
}

MeshTet::cell_iterator
MeshTet::cell_begin() const
{
  return 0;
}

MeshTet::cell_iterator
MeshTet::cell_end() const
{
  return tets_.size() >> 2;
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
MeshTet::locate_node(node_index &node, const Point &p)
{
  // TODO: Use search structure instead of exaustive search.
  int min_indx;
  double min_dist;

  if (points_.size() == 0)
  {
    min_indx = node_end();
  }
  else
  {
    min_indx = 0;
    min_dist = distance2(p, points_[0]);
  }

  for (int i = 1; i < points_.size(); i++)
  {
    const double dist = distance2(p, points_[i]);
    if (dist < min_dist)
    {
      min_dist = dist;
      min_indx = i;
    }
  }
  node = min_indx;
}


void
MeshTet::locate_edge(edge_index &edge, const Point & /* p */)
{
  edge = edge_end();
}

void
MeshTet::locate_face(face_index &face, const Point & /* p */)
{
  face = face_end();
}


void
MeshTet::locate_cell(cell_index &cell, const Point &p)
{
  for (int i=0; i < tets_.size(); i+=4)
  {
    if (inside4_p(i, p))
    {
      cell = i >> 2;
      return;
    }
  }
  cell = cell_end();
}



void
MeshTet::unlocate(Point &result, const Point &p)
{
  result = p;
}


void
MeshTet::get_point(Point &result, node_index index) const
{
  result = points_[index];
}


bool
MeshTet::inside4_p(int i, const Point &p)
{
  // TODO: This has not been tested.
  // TODO: Looks like too much code to check sign of 4 plane/point tests.

  const Point &p0 = points_[tets_[i+0]];
  const Point &p1 = points_[tets_[i+1]];
  const Point &p2 = points_[tets_[i+2]];
  const Point &p3 = points_[tets_[i+3]];
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


#define MESHTET_VERSION 1

void
MeshTet::io(Piostream &stream)
{
  stream.begin_class(type_id.type.c_str(), MESHTET_VERSION);

  Pio(stream, points_);
  Pio(stream, tets_);
  Pio(stream, neighbors_);

  stream.end_class();
}



} // namespace SCIRun
