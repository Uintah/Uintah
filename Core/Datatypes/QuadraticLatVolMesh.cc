//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : QuadraticLatVolMesh.cc
//    Author : Martin Cole
//    Date   : Sun Feb 24 14:38:20 2002


#include <Core/Datatypes/QuadraticLatVolMesh.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
#include <algorithm>

namespace SCIRun {


Persistent* make_QuadraticLatVolMesh() {
  return scinew QuadraticLatVolMesh;
}

PersistentTypeID QuadraticLatVolMesh::type_id("QuadraticLatVolMesh", 
					      "LatVolMesh",
					      make_QuadraticLatVolMesh);

const string
QuadraticLatVolMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name("QuadraticLatVolMesh");
  return name;
}

const TypeDescription*
QuadraticLatVolMesh::get_type_description() const
{
  return SCIRun::get_type_description((QuadraticLatVolMesh *)0);
}

const TypeDescription*
get_type_description(QuadraticLatVolMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("QuadraticLatVolMesh",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}


const TypeDescription*
get_type_description(QuadraticLatVolMesh::Node *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("QuadraticLatVolMesh::Node",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}


#define QUADRATICTETVOLMESH_VERSION 1

void
QuadraticLatVolMesh::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), QUADRATICTETVOLMESH_VERSION);
  LatVolMesh::io(stream);
  stream.end_class();
}


QuadraticLatVolMesh::QuadraticLatVolMesh() :
  LatVolMesh()
{
}

QuadraticLatVolMesh::QuadraticLatVolMesh(const QuadraticLatVolMesh& copy) :
  LatVolMesh(copy)
{
}

QuadraticLatVolMesh::QuadraticLatVolMesh(const LatVolMesh &tv) :
  LatVolMesh(tv)
{
}

QuadraticLatVolMesh::~QuadraticLatVolMesh()
{
}

void
QuadraticLatVolMesh::begin(Node::iterator &itr) const
{
  LatVolMesh::Node::iterator nb;
  LatVolMesh::Node::iterator ne;
  LatVolMesh::Edge::iterator eb;

  LatVolMesh::begin(nb);
  LatVolMesh::end(ne);
  LatVolMesh::begin(eb);
  
  itr = Node::iterator(nb, ne, eb);
}

void
QuadraticLatVolMesh::end(Node::iterator &itr) const
{
  LatVolMesh::Node::iterator nb;
  LatVolMesh::Node::iterator ne;
  LatVolMesh::Edge::iterator eb;

  LatVolMesh::end(nb);
  LatVolMesh::end(ne);
  LatVolMesh::end(eb);
  itr = Node::iterator(nb, ne, eb);
}

void
QuadraticLatVolMesh::size(Node::size_type &s) const
{
  LatVolMesh::Node::size_type ns;
  LatVolMesh::Edge::size_type es;
  s = Node::size_type(ns, es);
}

void
QuadraticLatVolMesh::begin(LatVolMesh::Edge::iterator &itr) const
{
  LatVolMesh::begin(itr);
}

void
QuadraticLatVolMesh::end(LatVolMesh::Edge::iterator &itr) const
{
  LatVolMesh::end(itr);
}

void
QuadraticLatVolMesh::size(LatVolMesh::Edge::size_type &s) const
{
  LatVolMesh::size(s);
}

void
QuadraticLatVolMesh::begin(LatVolMesh::Face::iterator &itr) const
{
  LatVolMesh::begin(itr);
}

void
QuadraticLatVolMesh::end(LatVolMesh::Face::iterator &itr) const
{
  LatVolMesh::end(itr);
}

void
QuadraticLatVolMesh::size(LatVolMesh::Face::size_type &s) const
{
  LatVolMesh::size(s);
}

void
QuadraticLatVolMesh::begin(LatVolMesh::Cell::iterator &itr) const
{
  LatVolMesh::begin(itr);
}

void
QuadraticLatVolMesh::end(LatVolMesh::Cell::iterator &itr) const
{
  LatVolMesh::end(itr);
}

void
QuadraticLatVolMesh::size(LatVolMesh::Cell::size_type &s) const
{
  LatVolMesh::size(s);
}

void 
QuadraticLatVolMesh::get_nodes(Node::array_type &array, 
			       Edge::index_type idx) const
{
  array.clear();

  // Push back node indices.
  LatVolMesh::Node::array_type nodes;
  LatVolMesh::get_nodes(nodes, idx);
  array.push_back(Node::index_type(nodes[0]));
  array.push_back(Node::index_type(nodes[1]));

  // Push back edge index;
  LatVolMesh::Node::iterator nend;
  LatVolMesh::end(nend);
  array.push_back(Node::index_type(*nend, idx));
}


void 
QuadraticLatVolMesh::get_nodes(Node::array_type &array, 
			       Face::index_type idx) const
{
  unsigned int i;

  array.clear();
  
  // Push back node indices
  LatVolMesh::Node::array_type nodes;
  LatVolMesh::get_nodes(nodes, idx);
  for (i = 0; i < nodes.size(); i++)
  {
    array.push_back(Node::index_type(nodes[i]));
  }

  // Push back edge indices.
  LatVolMesh::Node::iterator nend;
  LatVolMesh::end(nend);

  LatVolMesh::Edge::array_type edges;
  LatVolMesh::get_edges(edges, idx);
  for (i = 0; i < edges.size(); i++)
  {
    array.push_back(Node::index_type(*nend, edges[i]));
  }
}


void
QuadraticLatVolMesh::get_nodes(Node::array_type &array, 
			       const Cell::index_type &idx) const
{
  unsigned int i;

  array.clear();
  
  // Push back node indices
  LatVolMesh::Node::array_type nodes;
  LatVolMesh::get_nodes(nodes, idx);
  for (i = 0; i < nodes.size(); i++)
  {
    array.push_back(Node::index_type(nodes[i]));
  }

  // Push back edge indices.
  LatVolMesh::Node::iterator nend;
  LatVolMesh::end(nend);

  LatVolMesh::Edge::array_type edges;
  LatVolMesh::get_edges(edges, idx);
  for (i = 0; i < edges.size(); i++)
  {
    array.push_back(Node::index_type(*nend, edges[i]));
  }
}


void 
QuadraticLatVolMesh::get_center(Point &result,
				const Node::index_type &index) const
{ 
  if (index.which_ == 0)
  {
    LatVolMesh::get_center(result, index.node_index_);
  }
  else
  {
    LatVolMesh::get_center(result, index.edge_index_);
  }
}


} // end namespace SCIRun
