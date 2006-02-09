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


//    File   : InsertField.h
//    Author : Michael Callahan
//    Date   : Jan 2006

#if !defined(InsertField_h)
#define InsertField_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Geometry/CompGeom.h>

namespace SCIRun {

class InsertFieldAlgo : public DynamicAlgoBase
{
public:
  virtual void execute_0(FieldHandle tet, FieldHandle insert) = 0;
  virtual void execute_1(FieldHandle tet, FieldHandle insert) = 0;
  virtual void execute_2(FieldHandle tet, FieldHandle insert) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftet,
					    const TypeDescription *finsert);
};


template <class TFIELD, class IFIELD>
class InsertFieldAlgoT : public InsertFieldAlgo
{
public:

  //! virtual interface. 
  virtual void execute_0(FieldHandle tet, FieldHandle insert);
  virtual void execute_1(FieldHandle tet, FieldHandle insert);
  virtual void execute_2(FieldHandle tet, FieldHandle insert);
};


template <class TFIELD, class IFIELD>
void
InsertFieldAlgoT<TFIELD, IFIELD>::execute_0(FieldHandle tet_h,
                                            FieldHandle insert_h)
{
  TFIELD *tfield = dynamic_cast<TFIELD *>(tet_h.get_rep());
  typename TFIELD::mesh_handle_type tmesh = tfield->get_typed_mesh();
  IFIELD *ifield = dynamic_cast<IFIELD *>(insert_h.get_rep());
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  tmesh->synchronize(Mesh::EDGES_E | Mesh::FACE_NEIGHBORS_E | Mesh::FACES_E);

  typename IFIELD::mesh_type::Node::iterator ibi, iei;
  imesh->begin(ibi);
  imesh->end(iei);

  while (ibi != iei)
  {
    Point p;
    imesh->get_center(p, *ibi);

    typename TFIELD::mesh_type::Elem::index_type elem;
    if (tmesh->locate(elem, p))
    {
      typename TFIELD::mesh_type::Node::index_type newnode;
      typename TFIELD::mesh_type::Elem::array_type newelems;
      tmesh->insert_node_in_cell_2(newelems, newnode, elem, p);
    }

    ++ibi;
  }

  tfield->resize_fdata();
}


template <class TFIELD, class IFIELD>
void
InsertFieldAlgoT<TFIELD, IFIELD>::execute_1(FieldHandle tet_h,
                                            FieldHandle insert_h)
{
  TFIELD *tfield = dynamic_cast<TFIELD *>(tet_h.get_rep());
  typename TFIELD::mesh_handle_type tmesh = tfield->get_typed_mesh();
  IFIELD *ifield = dynamic_cast<IFIELD *>(insert_h.get_rep());
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  imesh->synchronize(Mesh::EDGES_E);
  tmesh->synchronize(Mesh::EDGES_E | Mesh::FACE_NEIGHBORS_E | Mesh::FACES_E);

  typename IFIELD::mesh_type::Edge::iterator ibi, iei;
  imesh->begin(ibi);
  imesh->end(iei);

  while (ibi != iei)
  {
    typename IFIELD::mesh_type::Node::array_type enodes;
    imesh->get_nodes(enodes, *ibi);
    ++ibi;

    typename TFIELD::mesh_type::Elem::index_type elem, neighbor;
    typename TFIELD::mesh_type::Face::array_type faces;
    typename TFIELD::mesh_type::Node::array_type nodes;
    typename TFIELD::mesh_type::Face::index_type minface;

    Point e0, e1;
    imesh->get_center(e0, enodes[0]);
    imesh->get_center(e1, enodes[1]);


    // Find our first element.  If e0 isn't inside then try e1.  Need
    // to handle falling outside of mesh better.
    if (!tmesh->locate(elem, e0))
    {
      Point tmp = e0;
      e0 = e1;
      e1 = tmp;
      if (!tmesh->locate(elem, e0))
        continue;
    }

    Vector dir = e1 - e0;


    vector<Point> points;
    points.push_back(e0);

    unsigned int i;
    unsigned int maxsteps = 10000;
    for (i=0; i < maxsteps; i++)
    {
      tmesh->get_faces(faces, elem);
      double mindist = 1.0-1.0e-6;
      bool found = false;
      Point ecenter;
      tmesh->get_center(ecenter, elem);
      for (unsigned int j=0; j < faces.size(); j++)
      {
        Point p0, p1, p2;
        tmesh->get_nodes(nodes, faces[j]);
        tmesh->get_center(p0, nodes[0]);
        tmesh->get_center(p1, nodes[1]);
        tmesh->get_center(p2, nodes[2]);
        Vector normal = Cross(p1-p0, p2-p0);
        if (Dot(normal, ecenter-p0) > 0.0) { normal *= -1.0; }
        const double dist = RayPlaneIntersection(e0, dir, p0, normal);
        if (dist > -1.0e-6 && dist < mindist)
        {
          mindist = dist;
          minface = faces[j];
          found = true;
        }
      }
      if (!found) { break; }

      if (mindist > 1.0e-6) { points.push_back(e0 + dir * mindist); }

      // TODO:  Handle falling outside of mesh better.  May not be convex.
      if (!tmesh->get_neighbor(neighbor, elem, minface)) { break; }
      elem = neighbor;
    }
    points.push_back(e1);

    typename TFIELD::mesh_type::Node::index_type newnode;
    typename TFIELD::mesh_type::Elem::array_type newelems;

    for (i = 0; i < points.size(); i++)
    {
      if (tmesh->locate(elem, points[i]))
      {
        tmesh->insert_node_in_cell_2(newelems, newnode, elem, points[i]);
      }
    }
  }

  tfield->resize_fdata();
}


template <class TFIELD, class IFIELD>
void
InsertFieldAlgoT<TFIELD, IFIELD>::execute_2(FieldHandle tet_h,
                                            FieldHandle insert_h)
{
  TFIELD *tfield = dynamic_cast<TFIELD *>(tet_h.get_rep());
  typename TFIELD::mesh_handle_type tmesh = tfield->get_typed_mesh();
  IFIELD *ifield = dynamic_cast<IFIELD *>(insert_h.get_rep());
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  imesh->synchronize(Mesh::FACES_E);
  tmesh->synchronize(Mesh::EDGES_E | Mesh::FACE_NEIGHBORS_E | Mesh::FACES_E);

  typename IFIELD::mesh_type::Face::iterator ibi, iei;
  imesh->begin(ibi);
  imesh->end(iei);

  while (ibi != iei)
  {
    typename IFIELD::mesh_type::Node::array_type fnodes;
    imesh->get_nodes(fnodes, *ibi);

    unsigned int i;

    Point tri[4];
    for (i = 0; i < 4 && i < fnodes.size(); i++)
    {
      imesh->get_center(tri[i], fnodes[i]);
    }
    
    vector<Point> points;

    // TODO:
    // Intersect all of the edges in the tetvol with the triangle.
    // Add each intersection between (0,1) to the results.
    
    typename TFIELD::mesh_type::Edge::iterator edge_iter, edge_iter_end;
    tmesh->begin(edge_iter);
    tmesh->end(edge_iter_end);
    while (edge_iter != edge_iter_end)
    {
      typename TFIELD::mesh_type::Node::array_type nodes;
      tmesh->get_nodes(nodes, *edge_iter);

      Point e0, e1;
      tmesh->get_center(e0, nodes[0]);
      tmesh->get_center(e1, nodes[1]);
      Vector dir = e1 - e0;

      double t, u, v;
      const bool hit = RayTriangleIntersection(t, u, v, false, e0, dir,
                                               tri[0], tri[1], tri[2]);
      
      if (hit && t > 0 && t < 1.0)
      {
        points.push_back(e0 + t * dir);
      }

      ++edge_iter;
    }

    typename TFIELD::mesh_type::Elem::index_type elem;
    typename TFIELD::mesh_type::Node::index_type newnode;
    typename TFIELD::mesh_type::Elem::array_type newelems;
    for (i = 0; i < points.size(); i++)
    {
      tmesh->locate(elem, points[i]);
      tmesh->insert_node_in_cell_2(newelems, newnode, elem, points[i]);
    }

    ++ibi;
  }

  tfield->resize_fdata();
}



} // end namespace SCIRun

#endif // InsertField_h
