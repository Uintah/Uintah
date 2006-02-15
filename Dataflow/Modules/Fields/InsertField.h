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
#include <Core/Basis/Constant.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/SparseRowMatrix.h>


namespace SCIRun {

class InsertFieldAlgo : public DynamicAlgoBase
{
public:
  virtual void execute_0(FieldHandle tet, FieldHandle insert,
                         vector<unsigned int> &added_nodes,
                         vector<unsigned int> &added_elems) = 0;
  virtual void execute_1(FieldHandle tet, FieldHandle insert,
                         vector<unsigned int> &added_nodes,
                         vector<unsigned int> &added_elems) = 0;
  virtual void execute_2(FieldHandle tet, FieldHandle insert,
                         vector<unsigned int> &added_nodes,
                         vector<unsigned int> &added_elems) = 0;


  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftet,
					    const TypeDescription *finsert);
};


template <class TFIELD, class IFIELD>
class InsertFieldAlgoT : public InsertFieldAlgo
{
public:

  //! virtual interface. 
  virtual void execute_0(FieldHandle tet, FieldHandle insert,
                         vector<unsigned int> &added_nodes,
                         vector<unsigned int> &added_elems);
  virtual void execute_1(FieldHandle tet, FieldHandle insert,
                         vector<unsigned int> &added_nodes,
                         vector<unsigned int> &added_elems);
  virtual void execute_2(FieldHandle tet, FieldHandle insert,
                         vector<unsigned int> &added_nodes,
                         vector<unsigned int> &added_elems);
};


template <class TFIELD, class IFIELD>
void
InsertFieldAlgoT<TFIELD, IFIELD>::execute_0(FieldHandle tet_h,
                                            FieldHandle insert_h,
                                            vector<unsigned int> &added_nodes,
                                            vector<unsigned int> &added_elems)
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

      added_nodes.push_back(newnode);
      added_elems.push_back(elem);
      for (unsigned int i = 0; i < newelems.size(); i++)
      {
        added_elems.push_back(newelems[i]);
      }
    }

    ++ibi;
  }

  tfield->resize_fdata();
}


template <class TFIELD, class IFIELD>
void
InsertFieldAlgoT<TFIELD, IFIELD>::execute_1(FieldHandle tet_h,
                                            FieldHandle insert_h,
                                            vector<unsigned int> &added_nodes,
                                            vector<unsigned int> &added_elems)
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

        added_nodes.push_back(newnode);
        added_elems.push_back(elem);
        for (unsigned int i = 0; i < newelems.size(); i++)
        {
          added_elems.push_back(newelems[i]);
        }
      }
    }
  }

  tfield->resize_fdata();
}


template <class TFIELD, class IFIELD>
void
InsertFieldAlgoT<TFIELD, IFIELD>::execute_2(FieldHandle tet_h,
                                            FieldHandle insert_h,
                                            vector<unsigned int> &added_nodes,
                                            vector<unsigned int> &added_elems)
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

    vector<Point> tri;
    for (i = 0; i < fnodes.size(); i++)
    {
      Point p;
      imesh->get_center(p, fnodes[i]);
      tri.push_back(p);
    }
    
    // Test each triangle in the face (fan the polygon).
    for (i = 2; i < tri.size(); i++)
    {
      // Intersects all of the edges in the tetvol with the triangle.
      // Add each intersection between (0,1) to the results.

      // TODO: We only need to test the edges that are 'close', not all
      // of them.  Augment the locate grid and use that to speed this up.

      vector<Point> points;
    
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
                                                 tri[0], tri[i-1], tri[i]);
      
        if (hit && t > 0 && t < 1.0)
        {
          points.push_back(e0 + t * dir);
        }

        ++edge_iter;
      }

      typename TFIELD::mesh_type::Elem::index_type elem;
      typename TFIELD::mesh_type::Node::index_type newnode;
      typename TFIELD::mesh_type::Elem::array_type newelems;
      for (unsigned int j = 0; j < points.size(); j++)
      {
        if (tmesh->locate(elem, points[j]))
        {
          tmesh->insert_node_in_cell_2(newelems, newnode, elem, points[j]);

          added_nodes.push_back(newnode);
          added_elems.push_back(elem);
          for (unsigned int k = 0; k < newelems.size(); k++)
          {
            added_elems.push_back(newelems[k]);
          }
        }
      }
    }
    ++ibi;
  }

  tfield->resize_fdata();
}



class InsertFieldExtract : public DynamicAlgoBase
{
public:

  virtual void extract(FieldHandle &result_field,
                       MatrixHandle &result_mapping,
                       FieldHandle tet_h,
                       vector<unsigned int> &added_nodes,
                       vector<unsigned int> &added_elems) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftet,
					    int dim);
};


template <class TFIELD, class IFIELD>
class InsertFieldExtractT : public InsertFieldExtract
{
public:

  virtual void extract(FieldHandle &result_field,
                       MatrixHandle &result_mapping,
                       FieldHandle tet_h,
                       vector<unsigned int> &added_nodes,
                       vector<unsigned int> &added_elems);
};


template <class TFIELD, class IFIELD>
void
InsertFieldExtractT<TFIELD, IFIELD>::extract(FieldHandle &result_field,
                                             MatrixHandle &result_mapping,
                                             FieldHandle tet_h,
                                             vector<unsigned int> &added_nodes,
                                             vector<unsigned int> &added_elems)
{
  TFIELD *tfield = dynamic_cast<TFIELD *>(tet_h.get_rep());
  typename TFIELD::mesh_handle_type tmesh = tfield->get_typed_mesh();

  typename IFIELD::mesh_handle_type omesh =
    scinew typename IFIELD::mesh_type();

  std::sort(added_nodes.begin(), added_nodes.end());
  vector<unsigned int>::iterator nodes_end, itr;
  nodes_end = std::unique(added_nodes.begin(), added_nodes.end());
  for (itr = added_nodes.begin(); itr != nodes_end; ++itr)
  {
    Point p;
    tmesh->get_point(p, *itr);
    omesh->add_point(p);
  }

  if (omesh->dimensionality() > 0)
  {
    std::sort(added_elems.begin(), added_elems.end());
    vector<unsigned int>::iterator elems_end;
    elems_end = std::unique(added_elems.begin(), added_elems.end());
    for (itr = added_elems.begin(); itr != elems_end; ++itr)
    {
      if (omesh->dimensionality() == 1)
      {
        typename TFIELD::mesh_type::Edge::array_type edges;
        tmesh->get_edges(edges,
                         typename TFIELD::mesh_type::Cell::index_type(*itr));
        for (unsigned int i = 0; i < edges.size(); i++)
        {
          typename TFIELD::mesh_type::Node::array_type oldnodes;
          typename IFIELD::mesh_type::Node::array_type newnodes;
          tmesh->get_nodes(oldnodes, edges[i]);
          bool all_found = true;
          for (unsigned int j = 0; j < oldnodes.size(); j++)
          {
            vector<unsigned int>::iterator loc =
              lower_bound(added_nodes.begin(), nodes_end, oldnodes[j]);
            if (loc != nodes_end && *loc == oldnodes[j])
            {
              newnodes.push_back(loc - added_nodes.begin());
            }
            else
            {
              all_found = false;
              break;
            }
          }

          // TODO:  Only add unique elements.
          if (all_found)
          {
            omesh->add_elem(newnodes);
          }
        }
      }
      if (omesh->dimensionality() == 2)
      {
        typename TFIELD::mesh_type::Face::array_type faces;
        tmesh->get_faces(faces,
                         typename TFIELD::mesh_type::Cell::index_type(*itr));
        for (unsigned int i = 0; i < faces.size(); i++)
        {
          typename TFIELD::mesh_type::Node::array_type oldnodes;
          typename IFIELD::mesh_type::Node::array_type newnodes;
          tmesh->get_nodes(oldnodes, faces[i]);
          bool all_found = true;
          for (unsigned int j = 0; j < oldnodes.size(); j++)
          {
            vector<unsigned int>::iterator loc =
              std::lower_bound(added_nodes.begin(), nodes_end, oldnodes[j]);
            if (loc != nodes_end && *loc == oldnodes[j])
            {
              newnodes.push_back(loc - added_nodes.begin());
            }
            else
            {
              all_found = false;
              break;
            }
          }

          // TODO:  Only add unique elements.
          if (all_found)
          {
            omesh->add_elem(newnodes);
          }
        }
      }
      if (omesh->dimensionality() == 3)
      {
        typename TFIELD::mesh_type::Node::array_type oldnodes;
        typename IFIELD::mesh_type::Node::array_type newnodes;
        tmesh->get_nodes(oldnodes,
                         typename TFIELD::mesh_type::Cell::index_type(*itr));
        bool all_found = true;
        for (unsigned int j = 0; j < oldnodes.size(); j++)
        {
          vector<unsigned int>::iterator loc =
            std::lower_bound(added_nodes.begin(), nodes_end, oldnodes[j]);
          if (loc != nodes_end && *loc == oldnodes[j])
          {
            newnodes.push_back(loc - added_nodes.begin());
          }
          else
          {
            all_found = false;
            break;
          }
        }

        if (all_found)
        {
          omesh->add_elem(newnodes);
        }
      }
    }
  }

  // Create the output field.
  result_field = scinew IFIELD(omesh);

  // Create the output mapping.
  typename TFIELD::mesh_type::Node::size_type tnodesize;
  tmesh->size(tnodesize);
  typename IFIELD::mesh_type::Node::size_type onodesize;
  omesh->size(onodesize);

  const int nrows = onodesize;
  const int ncols = tnodesize;
  int *rr = scinew int[nrows+1];
  int *cc = scinew int[nrows];
  double *d = scinew double[nrows];

  int i = 0;
  for (itr = added_nodes.begin(); itr != nodes_end; itr++)
  {
    cc[i] = *itr;
    i++;
  }

  int j;
  for (j = 0; j < nrows; j++)
  {
    rr[j] = j;
    d[j] = 1.0;
  }
  rr[j] = j;
  
  result_mapping = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);
}


} // end namespace SCIRun

#endif // InsertField_h
