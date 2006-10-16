/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
#include <set>

namespace SCIRun {

class InsertFieldAlgo : public DynamicAlgoBase
{
public:
  virtual void execute_0(FieldHandle tet, FieldHandle insert,
                         vector<unsigned int> &new_nodes,
                         vector<unsigned int> &new_elems) = 0;
  virtual void execute_1(FieldHandle tet, FieldHandle insert,
                         vector<unsigned int> &new_nodes,
                         vector<unsigned int> &new_elems) = 0;
  virtual void execute_2(FieldHandle tet, FieldHandle insert,
                         vector<unsigned int> &new_nodes,
                         vector<unsigned int> &new_elems) = 0;


  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *ftet,
					    const TypeDescription *finsert,
                                            bool tri);
};


template <class TFIELD, class IFIELD>
class InsertFieldAlgoTet : public InsertFieldAlgo
{
public:

  //! virtual interface. 
  virtual void execute_0(FieldHandle tet, FieldHandle insert,
                         vector<unsigned int> &new_nodes,
                         vector<unsigned int> &new_elems);
  virtual void execute_1(FieldHandle tet, FieldHandle insert,
                         vector<unsigned int> &new_nodes,
                         vector<unsigned int> &new_elems);
  virtual void execute_2(FieldHandle tet, FieldHandle insert,
                         vector<unsigned int> &new_nodes,
                         vector<unsigned int> &new_elems);
};


template <class TFIELD, class IFIELD>
void
InsertFieldAlgoTet<TFIELD, IFIELD>::execute_0(FieldHandle tet_h,
                                              FieldHandle insert_h,
                                              vector<unsigned int> &new_nodes,
                                              vector<unsigned int> &new_elems)
{
  TFIELD *tfield = dynamic_cast<TFIELD *>(tet_h.get_rep());
  typename TFIELD::mesh_handle_type tmesh = tfield->get_typed_mesh();
  IFIELD *ifield = dynamic_cast<IFIELD *>(insert_h.get_rep());
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  tmesh->synchronize(Mesh::EDGES_E | Mesh::FACE_NEIGHBORS_E
                     | Mesh::FACES_E | Mesh::LOCATE_E);

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
      tmesh->insert_node_in_elem(newelems, newnode, elem, p);

      new_nodes.push_back(newnode);
      for (unsigned int i = 0; i < newelems.size(); i++)
      {
        new_elems.push_back(newelems[i]);
      }
    }

    ++ibi;
  }

  tmesh->unsynchronize();
  tfield->resize_fdata();
}


template <class TFIELD, class IFIELD>
void
InsertFieldAlgoTet<TFIELD, IFIELD>::execute_1(FieldHandle tet_h,
                                              FieldHandle insert_h,
                                              vector<unsigned int> &new_nodes,
                                              vector<unsigned int> &new_elems)
{
  TFIELD *tfield = dynamic_cast<TFIELD *>(tet_h.get_rep());
  typename TFIELD::mesh_handle_type tmesh = tfield->get_typed_mesh();
  IFIELD *ifield = dynamic_cast<IFIELD *>(insert_h.get_rep());
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  imesh->synchronize(Mesh::EDGES_E);
  tmesh->synchronize(Mesh::EDGES_E | Mesh::FACE_NEIGHBORS_E
                     | Mesh::FACES_E | Mesh::LOCATE_E);

  typename IFIELD::mesh_type::Edge::iterator ibi, iei;
  imesh->begin(ibi);
  imesh->end(iei);

  vector<Point> points;

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
  }

  typename TFIELD::mesh_type::Elem::index_type elem;
  typename TFIELD::mesh_type::Node::index_type newnode;
  typename TFIELD::mesh_type::Elem::array_type newelems;
  for (unsigned int i = 0; i < points.size(); i++)
  {
    if (tmesh->locate(elem, points[i]))
    {
      tmesh->insert_node_in_elem(newelems, newnode, elem, points[i]);

      new_nodes.push_back(newnode);
      for (unsigned int i = 0; i < newelems.size(); i++)
      {
        new_elems.push_back(newelems[i]);
      }
    }
  }

  tmesh->unsynchronize();
  tfield->resize_fdata();
}


template <class TFIELD, class IFIELD>
void
InsertFieldAlgoTet<TFIELD, IFIELD>::execute_2(FieldHandle tet_h,
                                              FieldHandle insert_h,
                                              vector<unsigned int> &new_nodes,
                                              vector<unsigned int> &new_elems)
{
  TFIELD *tfield = dynamic_cast<TFIELD *>(tet_h.get_rep());
  typename TFIELD::mesh_handle_type tmesh = tfield->get_typed_mesh();
  IFIELD *ifield = dynamic_cast<IFIELD *>(insert_h.get_rep());
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  imesh->synchronize(Mesh::FACES_E);
  tmesh->synchronize(Mesh::EDGES_E | Mesh::FACE_NEIGHBORS_E
                     | Mesh::FACES_E | Mesh::LOCATE_E);

  typename IFIELD::mesh_type::Face::iterator ibi, iei;
  imesh->begin(ibi);
  imesh->end(iei);

  vector<Point> points;
    
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
    }
    ++ibi;
  }

  typename TFIELD::mesh_type::Elem::index_type elem;
  typename TFIELD::mesh_type::Node::index_type newnode;
  typename TFIELD::mesh_type::Elem::array_type newelems;
  for (unsigned int j = 0; j < points.size(); j++)
  {
    if (tmesh->locate(elem, points[j]))
    {
      tmesh->insert_node_in_elem(newelems, newnode, elem, points[j]);
      
      new_nodes.push_back(newnode);
      for (unsigned int k = 0; k < newelems.size(); k++)
      {
        new_elems.push_back(newelems[k]);
      }
    }
  }

  tmesh->unsynchronize();
  tfield->resize_fdata();
}



template <class TFIELD, class IFIELD>
class InsertFieldAlgoTri : public InsertFieldAlgo
{
public:

  //! virtual interface. 
  virtual void execute_0(FieldHandle tet, FieldHandle insert,
                         vector<unsigned int> &new_nodes,
                         vector<unsigned int> &new_elems);
  virtual void execute_1(FieldHandle tet, FieldHandle insert,
                         vector<unsigned int> &new_nodes,
                         vector<unsigned int> &new_elems);
  virtual void execute_2(FieldHandle tet, FieldHandle insert,
                         vector<unsigned int> &new_nodes,
                         vector<unsigned int> &new_elems) {}
};


template <class TFIELD, class IFIELD>
void
InsertFieldAlgoTri<TFIELD, IFIELD>::execute_0(FieldHandle tet_h,
                                              FieldHandle insert_h,
                                              vector<unsigned int> &new_nodes,
                                              vector<unsigned int> &new_elems)
{
  TFIELD *tfield = dynamic_cast<TFIELD *>(tet_h.get_rep());
  typename TFIELD::mesh_handle_type tmesh = tfield->get_typed_mesh();
  IFIELD *ifield = dynamic_cast<IFIELD *>(insert_h.get_rep());
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  tmesh->synchronize(Mesh::EDGES_E | Mesh::EDGE_NEIGHBORS_E
                     | Mesh::FACES_E | Mesh::LOCATE_E);

  typename IFIELD::mesh_type::Node::iterator ibi, iei;
  imesh->begin(ibi);
  imesh->end(iei);

  while (ibi != iei)
  {
    Point p;
    imesh->get_center(p, *ibi);

    Point cp;
    typename TFIELD::mesh_type::Elem::index_type cf;
    tmesh->find_closest_elem(cp, cf, p);

    typename TFIELD::mesh_type::Node::index_type newnode;
    typename TFIELD::mesh_type::Elem::array_type newelems;
    tmesh->insert_node_in_face(newelems, newnode, cf, cp);

    new_nodes.push_back(newnode);
    for (unsigned int i = 0; i < newelems.size(); i++)
    {
      new_elems.push_back(newelems[i]);
    }

    ++ibi;
  }

  tmesh->synchronize(Mesh::EDGES_E);

  tfield->resize_fdata();
}


#define EPSILON 1.e-12

template <class TFIELD, class IFIELD>
void
InsertFieldAlgoTri<TFIELD, IFIELD>::execute_1(FieldHandle tet_h,
                                              FieldHandle insert_h,
                                              vector<unsigned int> &new_nodes,
                                              vector<unsigned int> &new_elems)
{
  TFIELD *tfield = dynamic_cast<TFIELD *>(tet_h.get_rep());
  typename TFIELD::mesh_handle_type tmesh = tfield->get_typed_mesh();
  IFIELD *ifield = dynamic_cast<IFIELD *>(insert_h.get_rep());
  typename IFIELD::mesh_handle_type imesh = ifield->get_typed_mesh();

  tmesh->synchronize(Mesh::EDGES_E | Mesh::EDGE_NEIGHBORS_E
                     | Mesh::FACES_E | Mesh::LOCATE_E);
  imesh->synchronize(Mesh::EDGES_E);

  typename IFIELD::mesh_type::Edge::iterator ibi, iei;
  imesh->begin(ibi);
  imesh->end(iei);

  while (ibi != iei)
  {
    typename IFIELD::mesh_type::Node::array_type nodes;
    
    imesh->get_nodes(nodes, *ibi);
    Point p[2];
    imesh->get_center(p[0], nodes[0]);
    imesh->get_center(p[1], nodes[1]);

    vector<Point> insertpoints;

    Point closest[2];
    typename TFIELD::mesh_type::Elem::index_type elem;
    typename TFIELD::mesh_type::Elem::index_type elem_end;
    tmesh->find_closest_elem(closest[0], elem, p[1]);
    insertpoints.push_back(closest[0]);
    tmesh->find_closest_elem(closest[1], elem_end, p[0]);

    // TODO: Find closest could and will land on degeneracies meaning
    // that our choice of elements there is arbitrary.  Need to walk
    // along near surface elements (breadth first search) instead of
    // exact elements.
    int previous_edge = -1;
    while (elem != elem_end)
    {
      typename TFIELD::mesh_type::Node::array_type tnodes;
      tmesh->get_nodes(tnodes, elem);
    
      Point tpoints[3];
    
      for (unsigned int k = 0; k < 3; k++)
      {
        tmesh->get_center(tpoints[k], tnodes[k]);
      }

      bool found = false;
      for (int k = 0; k < 3; k++)
      {
        if (k != previous_edge)
        {
          Point tp[2];
          tp[0] = tpoints[k];
          tp[1] = tpoints[(k+1)%3];

          double s, t;
          closest_line_to_line(s, t, closest[0], closest[1], tp[0], tp[1]);
          
          if (s > EPSILON && s < 1.0 - EPSILON &&
              t > EPSILON && t < 1.0 - EPSILON)
          {
            found = true;
            insertpoints.push_back(tp[0] + t * (tp[1] - tp[0]));
            unsigned int nbrhalf;
            tmesh->get_neighbor(nbrhalf, elem*3+k);
            elem = nbrhalf / 3;
            previous_edge = nbrhalf % 3;
            break;
          }
        }
      }
      if (!found)
      {
        cout << "EDGE WALKER DEAD END! " << elem << " " << *ibi << "\n";
        break;
      }
    }

    insertpoints.push_back(closest[1]);

    typename TFIELD::mesh_type::Node::index_type newnode;
    typename TFIELD::mesh_type::Elem::array_type newelems;
    for (unsigned int i = 0; i < insertpoints.size(); i++)
    {
      Point closest;
      typename TFIELD::mesh_type::Elem::index_type elem;
      tmesh->find_closest_elem(closest, elem, insertpoints[i]);
      tmesh->insert_node_in_face(newelems, newnode, elem, closest);
      new_nodes.push_back(newnode);
      for (int j = newelems.size()-1; j >= 0; j--)
      {
        new_elems.push_back(newelems[j]);
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
                       vector<unsigned int> &new_nodes,
                       vector<unsigned int> &new_elems) = 0;

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
                       vector<unsigned int> &new_nodes,
                       vector<unsigned int> &new_elems);


  struct ltvn
  {
    bool operator()(const typename IFIELD::mesh_type::Node::array_type &a,
                    const typename IFIELD::mesh_type::Node::array_type &b) const
    {
      for (unsigned int i = 0; i < a.size(); i++)
      {
        if (a[i] < b[i]) return true;
        if (a[i] > b[i]) return false;
      }
      return false;
    }
  };

  typedef std::set<typename IFIELD::mesh_type::Node::array_type, ltvn> added_set_type;
};


template <class TFIELD, class IFIELD>
void
InsertFieldExtractT<TFIELD, IFIELD>::extract(FieldHandle &result_field,
                                             MatrixHandle &result_mapping,
                                             FieldHandle tet_h,
                                             vector<unsigned int> &new_nodes,
                                             vector<unsigned int> &new_elems)
{
  TFIELD *tfield = dynamic_cast<TFIELD *>(tet_h.get_rep());
  typename TFIELD::mesh_handle_type tmesh = tfield->get_typed_mesh();

  typename IFIELD::mesh_handle_type omesh =
    scinew typename IFIELD::mesh_type();

  tfield->mesh()->synchronize(Mesh::EDGES_E | Mesh::FACES_E);

  std::sort(new_nodes.begin(), new_nodes.end());
  vector<unsigned int>::iterator nodes_end, itr;
  nodes_end = std::unique(new_nodes.begin(), new_nodes.end());
  for (itr = new_nodes.begin(); itr != nodes_end; ++itr)
  {
    Point p;
    tmesh->get_point(p, *itr);
    omesh->add_point(p);
  }

  if (omesh->dimensionality() > 0)
  {
    added_set_type already_added;

    std::sort(new_elems.begin(), new_elems.end());
    vector<unsigned int>::iterator elems_end;
    elems_end = std::unique(new_elems.begin(), new_elems.end());
    for (itr = new_elems.begin(); itr != elems_end; ++itr)
    {
      if (omesh->dimensionality() == 1)
      {
        typename TFIELD::mesh_type::Edge::array_type edges;
        tmesh->get_edges(edges,
                         typename TFIELD::mesh_type::Elem::index_type(*itr));
        for (unsigned int i = 0; i < edges.size(); i++)
        {
          typename TFIELD::mesh_type::Node::array_type oldnodes;
          typename IFIELD::mesh_type::Node::array_type newnodes;
          tmesh->get_nodes(oldnodes, edges[i]);
          bool all_found = true;
          for (unsigned int j = 0; j < oldnodes.size(); j++)
          {
            vector<unsigned int>::iterator loc =
              lower_bound(new_nodes.begin(), nodes_end, oldnodes[j]);
            if (loc != nodes_end && *loc == oldnodes[j])
            {
              newnodes.push_back(loc - new_nodes.begin());
            }
            else
            {
              all_found = false;
              break;
            }
          }
          
          if (all_found)
          {
            std::sort(newnodes.begin(), newnodes.end());
            typename added_set_type::iterator found =
              already_added.find(newnodes);
            if (found == already_added.end())
            {
              already_added.insert(newnodes);
              omesh->add_elem(newnodes);
            }
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
              std::lower_bound(new_nodes.begin(), nodes_end, oldnodes[j]);
            if (loc != nodes_end && *loc == oldnodes[j])
            {
              newnodes.push_back(loc - new_nodes.begin());
            }
            else
            {
              all_found = false;
              break;
            }
          }

          if (all_found)
          {
            std::sort(newnodes.begin(), newnodes.end());
            typename added_set_type::iterator found =
              already_added.find(newnodes);
            if (found == already_added.end())
            {
              already_added.insert(newnodes);
              omesh->add_elem(newnodes);
            }
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
            std::lower_bound(new_nodes.begin(), nodes_end, oldnodes[j]);
          if (loc != nodes_end && *loc == oldnodes[j])
          {
            newnodes.push_back(loc - new_nodes.begin());
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

  const int nrows = (int)onodesize;
  const int ncols = (int)tnodesize;
  int *rr = scinew int[nrows+1];
  int *cc = scinew int[nrows];
  double *d = scinew double[nrows];

  int i = 0;
  for (itr = new_nodes.begin(); itr != nodes_end; itr++)
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
