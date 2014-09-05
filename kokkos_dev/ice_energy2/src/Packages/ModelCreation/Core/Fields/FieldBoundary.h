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


#ifndef MODELCREATION_CORE_FIELDS_FIELDBOUNDARY_H
#define MODELCREATION_CORE_FIELDS_FIELDBOUNDARY_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <sci_hash_map.h>

namespace ModelCreation {

using namespace SCIRun;

class FieldBoundaryAlgo;

class FieldBoundaryAlgo : public DynamicAlgoBase
{
public:
  virtual bool FieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle& mapping);
  virtual bool testinput(FieldHandle input);

  static AlgoList<FieldBoundaryAlgo> precompiled_;
};


template <class FSRC, class FDST>
class FieldBoundaryVolumeAlgoT : public FieldBoundaryAlgo
{
public:
  virtual bool FieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle& mapping);
  virtual bool testinput(FieldHandle input);
};


template <class FSRC, class FDST>
class FieldBoundarySurfaceAlgoT : public FieldBoundaryAlgo
{
public:
  virtual bool FieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle& mapping);
  virtual bool testinput(FieldHandle input);
};

template <class FSRC, class FDST>
class FieldBoundaryCurveAlgoT : public FieldBoundaryAlgo
{
public:
  virtual bool FieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle& mapping);
  virtual bool testinput(FieldHandle input);
};



template <class FSRC, class FDST>
bool FieldBoundaryVolumeAlgoT<FSRC, FDST>::FieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle& mapping)
{
  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  {
    pr->error("FieldBoundary: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh == 0)
  {
    pr->error("FieldBoundary: No mesh associated with input field");
    return (false);
  }

  typename FDST::mesh_handle_type omesh = scinew typename FDST::mesh_type();
  if (omesh == 0)
  {
    pr->error("FieldBoundary: Could not create output field");
    return (false);
  }
  
  FDST *ofield = scinew FDST(omesh);
  output = dynamic_cast<Field*>(ofield);
  if (ofield == 0)
  {
    pr->error("FieldBoundary: Could not create output field");
    return (false);
  }
  
#ifdef HAVE_HASH_MAP
  typedef hash_map<unsigned int,unsigned int> hash_map_type;
#else
  typedef map<unsigned int,unsigned int> hash_map_type;
#endif
  hash_map_type node_map;
  hash_map_type elem_map;
  
  imesh->synchronize(Mesh::NODES_E|Mesh::FACES_E|Mesh::CELLS_E|Mesh::FACE_NEIGHBORS_E);
  
  typename FSRC::mesh_type::Elem::iterator be, ee;
  typename FSRC::mesh_type::Elem::index_type nci, ci;
  typename FSRC::mesh_type::Face::array_type faces; 
  typename FSRC::mesh_type::Node::array_type inodes; 
  typename FDST::mesh_type::Node::array_type onodes; 
  typename FSRC::mesh_type::Node::index_type a;

  inodes.clear();
  onodes.clear();  
  Point point;

  imesh->begin(be); 
  imesh->end(ee);

  while (be != ee) 
  {
    ci = *be;
    imesh->get_faces(faces,ci);
    for (size_t p =0; p < faces.size(); p++)
    {
      bool includeface = false;
      
      if(!(imesh->get_neighbor(nci,ci,faces[p]))) includeface = true;

      if (includeface)
      {
        imesh->get_nodes(inodes,faces[p]);
        if (onodes.size() == 0) onodes.resize(inodes.size());
        for (int q=0; q< onodes.size(); q++)
        {
          a = inodes[q];
          hash_map_type::iterator it = node_map.find(static_cast<unsigned int>(a));
          if (it == node_map.end())
          {
            imesh->get_center(point,a);
            onodes[q] = omesh->add_point(point);
            node_map[a] = static_cast<unsigned int>(onodes[q]);            
          }
          else
          {
            onodes[q] = static_cast<typename FDST::mesh_type::Node::index_type>(node_map[a]);
          }
        }
        elem_map[static_cast<unsigned int>(ci)] = omesh->add_elem(onodes);
      }
    }
    ++be;
  }
  
  mapping = 0;
  
  if (ifield->basis_order() == 0)
  {
    typename FSRC::Elem::size_type isize;
    typename FSRC::Elem::size_type osize;
    imesh->size(isize);
    omesh->size(osize);

    int nrows = static_cast<int>(osize);
    int ncols = static_cast<int>(isize);
    int *rr = scinew int[nrows+1];
    int *cc = scinew int[nrows];
    double *d = scinew double[nrows];

    for (int p = 0; p < nrows; p++)
    {
      cc[p] = 0;
      rr[p] = p;
      d[p] = 0.0;
    }
    rr[nrows] = nrows; // An extra entry goes on the end of rr.

    hash_map_type::iterator it, it_end;
    it = elem_map.begin();
    it_end = elem_map.end();
    
    while (it != it_end)
    {
      cc[(*it).second] = (*it).first;
      d[(*it).second] += 1.0;
    }
    
    mapping = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);
  }
  else if (ifield->basis_order() == 1)
  {
    typename FSRC::Node::size_type isize;
    typename FSRC::Node::size_type osize;
    imesh->size(isize);
    omesh->size(osize);

    int nrows = static_cast<int>(osize);
    int ncols = static_cast<int>(isize);
    int *rr = scinew int[nrows+1];
    int *cc = scinew int[nrows];
    double *d = scinew double[nrows];

    for (int p = 0; p < nrows; p++)
    {
      cc[p] = 0;
      rr[p] = p;
      d[p] = 0.0;
    }
    rr[nrows] = nrows; // An extra entry goes on the end of rr.

    hash_map_type::iterator it, it_end;
    it = node_map.begin();
    it_end = node_map.end();
    
    while (it != it_end)
    {
      cc[(*it).second] = (*it).first;
      d[(*it).second] += 1.0;
    }
    
    mapping = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);
  }
  
  // copy property manager
	output->copy_properties(input.get_rep());
  return (true);
}

template <class FSRC, class FDST>
bool FieldBoundaryVolumeAlgoT<FSRC, FDST>::testinput(FieldHandle input)
{
  return(dynamic_cast<FSRC*>(input.get_rep())!=0);
}

template <class FSRC, class FDST>
bool FieldBoundarySurfaceAlgoT<FSRC, FDST>::FieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle& mapping)
{
  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  {
    pr->error("FieldBoundary: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh == 0)
  {
    pr->error("FieldBoundary: No mesh associated with input field");
    return (false);
  }

  typename FDST::mesh_handle_type omesh = scinew typename FDST::mesh_type();
  if (omesh == 0)
  {
    pr->error("FieldBoundary: Could not create output field");
    return (false);
  }
  
  FDST *ofield = scinew FDST(omesh);
  output = dynamic_cast<Field*>(ofield);
  if (ofield == 0)
  {
    pr->error("FieldBoundary: Could not create output field");
    return (false);
  }
  
#ifdef HAVE_HASH_MAP
  typedef hash_map<unsigned int,unsigned int> hash_map_type;
#else
  typedef map<unsigned int,unsigned int> hash_map_type;
#endif
  hash_map_type node_map;
  hash_map_type elem_map;
  
  imesh->synchronize(Mesh::NODES_E|Mesh::FACES_E|Mesh::CELLS_E|Mesh::FACE_NEIGHBORS_E);
  
  typename FSRC::mesh_type::Elem::iterator be, ee;
  typename FSRC::mesh_type::Elem::index_type nci, ci;
  typename FSRC::mesh_type::Edge::array_type edges; 
  typename FSRC::mesh_type::Node::array_type inodes; 
  typename FDST::mesh_type::Node::array_type onodes; 
  typename FSRC::mesh_type::Node::index_type a;

  inodes.clear();
  onodes.clear();  
  Point point;

  imesh->begin(be); 
  imesh->end(ee);

  while (be != ee) 
  {
    ci = *be;
    imesh->get_edges(edges,ci);
    for (size_t p =0; p < edges.size(); p++)
    {
      bool includeface = false;
      
      if(!(imesh->get_neighbor(nci,ci,edges[p]))) includeface = true;

      if (includeface)
      {
        imesh->get_nodes(inodes,edges[p]);
        if (onodes.size() == 0) onodes.resize(inodes.size());
        for (int q=0; q< onodes.size(); q++)
        {
          a = inodes[q];
          hash_map_type::iterator it = node_map.find(static_cast<unsigned int>(a));
          if (it == node_map.end())
          {
            imesh->get_center(point,a);
            onodes[q] = omesh->add_point(point);
            node_map[a] = static_cast<unsigned int>(onodes[q]);            
          }
          else
          {
            onodes[q] = static_cast<typename FDST::mesh_type::Node::index_type>(node_map[a]);
          }
        }
        elem_map[static_cast<unsigned int>(ci)] = omesh->add_elem(onodes);
      }
    }
    ++be;
  }
  
  mapping = 0;
  
  if (ifield->basis_order() == 0)
  {
    typename FSRC::Elem::size_type isize;
    typename FSRC::Elem::size_type osize;
    imesh->size(isize);
    omesh->size(osize);

    int nrows = static_cast<int>(osize);
    int ncols = static_cast<int>(isize);
    int *rr = scinew int[nrows+1];
    int *cc = scinew int[nrows];
    double *d = scinew double[nrows];

    for (int p = 0; p < nrows; p++)
    {
      cc[p] = 0;
      rr[p] = p;
      d[p] = 0.0;
    }
    rr[nrows] = nrows; // An extra entry goes on the end of rr.

    hash_map_type::iterator it, it_end;
    it = elem_map.begin();
    it_end = elem_map.end();
    
    while (it != it_end)
    {
      cc[(*it).second] = (*it).first;
      d[(*it).second] += 1.0;
    }
    
    mapping = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);
  }
  else if (ifield->basis_order() == 1)
  {
    typename FSRC::Node::size_type isize;
    typename FSRC::Node::size_type osize;
    imesh->size(isize);
    omesh->size(osize);

    int nrows = static_cast<int>(osize);
    int ncols = static_cast<int>(isize);
    int *rr = scinew int[nrows+1];
    int *cc = scinew int[nrows];
    double *d = scinew double[nrows];

    for (int p = 0; p < nrows; p++)
    {
      cc[p] = 0;
      rr[p] = p;
      d[p] = 0.0;
    }
    rr[nrows] = nrows; // An extra entry goes on the end of rr.

    hash_map_type::iterator it, it_end;
    it = node_map.begin();
    it_end = node_map.end();
    
    while (it != it_end)
    {
      cc[(*it).second] = (*it).first;
      d[(*it).second] += 1.0;
    }
    
    mapping = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);
  }
  
  // copy property manager
	output->copy_properties(input.get_rep());
  return (true);
}

template <class FSRC, class FDST>
bool FieldBoundarySurfaceAlgoT<FSRC, FDST>::testinput(FieldHandle input)
{
  return(dynamic_cast<FSRC*>(input.get_rep())!=0);
}


template <class FSRC, class FDST>
bool FieldBoundaryCurveAlgoT<FSRC, FDST>::FieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle& mapping)
{
  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  {
    pr->error("FieldBoundary: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh == 0)
  {
    pr->error("FieldBoundary: No mesh associated with input field");
    return (false);
  }

  typename FDST::mesh_handle_type omesh = scinew typename FDST::mesh_type();
  if (omesh == 0)
  {
    pr->error("FieldBoundary: Could not create output field");
    return (false);
  }
  
  FDST *ofield = scinew FDST(omesh);
  output = dynamic_cast<Field*>(ofield);
  if (ofield == 0)
  {
    pr->error("FieldBoundary: Could not create output field");
    return (false);
  }
  
#ifdef HAVE_HASH_MAP
  typedef hash_map<unsigned int,unsigned int> hash_map_type;
#else
  typedef map<unsigned int,unsigned int> hash_map_type;
#endif
  hash_map_type node_map;
  hash_map_type elem_map;
  
  imesh->synchronize(Mesh::NODES_E|Mesh::EDGES_E|Mesh::FACE_NEIGHBORS_E);
  
  typename FSRC::mesh_type::Elem::iterator be, ee;
  typename FSRC::mesh_type::Elem::index_type nci, ci;
  typename FSRC::mesh_type::Node::array_type nodes; 
  typename FSRC::mesh_type::Node::array_type inodes; 
  typename FDST::mesh_type::Node::array_type onodes; 
  typename FSRC::mesh_type::Node::index_type a;

  inodes.clear();
  onodes.clear();  
  Point point;

  imesh->begin(be); 
  imesh->end(ee);

  while (be != ee) 
  {
    ci = *be;
    imesh->get_nodes(nodes,ci);
    for (size_t p =0; p < nodes.size(); p++)
    {
      bool includeface = false;
      
      if(!(imesh->get_neighbor(nci,ci,nodes[p]))) includeface = true;

      if (includeface)
      {
        imesh->get_nodes(inodes,nodes[p]);
        if (onodes.size() == 0) onodes.resize(inodes.size());
        for (int q=0; q< onodes.size(); q++)
        {
          a = inodes[q];
          hash_map_type::iterator it = node_map.find(static_cast<unsigned int>(a));
          if (it == node_map.end())
          {
            imesh->get_center(point,a);
            onodes[q] = omesh->add_point(point);
            node_map[a] = static_cast<unsigned int>(onodes[q]);            
          }
          else
          {
            onodes[q] = static_cast<typename FDST::mesh_type::Node::index_type>(node_map[a]);
          }
        }
        elem_map[static_cast<unsigned int>(ci)] = omesh->add_elem(onodes);
      }
    }
    ++be;
  }
  
  mapping = 0;
  
  if (ifield->basis_order() == 0)
  {
    typename FSRC::Elem::size_type isize;
    typename FSRC::Elem::size_type osize;
    imesh->size(isize);
    omesh->size(osize);

    int nrows = static_cast<int>(osize);
    int ncols = static_cast<int>(isize);
    int *rr = scinew int[nrows+1];
    int *cc = scinew int[nrows];
    double *d = scinew double[nrows];

    for (int p = 0; p < nrows; p++)
    {
      cc[p] = 0;
      rr[p] = p;
      d[p] = 0.0;
    }
    rr[nrows] = nrows; // An extra entry goes on the end of rr.

    hash_map_type::iterator it, it_end;
    it = elem_map.begin();
    it_end = elem_map.end();
    
    while (it != it_end)
    {
      cc[(*it).second] = (*it).first;
      d[(*it).second] += 1.0;
    }
    
    mapping = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);
  }
  else if (ifield->basis_order() == 1)
  {
    typename FSRC::Node::size_type isize;
    typename FSRC::Node::size_type osize;
    imesh->size(isize);
    omesh->size(osize);

    int nrows = static_cast<int>(osize);
    int ncols = static_cast<int>(isize);
    int *rr = scinew int[nrows+1];
    int *cc = scinew int[nrows];
    double *d = scinew double[nrows];

    for (int p = 0; p < nrows; p++)
    {
      cc[p] = 0;
      rr[p] = p;
      d[p] = 0.0;
    }
    rr[nrows] = nrows; // An extra entry goes on the end of rr.

    hash_map_type::iterator it, it_end;
    it = node_map.begin();
    it_end = node_map.end();
    
    while (it != it_end)
    {
      cc[(*it).second] = (*it).first;
      d[(*it).second] += 1.0;
    }
    
    mapping = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);
  }
  
  // copy property manager
	output->copy_properties(input.get_rep());
  return (true);
}

template <class FSRC, class FDST>
bool FieldBoundaryCurveAlgoT<FSRC, FDST>::testinput(FieldHandle input)
{
  return(dynamic_cast<FSRC*>(input.get_rep())!=0);
}


} // end namespace ModelCreation

#endif 
