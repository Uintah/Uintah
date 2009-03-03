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


#ifndef CORE_ALGORITHMS_FIELDS_FIELDBOUNDARY_H
#define CORE_ALGORITHMS_FIELDS_FIELDBOUNDARY_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <sci_hash_map.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class FieldBoundaryAlgo : public DynamicAlgoBase
{
public:
  virtual bool FieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle& mapping);
};


template <class FSRC, class FDST>
class FieldBoundaryAlgoT : public FieldBoundaryAlgo
{
public:
  virtual bool FieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle& mapping);
};


template <class FSRC, class FDST>
bool FieldBoundaryAlgoT<FSRC, FDST>::FieldBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle& mapping)
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
  
  if (imesh->dimensionality() == 1) imesh->synchronize(Mesh::NODES_E|Mesh::EDGES_E);
  if (imesh->dimensionality() == 2) imesh->synchronize(Mesh::NODES_E|Mesh::EDGES_E|Mesh::FACES_E|Mesh::EDGE_NEIGHBORS_E);
  if (imesh->dimensionality() == 3) imesh->synchronize(Mesh::NODES_E|Mesh::FACES_E|Mesh::CELLS_E|Mesh::FACE_NEIGHBORS_E);
  
  typename FSRC::mesh_type::Elem::iterator be, ee;
  typename FSRC::mesh_type::Elem::index_type nci, ci;
  typename FSRC::mesh_type::DElem::array_type delems; 
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
    imesh->get_delems(delems,ci);
    for (size_t p =0; p < delems.size(); p++)
    {
      bool includeface = false;
      
      if(!(imesh->get_neighbor(nci,ci,delems[p]))) includeface = true;

      if (includeface)
      {
        imesh->get_nodes(inodes,delems[p]);
        if (onodes.size() == 0) onodes.resize(inodes.size());
        for (int q=0; q< onodes.size(); q++)
        {
          a = inodes[q];
          hash_map_type::iterator it = node_map.find(static_cast<unsigned int>(a));
          if (it == node_map.end())
          {
            imesh->get_center(point,a);
            onodes[q] = omesh->add_point(point);
            node_map[static_cast<unsigned int>(a)] = static_cast<unsigned int>(onodes[q]);            
          }
          else
          {
            onodes[q] = static_cast<typename FDST::mesh_type::Node::index_type>(node_map[static_cast<unsigned int>(a)]);
          }
        }
        elem_map[static_cast<unsigned int>(omesh->add_elem(onodes))] = static_cast<unsigned int>(ci);
      }
    }
    ++be;
  }
  
  mapping = 0;
  
  ofield->resize_fdata();
  
  if (ifield->basis_order() == 0)
  {
    typename FSRC::mesh_type::Elem::size_type isize;
    typename FDST::mesh_type::Elem::size_type osize;
    typename FDST::value_type val;
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
      cc[(*it).first] = (*it).second;
      d[(*it).first] += 1.0;
      
      typename FSRC::mesh_type::Elem::index_type idx1;
      typename FDST::mesh_type::Elem::index_type idx2;
      imesh->to_index(idx1,(*it).second);
      omesh->to_index(idx2,(*it).first);
      ifield->value(val,idx1);
      ofield->set_value(val,idx2);
      ++it;
    }
    
    mapping = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);
  }
  else if (ifield->basis_order() == 1)
  {
    typename FSRC::mesh_type::Node::size_type isize;
    typename FDST::mesh_type::Node::size_type osize;
    typename FDST::value_type val;
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

      typename FSRC::mesh_type::Node::index_type idx1;
      typename FDST::mesh_type::Node::index_type idx2;
      imesh->to_index(idx1,(*it).first);
      omesh->to_index(idx2,(*it).second);
      ifield->value(val,idx1);
      ofield->set_value(val,idx2);
      ++it;
    }
    
    mapping = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);
  }
  
  // copy property manager
	output->copy_properties(input.get_rep());
  return (true);
}

} // end namespace SCIRunAlgo

#endif 
