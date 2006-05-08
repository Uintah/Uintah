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


#ifndef CORE_ALGORITHMS_FIELDS_DOMAINBOUNDARY_H
#define CORE_ALGORITHMS_FIELDS_DOMAINBOUNDARY_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <sci_hash_map.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class DomainBoundaryAlgo : public DynamicAlgoBase
{
public:
  virtual bool DomainBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle DomainLink, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly);
};


template <class FSRC, class FDST>
class DomainBoundaryAlgoT : public DomainBoundaryAlgo
{
public:
  virtual bool DomainBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle DomainLink, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly);

private:
  typedef class {
  public:
    typename FDST::mesh_type::Node::index_type node;
    typename FSRC::value_type val1;
    typename FSRC::value_type val2;      
    bool hasneighbor;
  } pointtype;

};


template <class FSRC, class FDST>
bool DomainBoundaryAlgoT<FSRC, FDST>::DomainBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle DomainLink, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly)
{
  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  {
    pr->error("DomainBoundary: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh == 0)
  {
    pr->error("DomainBoundary: No mesh associated with input field");
    return (false);
  }

  typename FDST::mesh_handle_type omesh = scinew typename FDST::mesh_type();
  if (omesh == 0)
  {
    pr->error("DomainBoundary: Could not create output field");
    return (false);
  }
  
  FDST *ofield = scinew FDST(omesh);
  output = dynamic_cast<Field*>(ofield);
  if (ofield == 0)
  {
    pr->error("DomainBoundary: Could not create output field");
    return (false);
  }
  
#ifdef HAVE_HASH_MAP
  typedef hash_multimap<unsigned int,pointtype> hash_map_type;
#else
  typedef multimap<unsigned int,pointtype> hash_map_type;
#endif

  if (imesh->dimensionality() == 1) imesh->synchronize(Mesh::NODES_E|Mesh::EDGES_E|Mesh::NODE_NEIGHBORS_E);
  if (imesh->dimensionality() == 2) imesh->synchronize(Mesh::EDGES_E|Mesh::FACES_E|Mesh::EDGE_NEIGHBORS_E|Mesh::NODE_NEIGHBORS_E);
  if (imesh->dimensionality() == 3) imesh->synchronize(Mesh::CELLS_E|Mesh::FACES_E|Mesh::FACE_NEIGHBORS_E|Mesh::NODE_NEIGHBORS_E);
  
  typename FSRC::mesh_type::Node::size_type numnodes;
  typename FSRC::mesh_type::DElem::size_type numdelems;

  imesh->size(numnodes);
  imesh->size(numdelems);

  bool isdomlink = false;
  int* domlinkrr = 0;
  int* domlinkcc = 0;
  
  if (DomainLink.get_rep())
  {
    if ((numdelems != DomainLink->nrows())&&(numdelems != DomainLink->ncols()))
    {
      pr->error("DomainBoundary: The Domain Link property is not of the right dimensions");
      return (false);        
    }
    SparseRowMatrix *spr = dynamic_cast<SparseRowMatrix *>(DomainLink.get_rep());
    if (spr)
    {
      domlinkrr = spr->rows;
      domlinkcc = spr->columns;
      isdomlink = true;
    }
  }  

  hash_map_type node_map;
  
  typename FSRC::mesh_type::Elem::iterator be, ee;
  typename FSRC::mesh_type::Elem::index_type nci, ci;
  typename FSRC::mesh_type::DElem::array_type delems; 
  typename FSRC::mesh_type::Node::array_type inodes; 
  typename FDST::mesh_type::Node::array_type onodes; 
  typename FSRC::mesh_type::Node::index_type a;
  typename FSRC::value_type val1, val2, minval, maxval;

  minval = static_cast<typename FSRC::value_type>(minrange);
  maxval = static_cast<typename FSRC::value_type>(maxrange);
  
  Point point;

  imesh->begin(be); 
  imesh->end(ee);

  while (be != ee) 
  {
  
    ci = *be;
    imesh->get_delems(delems,ci);

    for (size_t p =0; p < delems.size(); p++)
    {
      bool neighborexist = false;
      bool includeface = false;

      neighborexist = imesh->get_neighbor(nci,ci,delems[p]);

      if ((!neighborexist)&&(isdomlink))
      {
        for (int rr = domlinkrr[static_cast<int>(delems[p])]; rr < domlinkrr[static_cast<int>(delems[p])+1]; rr++)
        {
          int cc = domlinkcc[rr];
          typename FSRC::mesh_type::Node::array_type nodes;
          typename FSRC::mesh_type::Elem::array_type elems;           
          typename FSRC::mesh_type::DElem::array_type delems2;           
          typename FSRC::mesh_type::DElem::index_type idx;

          imesh->to_index(idx,cc);
          imesh->get_nodes(nodes,idx);       
          imesh->get_elems(elems,nodes[0]);

          for (int r=0; r<elems.size(); r++)
          {
            imesh->get_delems(delems2,elems[r]);

            for (int s=0; s<delems2.size(); s++)
            {
              if (delems2[s]==idx) { nci = elems[r]; neighborexist = true; break; }
            }
            if (neighborexist) break;
          }
          if (neighborexist) break;
        }
      }

      if (neighborexist)
      {
        if (nci > ci)
        {
          ifield->value(val1,ci);
          ifield->value(val2,nci);
          if (innerboundaryonly == false)
          {
            if ((((val1 >= minval)&&(val1 <= maxval))||((val2 >= minval)&&(val2 <= maxval)))||(userange == false))
            {
              if (!(val1 == val2)) includeface = true;             
            }
          }
          else
          {
            if ((((val1 >= minval)&&(val2 >= minval))&&((val1 <= maxval)&&(val2 <= maxval)))||(userange == false))
            {
              if (!(val1 == val2)) includeface = true;             
            }          
          }
        }
      }
      else if ((addouterboundary)&&(innerboundaryonly == false))
      {
        ifield->value(val1,ci);
        if (((val1 >= minval)&&(val1 <= maxval))||(userange == false)) includeface = true;
      }

      if (includeface)
      {
        imesh->get_nodes(inodes,delems[p]);
        onodes.resize(inodes.size());
        for (int q=0; q< onodes.size(); q++)
        {
          a = inodes[q];
          
          std::pair<typename hash_map_type::iterator,typename hash_map_type::iterator> lit;
          lit = node_map.equal_range(static_cast<unsigned int>(a));
          
          typename FDST::mesh_type::Node::index_type nodeidx;
          typename FSRC::value_type v1, v2;
          bool hasneighbor;
          
          if (neighborexist)
          {
            if (val1 < val2) { v1 = val1; v2 = val2; } else { v1 = val2; v2 = val1; }
            hasneighbor = true;
          }
          else
          {
            v1 = val1; v2 = 0;
            hasneighbor = false;
          }
          
          while (lit.first != lit.second)
          {
            if (((*(lit.first)).second.val1 == v1)&&((*(lit.first)).second.val2 == v2)&&((*(lit.first)).second.hasneighbor == hasneighbor))
            {
              nodeidx = (*(lit.first)).second.node;
              break;
            }
            ++(lit.first);
          }
          
          if (lit.first == lit.second)
          {
            pointtype newpoint;
            imesh->get_center(point,a);
            onodes[q] = omesh->add_point(point);
            newpoint.node = onodes[q];
            newpoint.val1 = v1;
            newpoint.val2 = v2;
            newpoint.hasneighbor = hasneighbor;
            node_map.insert(typename hash_map_type::value_type(a,newpoint));
          }
          else
          {
            onodes[q] = nodeidx;
          }
          
        }
        omesh->add_elem(onodes);
      }
    }
    ++be;
  }
  
  // copy property manager
	output->copy_properties(input.get_rep());
  return (true);
}

} // end namespace SCIRunAlgo

#endif 
