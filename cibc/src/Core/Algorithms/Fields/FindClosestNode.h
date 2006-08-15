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


#ifndef CORE_ALGORITHMS_FIELDS_FINDCLOSESTNODE_H
#define CORE_ALGORITHMS_FIELDS_FINDCLOSESTNODE_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <float.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class FindClosestNodeAlgo : public DynamicAlgoBase
{
public:
  virtual bool FindClosestNode(ProgressReporter *pr, FieldHandle input, std::vector<unsigned int>& output, FieldHandle& points);
};


template <class FSRC, class FPNT>
class FindClosestNodeAlgoT : public FindClosestNodeAlgo
{
public:
  virtual bool FindClosestNode(ProgressReporter *pr, FieldHandle input, std::vector<unsigned int>& output, FieldHandle& points);
};


template <class FSRC, class FPNT>
bool FindClosestNodeAlgoT<FSRC, FPNT>::FindClosestNode(ProgressReporter *pr, FieldHandle input, std::vector<unsigned int>& output, FieldHandle& points)
{

  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  {
    pr->error("FindClosestNode: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh == 0)
  {
    pr->error("FindClosestNode: No mesh associated with input field");
    return (false);
  }


  FPNT *pfield = dynamic_cast<FPNT *>(points.get_rep());
  if (pfield == 0)
  {
    pr->error("FindClosestNode: Could not obtain input field with node locations");
    return (false);
  }

  typename FPNT::mesh_handle_type pmesh = pfield->get_typed_mesh();
  if (pmesh == 0)
  {
    pr->error("FindClosestNode: No mesh associated with input field with node locations");
    return (false);
  }

  typename FPNT::mesh_type::Node::size_type nnodes;
  pmesh->size(nnodes);
  
  if (nnodes == 0)
  {
    pr->error("FindClosestNode: No nodes locations are given in node mesh");
    return (false);  
  }

  typename FSRC::mesh_type::Node::size_type innodes;
  imesh->size(innodes);

  if (innodes == 0)
  {
    pr->error("FindClosestNode: Number of nodes in input field is 0");
    return (false);  
  }

  output.resize(nnodes);
  
  imesh->synchronize(Mesh::LOCATE_E);
       
  typename FPNT::mesh_type::Node::iterator pit, pit_end;
  typename FSRC::mesh_type::Node::iterator it, it_end;
  typename FSRC::mesh_type::Node::index_type idx;
  typename FSRC::mesh_type::Node::array_type nodes;
  typename FSRC::mesh_type::Elem::index_type cidx;
  double dist = DBL_MAX;
  double dist2;
  Point p, q;
  size_t m = 0;
          
  pmesh->begin(pit);
  pmesh->end(pit_end);
  while (pit != pit_end)
  {
    pmesh->get_center(p,*pit);
    dist = DBL_MAX;
    
    if (imesh->locate(cidx,p))
    {
      imesh->get_nodes(nodes,cidx);
      for (int r=0; r< nodes.size(); r++)
      {
        imesh->get_center(q,nodes[r]);
        dist2 = Vector(p-q).length2();
        if (dist2 < dist)
        {
          idx = nodes[r];
          dist = dist2; 
        }
      }
    }
    else
    {
      imesh->begin(it);
      imesh->end(it_end);
      while (it != it_end)
      {  
        imesh->get_center(q,*it);
        dist2 = Vector(p-q).length2();
        if (dist2 < dist) { idx = *it; dist = dist2; }
        ++it;
      }
    }
    ++pit;
    output[m] = static_cast<unsigned int>(idx);
    m++;
  }
  
  return (true); 
}


} // end namespace

#endif
