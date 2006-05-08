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

#ifndef CORE_ALGORITHMS_FIELDS_CONVERTTOTRISURF_H
#define CORE_ALGORITHMS_FIELDS_CONVERTTOTRISURF_H 1

// The following include file will include all tools needed for doing 
// dynamic compilation and will include all the standard dataflow types
#include <Core/Algorithms/Util/DynamicAlgo.h>

// Additionally we include sci_hash_map here as it is needed by the algorithm

namespace SCIRunAlgo {

using namespace SCIRun;

class ConvertToTriSurfAlgo : public DynamicAlgoBase
{
public:
  virtual bool ConvertToTriSurf(ProgressReporter *pr, FieldHandle input, FieldHandle& output);
};

template <class FSRC, class FDST>
class ConvertImageToTriSurfAlgoT : public ConvertToTriSurfAlgo
{
public:
  virtual bool ConvertToTriSurf(ProgressReporter *pr, FieldHandle input, FieldHandle& output);
};

template <class FSRC, class FDST>
class ConvertQuadSurfToTriSurfAlgoT : public ConvertToTriSurfAlgo
{
public:
  virtual bool ConvertToTriSurf(ProgressReporter *pr, FieldHandle input, FieldHandle& output);
};


template <class FSRC, class FDST>
bool ConvertQuadSurfToTriSurfAlgoT<FSRC, FDST>::ConvertToTriSurf(ProgressReporter *pr, FieldHandle input, FieldHandle& output)
{

  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  { 
    pr->error("ConvertToTriSurf: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh == 0)
  {
    pr->error("ConvertToTriSurf: No mesh associated with input field");
    return (false);
  }

  typename FDST::mesh_handle_type omesh = scinew typename FDST::mesh_type();
  if (omesh == 0)
  {
    pr->error("ConvertToTriSurf: Could not create output field");
    return (false);
  }
  
  typename FSRC::mesh_type::Node::size_type numnodes; 
  typename FSRC::mesh_type::Elem::size_type numelems; 
  imesh->size(numnodes);
  imesh->size(numelems);
  
  omesh->node_reserve(static_cast<unsigned int>(numnodes));

  typename FSRC::mesh_type::Node::iterator nbi, nei;
  typename FSRC::mesh_type::Elem::iterator ebi, eei;
  typename FDST::mesh_type::Node::iterator dbi, dei;
    
  imesh->begin(nbi); 
  imesh->end(nei);
  while (nbi != nei)
  {
    Point point;
    imesh->get_center(point, *nbi);
    omesh->add_point(point);
    ++nbi;
  }

  imesh->synchronize(Mesh::NODE_NEIGHBORS_E);
  omesh->elem_reserve(static_cast<unsigned int>(numelems*2));

  vector<typename FDST::mesh_type::Elem::index_type> elemmap(numelems);
  vector<signed char> visited(numelems, 0);
  vector<unsigned char> nodeisdiagonal(numnodes,0);

  typename FSRC::mesh_type::Elem::iterator bi, ei;
  imesh->begin(bi); imesh->end(ei);

  size_t surfsize = static_cast<size_t>(pow(numelems, 2.0 / 3.0));
  vector<typename FSRC::mesh_type::Elem::index_type> buffer;
  buffer.reserve(surfsize);
  
  imesh->synchronize(Mesh::EDGES_E);

  while (bi != ei)
  {
    // if list of elements to process is empty ad the next one
    if (buffer.size() == 0)
    {
      if(visited[static_cast<unsigned int>(*bi)] == 0) 
      {
        typename FSRC::mesh_type::Node::array_type qsnodes;
        buffer.push_back(*bi);
        imesh->get_nodes(qsnodes,*bi);
        nodeisdiagonal[static_cast<unsigned int>(qsnodes[0])] = true;
        nodeisdiagonal[static_cast<unsigned int>(qsnodes[2])] = true;        
      }
    }
    
    if (buffer.size() > 0)
    {
      for (unsigned int i=0; i< buffer.size(); i++)
      {
        if (visited[static_cast<unsigned int>(buffer[i])] > 0) { continue; }
        
        typename FSRC::mesh_type::Cell::array_type neighbors;
        imesh->get_neighbors(neighbors, buffer[i]);
 
        typename FSRC::mesh_type::Node::array_type qsnodes;
        imesh->get_nodes(qsnodes,buffer[i]);
 
        for (unsigned int p=0; p<neighbors.size(); p++)
        {
          // bigger than 0 => already processed
          // smaller than 0 => already on the list
          if(visited[static_cast<unsigned int>(neighbors[p])] == 0)
          {
            buffer.push_back(neighbors[p]);
            visited[static_cast<unsigned int>(neighbors[p])] = -1;
          }
        }

        // In case mesh is weird an not logically numbered
        if (static_cast<unsigned int>(buffer[i]) >= elemmap.size()) elemmap.resize(static_cast<unsigned int>(buffer[i]));

 
        if (nodeisdiagonal[static_cast<unsigned int>(qsnodes[0])] || 
            nodeisdiagonal[static_cast<unsigned int>(qsnodes[2])])
        {
            nodeisdiagonal[static_cast<unsigned int>(qsnodes[0])] = true;
            nodeisdiagonal[static_cast<unsigned int>(qsnodes[2])] = true;
            elemmap[static_cast<unsigned int>(buffer[i])] =
            omesh->add_triangle((typename FDST::mesh_type::Node::index_type)(qsnodes[0]),
                                (typename FDST::mesh_type::Node::index_type)(qsnodes[1]),
                                (typename FDST::mesh_type::Node::index_type)(qsnodes[2]));

            omesh->add_triangle((typename FDST::mesh_type::Node::index_type)(qsnodes[0]),
                                (typename FDST::mesh_type::Node::index_type)(qsnodes[2]),
                                (typename FDST::mesh_type::Node::index_type)(qsnodes[3]));
            visited[static_cast<unsigned int>(buffer[i])] = 1; 
        }
        else
        {
            nodeisdiagonal[static_cast<unsigned int>(qsnodes[1])] = true;
            nodeisdiagonal[static_cast<unsigned int>(qsnodes[3])] = true;
            elemmap[static_cast<unsigned int>(buffer[i])] =
            omesh->add_triangle((typename FDST::mesh_type::Node::index_type)(qsnodes[0]),
                                (typename FDST::mesh_type::Node::index_type)(qsnodes[1]),
                                (typename FDST::mesh_type::Node::index_type)(qsnodes[3]));

            omesh->add_triangle((typename FDST::mesh_type::Node::index_type)(qsnodes[1]),
                                (typename FDST::mesh_type::Node::index_type)(qsnodes[2]),
                                (typename FDST::mesh_type::Node::index_type)(qsnodes[3]));        
            visited[static_cast<unsigned int>(buffer[i])] = 2; 
        }
      }
      buffer.clear();
    }
    ++bi;
  }
  
  FDST* ofield = scinew FDST(omesh);
  if (ofield == 0)
  {
    pr->error("ConvertToTriSurf: Could not create output field");
    return (false);  
  }
  
  output = dynamic_cast<Field*>(ofield);
  ofield->resize_fdata();

  if (ifield->basis_order() == 0)
  {
    imesh->begin(ebi); 
    imesh->end(eei);
    typename FSRC::value_type val;
    typename FDST::mesh_type::Elem::index_type idx;
    
    while (ebi != eei)
    {
      idx = elemmap[static_cast<unsigned int>(*ebi)];
      ifield->value(val, *ebi);
      ofield->set_value(val, idx);
      ofield->set_value(val, idx+1);
      ++ebi;
    }
  }
  
  if (ifield->basis_order() == 1)
  {
    imesh->begin(nbi);
    imesh->end(nei);
    omesh->begin(dbi); 
    omesh->end(dei);
    typename FSRC::value_type val;

    while (nbi != nei)
    {
      ifield->value(val,*nbi);
      ofield->set_value(val,*dbi);
      ++dbi; ++nbi;
    }
  }

	output->copy_properties(input.get_rep());
  
  // Success:
  return (true);
}


template <class FSRC, class FDST>
bool ConvertImageToTriSurfAlgoT<FSRC, FDST>::ConvertToTriSurf(ProgressReporter *pr, FieldHandle input, FieldHandle& output)
{

  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  { 
    pr->error("ConvertToTriSurf: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh == 0)
  {
    pr->error("ConvertToTriSurf: No mesh associated with input field");
    return (false);
  }

  typename FDST::mesh_handle_type omesh = scinew typename FDST::mesh_type();
  if (omesh == 0)
  {
    pr->error("ConvertToTriSurf: Could not create output field");
    return (false);
  }

  typename FSRC::mesh_type::Node::size_type numnodes;
  imesh->size(numnodes);
  omesh->node_reserve(static_cast<unsigned int>(numnodes));

  // Copy points directly, assuming they will have the same order.
  typename FSRC::mesh_type::Node::iterator nbi, nei;
  typename FDST::mesh_type::Node::iterator dbi, dei;
  imesh->begin(nbi); 
  imesh->end(nei);
  
  while (nbi != nei)
  {
    Point point;
    imesh->get_center(point, *nbi);
    omesh->add_point(point);
    ++nbi;
  }

  typename FSRC::mesh_type::Elem::size_type numelems;
  imesh->size(numelems);
  omesh->elem_reserve(static_cast<unsigned int>(numelems*2));

  typename FSRC::mesh_type::Elem::iterator bi, ei;
  typename FDST::mesh_type::Elem::iterator obi, oei;

  imesh->begin(bi); 
  imesh->end(ei);
  
  while (bi != ei)
  {
    typename FSRC::mesh_type::Node::array_type qsnodes;
    
    imesh->get_nodes(qsnodes, *bi);
    
    if (!(((*bi).i_ ^ (*bi).j_)&1))
    {
      omesh->add_triangle((typename FDST::mesh_type::Node::index_type)(qsnodes[0]),
                          (typename FDST::mesh_type::Node::index_type)(qsnodes[1]),
                          (typename FDST::mesh_type::Node::index_type)(qsnodes[2]));

      omesh->add_triangle((typename FDST::mesh_type::Node::index_type)(qsnodes[0]),
                          (typename FDST::mesh_type::Node::index_type)(qsnodes[2]),
                          (typename FDST::mesh_type::Node::index_type)(qsnodes[3]));
    }
    else
    {
      omesh->add_triangle((typename FDST::mesh_type::Node::index_type)(qsnodes[0]),
                          (typename FDST::mesh_type::Node::index_type)(qsnodes[1]),
                          (typename FDST::mesh_type::Node::index_type)(qsnodes[3]));

      omesh->add_triangle((typename FDST::mesh_type::Node::index_type)(qsnodes[1]),
                          (typename FDST::mesh_type::Node::index_type)(qsnodes[2]),
                          (typename FDST::mesh_type::Node::index_type)(qsnodes[3]));    
    }
    ++bi;
  }
  
  FDST* ofield = scinew FDST(omesh);
  if (ofield == 0)
  {
    pr->error("ConvertToTriSurf: Could not create output field");
    return (false);  
  }
  
  output = dynamic_cast<Field*>(ofield);
  ofield->resize_fdata();

  if (ifield->basis_order() == 0)
  {
    imesh->begin(bi); 
    imesh->end(ei);
    omesh->begin(obi); 
    omesh->end(oei);

    typename FSRC::value_type val;
    while (bi != ei)
    {
      ifield->value(val,*bi); ++bi;
      ofield->set_value(val,*obi); ++obi;
      ofield->set_value(val,*obi); ++obi;
    }
  }
  
  if (ifield->basis_order() == 1)
  {
    imesh->begin(nbi);
    imesh->end(nei);
    omesh->begin(dbi); 
    omesh->end(dei);

    typename FSRC::value_type val;
    while (nbi != nei)
    {
      ifield->value(val,*nbi);
      ofield->set_value(val,*dbi);
      ++dbi; ++nbi;
    }
  }

	output->copy_properties(input.get_rep());
  
  // Success:
  return (true);
}
  
} // end namespace SCIRunAlgo

#endif 

