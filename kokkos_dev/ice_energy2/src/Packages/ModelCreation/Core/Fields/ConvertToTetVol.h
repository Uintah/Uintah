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

#ifndef MODELCREATION_CORE_FIELDS_CONVERTTOTETVOL_H
#define MODELCREATION_CORE_FIELDS_CONVERTTOTETVOL_H 1

// The following include file will include all tools needed for doing 
// dynamic compilation and will include all the standard dataflow types
#include <Core/Algorithms/Util/DynamicAlgo.h>

// Additionally we include sci_hash_map here as it is needed by the algorithm

namespace ModelCreation {

using namespace SCIRun;

class ConvertToTetVolAlgo;

class ConvertToTetVolAlgo : public DynamicAlgoBase
{
public:
  virtual bool ConvertToTetVol(ProgressReporter *pr, FieldHandle input, FieldHandle& output);
  virtual bool testinput(FieldHandle input);

  static AlgoList<ConvertToTetVolAlgo> precompiled_;
};

template <class FSRC, class FDST>
class ConvertLatVolToTetVolAlgoT : public ConvertToTetVolAlgo
{
public:
  virtual bool ConvertToTetVol(ProgressReporter *pr, FieldHandle input, FieldHandle& output);
  virtual bool testinput(FieldHandle input);
};

template <class FSRC, class FDST>
class ConvertHexVolToTetVolAlgoT : public ConvertToTetVolAlgo
{
public:
  virtual bool ConvertToTetVol(ProgressReporter *pr, FieldHandle input, FieldHandle& output);
  virtual bool testinput(FieldHandle input);
};


template <class FSRC, class FDST>
bool ConvertHexVolToTetVolAlgoT<FSRC, FDST>::ConvertToTetVol(ProgressReporter *pr, FieldHandle input, FieldHandle& output)
{

  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  { 
    pr->error("ConvertToTetVol: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh == 0)
  {
    pr->error("ConvertToTetVol: No mesh associated with input field");
    return (false);
  }

  typename FDST::mesh_handle_type omesh = scinew typename FDST::mesh_type();
  if (omesh == 0)
  {
    pr->error("ConvertToTetVol: Could not create output field");
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
  omesh->elem_reserve(static_cast<unsigned int>(numelems*5));

  vector<typename FDST::mesh_type::Elem::index_type> elemmap(numelems);
  vector<char> visited(numelems, 0);

  typename FSRC::mesh_type::Elem::iterator bi, ei;
  imesh->begin(bi); imesh->end(ei);

  size_t surfsize = static_cast<size_t>(pow(numelems, 2.0 / 3.0));
  vector<typename FSRC::mesh_type::Elem::index_type> buffer;
  buffer.reserve(surfsize);
  
  imesh->synchronize(Mesh::FACES_E);

  while (bi != ei)
  {
    // if list of elements to process is empty ad the next one
    if (buffer.size() == 0)
    {
      if(visited[static_cast<unsigned int>(*bi)] == 0) buffer.push_back(*bi);
    }
    
    if (buffer.size() > 0)
    {
      for (unsigned int i=0; i< buffer.size(); i++)
      {
        if (visited[static_cast<unsigned int>(buffer[i])] > 0) { continue; }
        typename FSRC::mesh_type::Cell::array_type neighbors;
 
        int newtype = 0;
        imesh->get_neighbors(neighbors, buffer[i]);
        for (unsigned int p=0; p<neighbors.size(); p++)
        {
          if(visited[static_cast<unsigned int>(neighbors[p])] > 0)
          {
            if (newtype)
            {
              if (visited[static_cast<unsigned int>(neighbors[p])] != newtype)
              {
                pr->error("ConvertToTetVol: Algorithm cannot deal with topology of input field, field cannot by sorted into checker board type of ordering");
                return (false);
              }
            }
            else if(visited[static_cast<unsigned int>(neighbors[p])] == 0)
            {
              visited[static_cast<unsigned int>(neighbors[p])] = -1;
              newtype = visited[static_cast<unsigned int>(neighbors[p])];
            }
          }
          else
          {
            buffer.push_back(neighbors[p]);
          }
        }
        
        if (newtype == 0) newtype = 1;
        if (newtype == 1) newtype = 2; else newtype = 1;
        
        typename FSRC::mesh_type::Node::array_type hvnodes;
        imesh->get_nodes(hvnodes, buffer[i]);
        
        // In case mesh is weird an not logically numbered
        if (static_cast<unsigned int>(buffer[i]) >= elemmap.size()) elemmap.resize(static_cast<unsigned int>(buffer[i]));
        
        if (newtype == 1)
        {
          elemmap[static_cast<unsigned int>(buffer[i])] =
          omesh->add_tet((typename FDST::mesh_type::Node::index_type)(hvnodes[0]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[1]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[2]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[5]));

          omesh->add_tet((typename FDST::mesh_type::Node::index_type)(hvnodes[0]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[2]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[3]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[7]));

          omesh->add_tet((typename FDST::mesh_type::Node::index_type)(hvnodes[0]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[5]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[2]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[7]));

          omesh->add_tet((typename FDST::mesh_type::Node::index_type)(hvnodes[0]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[5]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[7]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[4]));

          omesh->add_tet((typename FDST::mesh_type::Node::index_type)(hvnodes[5]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[2]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[7]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[6]));
          visited[static_cast<unsigned int>(buffer[i])] = 1;
        }
        else
        {
          elemmap[static_cast<unsigned int>(buffer[i])] =
          omesh->add_tet((typename FDST::mesh_type::Node::index_type)(hvnodes[0]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[1]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[3]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[4]));

          omesh->add_tet((typename FDST::mesh_type::Node::index_type)(hvnodes[1]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[2]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[3]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[6]));

          omesh->add_tet((typename FDST::mesh_type::Node::index_type)(hvnodes[1]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[3]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[4]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[6]));

          omesh->add_tet((typename FDST::mesh_type::Node::index_type)(hvnodes[1]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[5]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[6]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[4]));

          omesh->add_tet((typename FDST::mesh_type::Node::index_type)(hvnodes[3]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[4]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[6]),
              (typename FDST::mesh_type::Node::index_type)(hvnodes[7]));
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
    pr->error("ConvertToTetVol: Could not create output field");
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
      ofield->set_value(val, static_cast<typename FDST::mesh_type::Elem::index_type>(idx));
      ofield->set_value(val, static_cast<typename FDST::mesh_type::Elem::index_type>(idx+1));
      ofield->set_value(val, static_cast<typename FDST::mesh_type::Elem::index_type>(idx+2));
      ofield->set_value(val, static_cast<typename FDST::mesh_type::Elem::index_type>(idx+3));
      ofield->set_value(val, static_cast<typename FDST::mesh_type::Elem::index_type>(idx+4));
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
bool ConvertHexVolToTetVolAlgoT<FSRC, FDST>::testinput(FieldHandle input)
{
  return (dynamic_cast<FSRC*>(input.get_rep()));
}


template <class FSRC, class FDST>
bool ConvertLatVolToTetVolAlgoT<FSRC, FDST>::ConvertToTetVol(ProgressReporter *pr, FieldHandle input, FieldHandle& output)
{

  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  { 
    pr->error("ConvertToTetVol: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh == 0)
  {
    pr->error("ConvertToTetVol: No mesh associated with input field");
    return (false);
  }

  typename FDST::mesh_handle_type omesh = scinew typename FDST::mesh_type();
  if (omesh == 0)
  {
    pr->error("ConvertToTetVol: Could not create output field");
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
  omesh->elem_reserve(static_cast<unsigned int>(numelems*5));

  typename FSRC::mesh_type::Elem::iterator bi, ei;
  typename FDST::mesh_type::Elem::iterator obi, oei;

  imesh->begin(bi); 
  imesh->end(ei);
  
  while (bi != ei)
  {
    typename FSRC::mesh_type::Node::array_type hvnodes;
    
    imesh->get_nodes(hvnodes, *bi);
    
    if (!(((*bi).i_ ^ (*bi).j_ ^ (*bi).k_)&1))
    {
      omesh->add_tet((typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[0]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[1]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[2]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[5]));

      omesh->add_tet((typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[0]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[2]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[3]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[7]));

      omesh->add_tet((typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[0]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[5]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[2]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[7]));

      omesh->add_tet((typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[0]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[5]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[7]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[4]));

      omesh->add_tet((typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[5]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[2]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[7]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[6]));
    }
    else
    {
      omesh->add_tet((typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[0]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[1]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[3]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[4]));

      omesh->add_tet((typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[1]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[2]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[3]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[6]));

      omesh->add_tet((typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[1]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[3]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[4]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[6]));

      omesh->add_tet((typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[1]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[5]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[6]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[4]));

      omesh->add_tet((typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[3]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[4]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[6]),
		      (typename FDST::mesh_type::Node::index_type)((unsigned int)hvnodes[7]));
    }
    ++bi;
  }
  
  FDST* ofield = scinew FDST(omesh);
  if (ofield == 0)
  {
    pr->error("ConvertToTetVol: Could not create output field");
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
      ofield->set_value(val,*obi); ++obi;
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

template <class FSRC, class FDST>
bool ConvertLatVolToTetVolAlgoT<FSRC, FDST>::testinput(FieldHandle input)
{
  return (dynamic_cast<FSRC*>(input.get_rep()));
}

  
} // end namespace ModelCreation

#endif 

