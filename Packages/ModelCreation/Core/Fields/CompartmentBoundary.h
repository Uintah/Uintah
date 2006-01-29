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


#ifndef MODELCREATION_CORE_FIELDS_COMPARTMENTBOUNDARY_H
#define MODELCREATION_CORE_FIELDS_COMPARTMENTBOUNDARY_H 1

#include <Packages/ModelCreation/Core/Util/DynamicAlgo.h>
#include <sci_hash_map.h>

namespace ModelCreation {

using namespace SCIRun;

class CompartmentBoundaryAlgo : public DynamicAlgoBase
{
public:
  virtual bool CompartmentBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output);
};


template <class FSRC, class FDST>
class CompartmentBoundaryVolumeAlgoT : public CompartmentBoundaryAlgo
{
public:
  virtual bool CompartmentBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output);
};


template <class FSRC, class FDST>
class CompartmentBoundarySurfaceAlgoT : public CompartmentBoundaryAlgo
{
public:
  virtual bool CompartmentBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output);
};


template <class FSRC, class FDST>
bool CompartmentBoundaryVolumeAlgoT<FSRC, FDST>::CompartmentBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output)
{
  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  {
    pr->error("CompartmentBoundary: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh == 0)
  {
    pr->error("CompartmentBoundary: No mesh associated with input field");
    return (false);
  }

  typename FDST::mesh_handle_type omesh = scinew typename FDST::mesh_type();
  if (omesh == 0)
  {
    pr->error("CompartmentBoundary: Could not create output field");
    return (false);
  }
  
  FDST *ofield = scinew FDST(omesh);
  output = dynamic_cast<Field*>(ofield);
  if (ofield == 0)
  {
    pr->error("CompartmentBoundary: Could not create output field");
    return (false);
  }
  
#ifdef HAVE_HASH_MAP
  typedef hash_map<unsigned int,unsigned int> hash_map_type;
#else
  typedef map<unsigned int,unsigned int> hash_map_type;
#endif
  
  hash_map_type node_map;
  
  imesh->synchronize(Mesh::NODES_E|Mesh::FACES_E|Mesh::CELLS_E|Mesh::FACE_NEIGHBORS_E);
  
  typename FSRC::mesh_type::Elem::iterator be, ee;
  typename FSRC::mesh_type::Elem::index_type nci, ci;
  typename FSRC::mesh_type::Face::array_type faces; 
  typename FSRC::mesh_type::Node::array_type inodes; 
  typename FDST::mesh_type::Node::array_type onodes; 
  typename FSRC::mesh_type::Node::index_type a;
  typename FSRC::value_type val1, val2;
  Point point;

  imesh->begin(be); 
  imesh->end(ee);

  while (be != ee) 
  {
    ci = *be;
    imesh->get_faces(faces,ci);
    for (size_t p =0; p < faces.size(); p++)
    {
      if (imesh->get_neighbor(nci,ci,faces[p]))
      {
        if (nci > ci)
        {
          ifield->value(val1,ci);
          ifield->value(val2,nci);
          if (!(val1 == val2))
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
                if (val1 > val2)
                  ofield->fdata().push_back(val1-val2);
                else
                  ofield->fdata().push_back(val2-val1);
              }
              else
              {
                onodes[q] = static_cast<typename FDST::mesh_type::Node::index_type>(node_map[a]);
              }
            }
            omesh->add_elem(onodes);
          }
        }
      }
    }
    ++be;
  }
  
  // copy property manager
	output->copy_properties(input.get_rep());
  return (true);
}


template <class FSRC, class FDST>
bool CompartmentBoundarySurfaceAlgoT<FSRC, FDST>::CompartmentBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output)
{
  FSRC *ifield = dynamic_cast<FSRC *>(input.get_rep());
  if (ifield == 0)
  {
    pr->error("CompartmentBoundary: Could not obtain input field");
    return (false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh();
  if (imesh == 0)
  {
    pr->error("CompartmentBoundary: No mesh associated with input field");
    return (false);
  }

  typename FDST::mesh_handle_type omesh = scinew typename FDST::mesh_type();
  if (omesh == 0)
  {
    pr->error("CompartmentBoundary: Could not create output field");
    return (false);
  }
  
  FDST *ofield = scinew FDST(omesh);
  output = dynamic_cast<Field*>(ofield);
  if (ofield == 0)
  {
    pr->error("CompartmentBoundary: Could not create output field");
    return (false);
  }
  
#ifdef HAVE_HASH_MAP
  typedef hash_map<unsigned int,unsigned int> hash_map_type;
#else
  typedef map<unsigned int,unsigned int> hash_map_type;
#endif
  
  hash_map_type node_map;
  
  imesh->synchronize(Mesh::NODES_E|Mesh::EDGES_E|Mesh::FACES_E|Mesh::EDGE_NEIGHBORS_E);
  
  typename FSRC::mesh_type::Elem::iterator be, ee;
  typename FSRC::mesh_type::Elem::index_type nci, ci;
  typename FSRC::mesh_type::Edge::index_type fi;
  typename FSRC::mesh_type::Edge::array_type faces; 
  typename FSRC::mesh_type::Node::array_type inodes; 
  typename FDST::mesh_type::Node::array_type onodes; 
  typename FSRC::mesh_type::Node::index_type a;
  typename FSRC::value_type val1, val2;
  Point point;

  imesh->begin(be); 
  imesh->end(ee);

  while (be != ee) 
  {
    ci = *be;
    imesh->get_edges(faces,ci);
    for (size_t p =0; p < faces.size(); p++)
    {
      if (imesh->get_neighbor(nci,ci,fi))
      {
        if (nci > ci)
        {
          ifield->get_value(val1,ci);
          ifield->get_value(val2,nci);
          if (!(val1 == val2))
          {
            imesh->get_nodes(inodes,fi);
            if (onodes.size() == 0) onodes.resize(inodes.size());
            for (int q=0; q< onodes.size(); q++)
            {
              a = inodes[q];
              hash_map_type::iterator it = node_map.find(static_cast<unsigned int>(a));
              if (it == node_map.end())
              {
                imesh->get_center(point,a);
                onodes[q] = omesh->add_point(point);
                if (val1 > val2)
                  ofield->fdata().pushback(val1-val2);
                else
                  ofield->fdata().pushback(val2-val1);
              }
              else
              {
                onodes[q] = static_cast<typename FDST::mesh_type::Node::index_type>(node_map[a]);
              }
            }
            omesh->add_elem(onodes);
          }
        }
      }
    }
  }
  
  // copy property manager
	output->copy_properties(input.get_rep());
  return (true);
}



} // end namespace ModelCreation

#endif 
