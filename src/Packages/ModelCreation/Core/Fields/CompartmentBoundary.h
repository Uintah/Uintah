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

#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <sci_hash_map.h>

namespace ModelCreation {

using namespace SCIRun;

class CompartmentBoundaryAlgo;

class CompartmentBoundaryAlgo : public DynamicAlgoBase
{
public:
  virtual bool CompartmentBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle DomainLink, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly);
  virtual bool testinput(FieldHandle input);

  static AlgoList<CompartmentBoundaryAlgo> precompiled_;
};


template <class FSRC, class FDST>
class CompartmentBoundaryVolumeAlgoT : public CompartmentBoundaryAlgo
{
public:
  virtual bool CompartmentBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle DomainLink, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly);
  virtual bool testinput(FieldHandle input);

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
class CompartmentBoundarySurfaceAlgoT : public CompartmentBoundaryAlgo
{
public:
  virtual bool CompartmentBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle DomainLink, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly);
  virtual bool testinput(FieldHandle input);
  
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
bool CompartmentBoundaryVolumeAlgoT<FSRC, FDST>::CompartmentBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle DomainLink, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly)
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
  typedef hash_multimap<unsigned int,pointtype> hash_map_type;
#else
  typedef multimap<unsigned int,pointtype> hash_map_type;
#endif
  
  hash_map_type node_map;
  
  imesh->synchronize(Mesh::NODES_E|Mesh::FACES_E|Mesh::CELLS_E|Mesh::FACE_NEIGHBORS_E);
  
  typename FSRC::mesh_type::Elem::iterator be, ee;
  typename FSRC::mesh_type::Elem::index_type nci, ci;
  typename FSRC::mesh_type::Face::array_type faces; 
  typename FSRC::mesh_type::Node::array_type inodes; 
  typename FDST::mesh_type::Node::array_type onodes; 
  typename FSRC::mesh_type::Node::index_type a;
  typename FSRC::value_type val1, val2, minval, maxval;

  inodes.clear();
  onodes.clear();

  minval = static_cast<typename FSRC::value_type>(minrange);
  maxval = static_cast<typename FSRC::value_type>(maxrange);
  
  Point point;

  imesh->begin(be); 
  imesh->end(ee);

  while (be != ee) 
  {
    ci = *be;
    imesh->get_faces(faces,ci);
    for (size_t p =0; p < faces.size(); p++)
    {
      bool neighborexist = false;
      bool includeface = false;
      
      neighborexist = imesh->get_neighbor(nci,ci,faces[p]);

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
        imesh->get_nodes(inodes,faces[p]);
        if (onodes.size() == 0) onodes.resize(inodes.size());
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

template <class FSRC, class FDST>
bool CompartmentBoundaryVolumeAlgoT<FSRC, FDST>::testinput(FieldHandle input)
{
  return(dynamic_cast<FSRC*>(input.get_rep())!=0);
}


template <class FSRC, class FDST>
bool CompartmentBoundarySurfaceAlgoT<FSRC, FDST>::CompartmentBoundary(ProgressReporter *pr, FieldHandle input, FieldHandle& output, MatrixHandle DomainLink, double minrange, double maxrange, bool userange, bool addouterboundary, bool innerboundaryonly)
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
  typedef hash_multimap<unsigned int,pointtype> hash_map_type;
#else
  typedef multimap<unsigned int,pointtype> hash_map_type;
#endif
  
  hash_map_type node_map;
  
  imesh->synchronize(Mesh::NODES_E|Mesh::EDGES_E|Mesh::FACES_E|Mesh::EDGE_NEIGHBORS_E);
  
  typename FSRC::mesh_type::Elem::iterator be, ee;
  typename FSRC::mesh_type::Elem::index_type nci, ci;
  typename FSRC::mesh_type::Edge::array_type edges; 
  typename FSRC::mesh_type::Node::array_type inodes; 
  typename FDST::mesh_type::Node::array_type onodes; 
  typename FSRC::mesh_type::Node::index_type a;
  typename FSRC::value_type val1, val2, minval, maxval;

  inodes.clear();
  onodes.clear();

  minval = static_cast<typename FSRC::value_type>(minrange);
  maxval = static_cast<typename FSRC::value_type>(maxrange);
  
  Point point;

  imesh->begin(be); 
  imesh->end(ee);

  while (be != ee) 
  {
    ci = *be;
    imesh->get_edges(edges,ci);
    for (size_t p =0; p < edges.size(); p++)
    {
      bool neighborexist = false;
      bool includeface = false;
      
      neighborexist = imesh->get_neighbor(nci,ci,edges[p]);

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
        imesh->get_nodes(inodes,edges[p]);
        if (onodes.size() == 0) onodes.resize(inodes.size());
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
  return (true);}

template <class FSRC, class FDST>
bool CompartmentBoundarySurfaceAlgoT<FSRC, FDST>::testinput(FieldHandle input)
{
  return(dynamic_cast<FSRC*>(input.get_rep())!=0);
}



} // end namespace ModelCreation

#endif 
