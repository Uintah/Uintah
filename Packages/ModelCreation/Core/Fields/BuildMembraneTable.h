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

#ifndef MODELCREATION_CORE_FIELDS_BUILDMEMBRANETABLE_H
#define MODELCREATION_CORE_FIELDS_BUILDMEMBRANETABLE_H 1

// The following include file will include all tools needed for doing 
// dynamic compilation and will include all the standard dataflow types
#include <Packages/ModelCreation/Core/Util/DynamicAlgo.h>

#include <sci_hash_map.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <algorithm>
#include <sci_hash_map.h>

namespace ModelCreation {

using namespace SCIRun;

class BuildMembraneTableAlgo : public DynamicAlgoBase
{
public:
  virtual bool BuildMembraneTable(ProgressReporter *pr, FieldHandle elementtypevol, FieldHandle membranesurf, MatrixHandle& membranetable);

};

template <class FVOL, class FSURF>
class BuildMembraneTableVolAlgoT : public BuildMembraneTableAlgo
{
public:
  virtual bool BuildMembraneTable(ProgressReporter *pr, FieldHandle elementtypevol, FieldHandle membranesurf, MatrixHandle& membranetable);

  class membraneparam_type {
    public:
      typename FVOL::mesh_type::Node::index_type node1;
      typename FVOL::mesh_type::Node::index_type node2;
      double                                     surface;
      
      bool operator<(membraneparam_type& p)
      {
        if (node1 < p.node1) return(true);
        if (node2 < p.node2) return(true);
        return (false);
      }
      
      bool operator==(membraneparam_type& p)
      {
        if ((node1 == p.node1)&&(node2 == p.node2)) return (true);
        return (false);
      }         
  };
  
};


template <class FVOL, class FSURF>
bool BuildMembraneTableVolAlgoT<FVOL,FSURF>::BuildMembraneTable(ProgressReporter *pr, FieldHandle elementtypevol, FieldHandle membranesurf, MatrixHandle& membranetable)
{

  FVOL *elementtypefield = dynamic_cast<FVOL *>(elementtypevol.get_rep());
  if (elementtypefield == 0)
  { 
    pr->error("BuildMembraneTable: Could not obtain element class description field");
    return (false);
  }

  typename FVOL::mesh_handle_type elementtypemesh = elementtypefield->get_typed_mesh();
  if (elementtypemesh == 0)
  {
    pr->error("BuildMembraneTable: No mesh associated with element class description field");
    return (false);
  }

  FSURF *membranefield = dynamic_cast<FSURF *>(membranesurf.get_rep());
  if (membranefield == 0)
  { 
    pr->error("BuildMembraneTable: Could not obtain model description field");
    return (false);
  }

  typename FSURF::mesh_handle_type membranemesh = membranefield->get_typed_mesh();
  if (membranemesh == 0)
  {
    pr->error("BuildMembraneTable: No mesh associated with model description field");
    return (false);
  }

  // VVV ALGORITHM STARTS HERE VVV

  elementtypemesh->synchronize(Mesh::NODE_NEIGHBORS_E|Mesh::FACES_E);

#ifdef HAVE_HASH_MAP
  typedef hash_multimap<unsigned int,typename FVOL::mesh_type::Node::index_type> nodemap_type;
#else
  typedef multimap<unsigned int,typename FVOL::mesh_type::Node::index_type> nodemap_type;
#endif

  int key;
  
  nodemap_type nodemap;
  Point point, point2;

  double xmin = 0.0;
  double xmax = 0.0;
  double ymin = 0.0;
  double ymax = 0.0;
  double zmin = 0.0;
  double zmax = 0.0;
    
  typename FVOL::mesh_type::Node::iterator it, it_end;
  elementtypemesh->begin(it);
  elementtypemesh->end(it_end);
  
  if (it != it_end)
  {
    elementtypemesh->get_center(point,*it);
    xmin = point.x(); xmax = point.x();
    ymin = point.y(); ymax = point.y();
    zmin = point.z(); zmax = point.z();
    ++it;
  }
    
  while (it != it_end)
  {
    elementtypemesh->get_center(point,*it);
    if (point.x() < xmin) xmin = point.x();
    if (point.x() > xmax) xmax = point.x();
    if (point.y() < ymin) ymin = point.y();
    if (point.y() > ymax) ymax = point.y();
    if (point.z() < zmin) zmin = point.z();
    if (point.z() > zmax) zmax = point.z();
    ++it;
  }

  double xmul = 0.0; if ((xmax-xmin) > 0.0 ) xmul = 250/(xmax-xmin);
  double ymul = 0.0; if ((ymax-ymin) > 0.0 ) ymul = 250/(ymax-ymin);
  double zmul = 0.0; if ((zmax-zmin) > 0.0 ) zmul = 250/(zmax-zmin);

  xmin -= (xmax-xmin)*0.01;
  ymin -= (ymax-ymin)*0.01;
  zmin -= (zmax-zmin)*0.01;
                  
  elementtypemesh->begin(it);
  elementtypemesh->end(it_end);
  
  while (it != it_end)
  {
    elementtypemesh->get_center(point,*it);
    
    key = static_cast<int>((point.x()-xmin)*xmul);
    key += (static_cast<int>((point.y()-ymin)*ymul))<<8;
    key += (static_cast<int>((point.z()-zmin)*zmul))<<16;  
  
    nodemap.insert(typename nodemap_type::value_type(key,*it));
  }

  typename FSURF::mesh_type::Elem::iterator eit, eit_end;
  typename FSURF::mesh_type::Node::array_type nodes;  

  membranemesh->begin(eit);
  membranemesh->end(eit_end);  

  membranemesh->get_nodes(nodes,*eit); 
  
  int nodespersurf =nodes.size();
  std::vector<Point> points(nodespersurf);

  bool foundsurf1;
  bool foundsurf2;

  std::pair<typename nodemap_type::iterator,typename nodemap_type::iterator> lit;
  typename FVOL::mesh_type::Node::index_type idx;
  typename FVOL::mesh_type::Elem::array_type elems;
  typename FVOL::mesh_type::Face::array_type faces;
  typename FSURF::value_type modeltype;
  typename FVOL::value_type classvalue1,classvalue2;
  double   surface;
  
  int k = 0;
  typename FSURF::mesh_type::Elems::size_type numelems;
  membranemesh->size(numelems);
  std::vector<membraneparam_type> membranetablelist(nodespersurf*numelems);
  
  if (points.size() > 1)
  {
    while (eit != eit_end)
    {
      foundsurf1 = false;
      foundsurf2 = false;
      membranemesh->get_nodes(nodes,*eit); 
      std::sort(nodes.begin(),nodes.end());
      for (size_t p =0; p < nodes.size(); p++) 
      {
        membranemesh->get_center(point2,nodes[p]);
        points[p] = point2;
      }
      
      membranemesh->get_center(point,nodes[0]);
      
      // NEED TO THIS MORE PROPERLY USING BASIS FUNCTIONS
      surface = (membranemesh->get_size(*eit))/(nodes.size());
      
      key = static_cast<int>((point.x()-xmin)*xmul);
      key += (static_cast<int>((point.y()-ymin)*ymul))<<8;
      key += (static_cast<int>((point.z()-zmin)*zmul))<<16;
      
      lit = nodemap.equal_range(key);

      while ((lit.first != lit.second)&&(foundsurf1 == false)&&(foundsurf2 == false))
      {
        idx = (*(lit.first)).second;
        elementtypemesh->get_center(point2,idx);
 
        if (point2 == point)
        {
          elementtypemesh->get_elems(elems,idx);
          for (size_t p = 0; p < elems.size(); p++)
          {
            if (elementtypemesh->get_faces(faces,elems[p]));
            {
              std::sort(faces.begin(),faces.end());
              bool isequal = true;
              for (size_t q = 0; q < faces.size(); q++)
              {
                elementtypemesh->get_center(point2,faces[q]);
                if (point2 != points[q]) { isequal = false; break; }
              }
              
              if (isequal)
              {
                if (foundsurf1 == false)
                {
                  elementtypefield->value(classvalue1,idx);

                  for (size_t q = 0; q < faces.size(); q++)
                  {           
                    membranetablelist[k+q].node1 = faces[q];
                    membranetablelist[k+q].surface = surface;
                  }
                  foundsurf1 = true;
                }
                else if (foundsurf2 == false)
                {
                  elementtypefield->value(classvalue2,idx);
                  for (size_t q = 0; q < faces.size(); q++)
                  { 
                    if (classvalue1 < classvalue2)
                    {
                      membranetablelist[k+q].node2 = faces[q];
                    }
                    else
                    {
                      membranetablelist[k+q].node2 = membranetablelist[k+q].node1;
                      membranetablelist[k+q].node1 = faces[q];
                    }
                    membranetablelist[k+q].surface = surface;
                  }
                  foundsurf2 = true;
                  k += faces.size();
                }
              }
            }
          }
        }
      }
        
      if ((foundsurf1 == false)||(foundsurf2 == false))
      {
        pr->error("BuildMembraneTable: Not every surface/curve in the membrane model can be found in the element type mesh");
        return (false);
      }
    }
  }
  
  std::sort(membranetablelist.begin(),membranetablelist.end());
  
  if (membranetablelist.size() > 0)
  {
    size_t tablesize = 1;
    size_t q = 0;
    for (size_t p=1; p < membranetablelist.size(); p++)
    {
      if (membranetablelist[p] == membranetablelist[q])
      {
        membranetablelist[q].surface += membranetablelist[p].surface;
        membranetablelist[p].surface = 0.0;
      }
      else if (membranetablelist[p].surface != 0.0)
      {
        q = p;
        tablesize++;
      }
    }
    
    membranetable = dynamic_cast<Matrix *>(scinew DenseMatrix(tablesize,3));
    if (membranetable.get_rep() == 0)
    {
      pr->error("BuildMembraneTable: Could not allocate Membrane table");
      return (false);  
    }

    double* dataptr = membranetable->get_data_pointer();
    q = 0;
    for (size_t p=1; p < membranetablelist.size(); p++)
    {
      if (membranetablelist[p].surface != 0.0)
      {
        dataptr[q++] = membranetablelist[p].node1;
        dataptr[q++] = membranetablelist[p].node2;
        dataptr[q++] = membranetablelist[p].surface;        
      }
    }
  }
  else
  {
    pr->warning("BuildMembraneTable: The Membrane geometry does not correspond to any of the element faces/edges of the element type field");
    return (true);    
  }

  // Success:
  return (true);
}

template <class FVOL, class FSURF>
class BuildMembraneTableSurfAlgoT : public BuildMembraneTableAlgo
{
public:
  virtual bool BuildMembraneTable(ProgressReporter *pr, FieldHandle elementtypevol, FieldHandle membranesurf, MatrixHandle& membranetable);

  class membraneparam_type {
    public:
      typename FVOL::mesh_type::Node::index_type node1;
      typename FVOL::mesh_type::Node::index_type node2;
      double                                     surface;
      
      bool operator<(membraneparam_type& p)
      {
        if (node1 < p.node1) return(true);
        if (node2 < p.node2) return(true);
        return (false);
      }
      
      bool operator==(membraneparam_type& p)
      {
        if ((node1 == p.node1)&&(node2 == p.node2)) return (true);
        return (false);
      }         
  };
  
};


template <class FVOL, class FSURF>
bool BuildMembraneTableSurfAlgoT<FVOL,FSURF>::BuildMembraneTable(ProgressReporter *pr, FieldHandle elementtypevol, FieldHandle membranesurf, MatrixHandle& membranetable)
{

  FVOL *elementtypefield = dynamic_cast<FVOL *>(elementtypevol.get_rep());
  if (elementtypefield == 0)
  { 
    pr->error("BuildMembraneTable: Could not obtain element class description field");
    return (false);
  }

  typename FVOL::mesh_handle_type elementtypemesh = elementtypefield->get_typed_mesh();
  if (elementtypemesh == 0)
  {
    pr->error("BuildMembraneTable: No mesh associated with element class description field");
    return (false);
  }

  FSURF *membranefield = dynamic_cast<FSURF *>(membranesurf.get_rep());
  if (membranefield == 0)
  { 
    pr->error("BuildMembraneTable: Could not obtain model description field");
    return (false);
  }

  typename FSURF::mesh_handle_type membranemesh = membranefield->get_typed_mesh();
  if (membranemesh == 0)
  {
    pr->error("BuildMembraneTable: No mesh associated with model description field");
    return (false);
  }

  // VVV ALGORITHM STARTS HERE VVV

  elementtypemesh->synchronize(Mesh::NODE_NEIGHBORS_E|Mesh::EDGES_E);

#ifdef HAVE_HASH_MAP
  typedef hash_multimap<unsigned int,typename FVOL::mesh_type::Node::index_type> nodemap_type;
#else
  typedef multimap<unsigned int,typename FVOL::mesh_type::Node::index_type> nodemap_type;
#endif

  nodemap_type nodemap;
  Point point, point2;
  int key;

  double xmin = 0.0;
  double xmax = 0.0;
  double ymin = 0.0;
  double ymax = 0.0;
  double zmin = 0.0;
  double zmax = 0.0;
    
  typename FVOL::mesh_type::Node::iterator it, it_end;
  elementtypemesh->begin(it);
  elementtypemesh->end(it_end);
  
  if (it != it_end)
  {
    elementtypemesh->get_center(point,*it);
    xmin = point.x(); xmax = point.x();
    ymin = point.y(); ymax = point.y();
    zmin = point.z(); zmax = point.z();
    ++it;
  }
    
  while (it != it_end)
  {
    elementtypemesh->get_center(point,*it);
    if (point.x() < xmin) xmin = point.x();
    if (point.x() > xmax) xmax = point.x();
    if (point.y() < ymin) ymin = point.y();
    if (point.y() > ymax) ymax = point.y();
    if (point.z() < zmin) zmin = point.z();
    if (point.z() > zmax) zmax = point.z();
    ++it;
  }

  double xmul = 0.0; if ((xmax-xmin) > 0.0 ) xmul = 250/(xmax-xmin);
  double ymul = 0.0; if ((ymax-ymin) > 0.0 ) ymul = 250/(ymax-ymin);
  double zmul = 0.0; if ((zmax-zmin) > 0.0 ) zmul = 250/(zmax-zmin);

  xmin -= (xmax-xmin)*0.01;
  ymin -= (ymax-ymin)*0.01;
  zmin -= (zmax-zmin)*0.01;
                  
  elementtypemesh->begin(it);
  elementtypemesh->end(it_end);
  
  while (it != it_end)
  {
    elementtypemesh->get_center(point,*it);
    
    key = static_cast<int>((point.x()-xmin)*xmul);
    key += (static_cast<int>((point.y()-ymin)*ymul))<<8;
    key += (static_cast<int>((point.z()-zmin)*zmul))<<16;  
  
    nodemap.insert(typename nodemap_type::value_type(key,*it));
  }

  typename FSURF::mesh_type::Elem::iterator eit, eit_end;
  typename FSURF::mesh_type::Node::array_type nodes;  

  membranemesh->begin(eit);
  membranemesh->end(eit_end);  

  membranemesh->get_nodes(nodes,*eit); 
  
  int nodespersurf =nodes.size();
  std::vector<Point> points(nodespersurf);

  bool foundsurf1;
  bool foundsurf2;

  std::pair<typename nodemap_type::iterator,typename nodemap_type::iterator> lit;
  typename FVOL::mesh_type::Node::index_type idx;
  typename FVOL::mesh_type::Elem::array_type elems;
  typename FVOL::mesh_type::Edge::array_type edges;
  typename FSURF::value_type modeltype;
  typename FVOL::value_type classvalue1,classvalue2;
  double   surface;
  
  int k = 0;
  typename FSURF::mesh_type::Elems::size_type numelems;
  membranemesh->size(numelems);
  std::vector<membraneparam_type> membranetablelist(nodespersurf*numelems);
  
  if (points.size() > 1)
  {
    while (eit != eit_end)
    {
      foundsurf1 = false;
      foundsurf2 = false;
      membranemesh->get_nodes(nodes,*eit); 
      std::sort(nodes.begin(),nodes.end());
      for (size_t p =0; p < nodes.size(); p++) 
      {
        membranemesh->get_center(point2,nodes[p]);
        points[p] = point2;
      }
      
      membranemesh->get_center(point,nodes[0]);
      
      // NEED TO THIS MORE PROPERLY USING BASIS FUNCTIONS
      surface = (membranemesh->get_size(*eit))/(nodes.size());
      
      key = static_cast<int>((point.x()-xmin)*xmul);
      key += (static_cast<int>((point.y()-ymin)*ymul))<<8;
      key += (static_cast<int>((point.z()-zmin)*zmul))<<16;
      
      lit = nodemap.equal_range(key);

      while ((lit.first != lit.second)&&(foundsurf1 == false)&&(foundsurf2 == false))
      {
        idx = (*(lit.first)).second;
        elementtypemesh->get_center(point2,idx);
 
        if (point2 == point)
        {
          elementtypemesh->get_elems(elems,idx);
          for (size_t p = 0; p < elems.size(); p++)
          {
            if (elementtypemesh->get_edges(edges,elems[p]));
            {
              std::sort(edges.begin(),edges.end());
              bool isequal = true;
              for (size_t q = 0; q < edges.size(); q++)
              {
                elementtypemesh->get_center(point2,edges[q]);
                if (point2 != points[q]) { isequal = false; break; }
              }
              if (isequal)
              {
                if (foundsurf1 == false)
                {
                  elementtypefield->value(classvalue1,idx);

                  for (size_t q = 0; q < edges.size(); q++)
                  {           
                    membranetablelist[k+q].node1 = edges[q];
                    membranetablelist[k+q].surface = surface;
                  }
                  foundsurf1 = true;
                }
                else if (foundsurf2 == false)
                {
                  elementtypefield->value(classvalue2,idx);
                  for (size_t q = 0; q < edges.size(); q++)
                  { 
                    if (classvalue1 < classvalue2)
                    {
                      membranetablelist[k+q].node2 = edges[q];
                    }
                    else
                    {
                      membranetablelist[k+q].node2 = membranetablelist[k+q].node1;
                      membranetablelist[k+q].node1 = edges[q];
                    }
                    membranetablelist[k+q].surface = surface;
                  }
                  foundsurf2 = true;
                  k += edges.size();
                }
              }
            }
          }
        }
      }
        
      if ((foundsurf1 == false)||(foundsurf2 == false))
      {
        pr->error("BuildMembraneTable: Not every surface/curve in the membrane model can be found in the element type mesh");
        return (false);
      }
    }
  }
  
  std::sort(membranetablelist.begin(),membranetablelist.end());
  
  if (membranetablelist.size() > 0)
  {
    size_t tablesize = 1;
    size_t q = 0;
    for (size_t p=1; p < membranetablelist.size(); p++)
    {
      if (membranetablelist[p] == membranetablelist[q])
      {
        membranetablelist[q].surface += membranetablelist[p].surface;
        membranetablelist[p].surface = 0.0;
      }
      else if (membranetablelist[p].surface != 0.0)
      {
        q = p;
        tablesize++;
      }
    }
    
    membranetable = dynamic_cast<Matrix *>(scinew DenseMatrix(tablesize,3));
    if (membranetable.get_rep() == 0)
    {
      pr->error("BuildMembraneTable: Could not allocate Membrane table");
      return (false);  
    }

    double* dataptr = membranetable->get_data_pointer();
    q = 0;
    for (size_t p=1; p < membranetablelist.size(); p++)
    {
      if (membranetablelist[p].surface != 0.0)
      {
        dataptr[q++] = membranetablelist[p].node1;
        dataptr[q++] = membranetablelist[p].node2;
        dataptr[q++] = membranetablelist[p].surface;        
      }
    }
  }
  else
  {
    pr->warning("BuildMembraneTable: The Membrane geometry does not correspond to any of the element faces/edges of the element type field");
    return (true);    
  }

  // Success:
  return (true);
}



} // end namespace ModelCreation

#endif 

