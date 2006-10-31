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

#ifndef CARDIOWAVE_CORE_FIELDS_BUILDSTIMULUSTABLE_H
#define CARDIOWAVE_CORE_FIELDS_BUILDSTIMULUSTABLE_H 1

// The following include file will include all tools needed for doing 
// dynamic compilation and will include all the standard dataflow types
#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <Core/Basis/Bases.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/HexVolMesh.h>


#include <sci_hash_map.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <algorithm>
#include <sci_hash_map.h>

namespace CardioWave {

using namespace SCIRun;

class stimulusparam_type {
  public:
    unsigned int node;
    double       weight;
};


typedef std::vector<stimulusparam_type> StimulusTable;
typedef std::vector<stimulusparam_type> ReferenceTable;

inline bool operator==(const stimulusparam_type& p1,const stimulusparam_type& p2)
{
  if (p1.node == p2.node) return (true);
  return (false);
}    

inline bool operator<(const stimulusparam_type& p1, const stimulusparam_type& p2)
{
  if (p1.node < p2.node) return(true); 
  return (false);
}


class BuildStimulusTableAlgo : public DynamicAlgoBase
{
public:
  virtual bool BuildStimulusTable(ProgressReporter *pr, FieldHandle ElementType, FieldHandle Stimulus, MatrixHandle CompToGeom, double domaintype, bool selectbynode, StimulusTable& stimulustablelist);

protected:
  template<class MESH>
  double weight_factor(MESH *mesh, typename MESH::Cell::index_type elemidx, typename MESH::Node::index_type nodeidx)
  {
     return (static_cast<double>(mesh->get_size(elemidx)/mesh->get_basis().number_of_vertices()));
  }

  template<class MESH>
  double weight_factor(MESH *mesh, typename MESH::Face::index_type elemidx, typename MESH::Node::index_type nodeidx)
  {
     return (static_cast<double>(mesh->get_size(elemidx)/mesh->get_basis().vertices_of_face()));
  }


  template<class MESH>
  double weight_factor(MESH *mesh, typename MESH::Edge::index_type elemidx, typename MESH::Node::index_type nodeidx)
  {
     return (static_cast<double>(mesh->get_size(elemidx)/2));
  }

  double weight_factor(QuadSurfMesh<QuadBilinearLgn<Point> >* mesh,
    QuadSurfMesh<QuadBilinearLgn<Point> >::Elem::index_type elemidx,
    QuadSurfMesh<QuadBilinearLgn<Point> >::Node::index_type nodeidx)
  {
     return (static_cast<double>(mesh->get_size(elemidx)/4));
  }

  double weight_factor(HexVolMesh<HexTrilinearLgn<Point> >* mesh,
    HexVolMesh<HexTrilinearLgn<Point> >::Elem::index_type elemidx,
    HexVolMesh<HexTrilinearLgn<Point> >::Node::index_type nodeidx)
  {
     return (static_cast<double>(mesh->get_size(elemidx)/8));
  }

  double weight_factor(HexVolMesh<HexTrilinearLgn<Point> >* mesh,
    HexVolMesh<HexTrilinearLgn<Point> >::Face::index_type elemidx,
    HexVolMesh<HexTrilinearLgn<Point> >::Node::index_type nodeidx)
  {
     return (static_cast<double>(mesh->get_size(elemidx)/4));
  }

};

template <class FNODE, class FIELD>
class BuildStimulusTableCellAlgoT : public BuildStimulusTableAlgo
{
public:
  virtual bool BuildStimulusTable(ProgressReporter *pr, FieldHandle ElementType, FieldHandle Stimulus, MatrixHandle CompToGeom, double domaintype, bool selectbynode, StimulusTable& stimulustablelist);  
};

template <class FNODE, class FIELD>
class BuildStimulusTableFaceAlgoT : public BuildStimulusTableAlgo
{
public:
  virtual bool BuildStimulusTable(ProgressReporter *pr, FieldHandle ElementType, FieldHandle Stimulus, MatrixHandle CompToGeom, double domaintype, bool selectbynode, StimulusTable& stimulustablelist);  
};

template <class FNODE, class FIELD>
class BuildStimulusTableEdgeAlgoT : public BuildStimulusTableAlgo
{
public:
  virtual bool BuildStimulusTable(ProgressReporter *pr, FieldHandle ElementType, FieldHandle Stimulus, MatrixHandle CompToGeom, double domaintype, bool selectbynode, StimulusTable& stimulustablelist);  
};

template <class FNODE, class FIELD>
class BuildStimulusTableNodeAlgoT : public BuildStimulusTableAlgo
{
public:
  virtual bool BuildStimulusTable(ProgressReporter *pr, FieldHandle ElementType, FieldHandle Stimulus, MatrixHandle CompToGeom, double domaintype, bool selectbynode, StimulusTable& stimulustablelist);  
};

template <class FNODE, class FIELD>
bool BuildStimulusTableCellAlgoT<FNODE,FIELD>::BuildStimulusTable(ProgressReporter *pr, FieldHandle ElementType, FieldHandle Stimulus, MatrixHandle CompToGeom, double domaintype, bool selectbynode, StimulusTable& stimulustablelist)
{
  // Check whether we have all pointers
  FNODE *elementtypefield = dynamic_cast<FNODE*>(ElementType.get_rep());
  FIELD *stimfield = dynamic_cast<FIELD*>(Stimulus.get_rep());

  if (stimfield == 0)
  {
    pr->error("BuildStimulusTable: The Stimulus field is empty");
    return (false);
  }

  if (elementtypefield == 0)
  {
    pr->error("BuildStimulusTable: The domaintype field is empty");
    return (false);
  }

  typename FNODE::mesh_handle_type elementtypemesh = elementtypefield->get_typed_mesh();
  if (elementtypemesh.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The domaintype mesh is empty");
    return (false);
  }

  typename FIELD::mesh_handle_type stimmesh = stimfield->get_typed_mesh();
  if (stimmesh.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The stimmesh is empty");
    return (false);
  }

  // We have pointers and handles

  // Setup algorithm to renumber nodes as the domain has linked nodes somewhere
  typename FNODE::mesh_type::Node::size_type nnodes;
  bool isgeomtocomp = false;
  int  *geomtocomprr = 0;
  int  *geomtocompcc = 0;

  // If it is there we setup the tables
  if (CompToGeom.get_rep())
  {
    // Sanity check..
    elementtypemesh->size(nnodes);
    if (nnodes != CompToGeom->nrows())
    {
      pr->error("BuildStimulusTable: The number of rows in CompToGeom is not of the right size");
      return (false);    
    }
    
    SparseRowMatrix* mat = dynamic_cast<SparseRowMatrix *>(CompToGeom.get_rep());
    if (mat == 0)
    {
      pr->error("BuildStimulusTable: CompToGeom is not a sparse matrix");
      return (false);    
    }
  
    geomtocomprr = mat->rows;
    geomtocompcc = mat->columns;
    
    // WE only support 1-on-1 mappings here
    for (int r=0; r<nnodes+1;r++)
    {
      if (geomtocomprr[r] != r)
      {
        pr->error("BuildStimulusTable: CompToGeom is mapping a geometrical node to multiple computational nodes");
        return (false);     
      }
    }
    isgeomtocomp = true;
  }


  typename FNODE::mesh_type::Node::iterator it, it_end;
  typename FNODE::mesh_type::Node::size_type sz;
  typename FNODE::value_type val, dval;
  typename FNODE::mesh_type::Elem::array_type elems;
  typename FIELD::mesh_type::Elem::index_type ci;
  
  Point point;

  elementtypemesh->size(sz);
  elementtypemesh->begin(it);
  elementtypemesh->end(it_end);  
  
  std::vector<bool> indomain(sz);
  dval = static_cast<typename FNODE::value_type>(domaintype);
  
  stimmesh->synchronize(Mesh::LOCATE_E|Mesh::CELLS_E);


  // First make a table of which nodes are actually in the domain.

  while (it != it_end)
  {
    indomain[(static_cast<unsigned int>(*it))] = false;
    
    elementtypemesh->get_center(point,*it);
    if (stimmesh->locate(ci,point))
    {
      elementtypemesh->get_elems(elems,*it);
      if (elems.size() > 0)
      {
        val = elementtypefield->value(elems[0]);
        if (val == dval)
        {
          indomain[(static_cast<unsigned int>(*it))] = true;        
        }
      }
    }
    ++it;
  }

  typename FNODE::mesh_type::Cell::iterator cit, cit_end;
  typename FNODE::mesh_type::Node::array_type nodes;  
    
  elementtypemesh->begin(cit);
  elementtypemesh->end(cit_end);  
 
  stimulustablelist.clear();

  // now iterate over each element

  while (cit != cit_end)
  {
    elementtypemesh->get_nodes(nodes,*cit);
    size_t p = 0;
    for (; p < nodes.size(); p++)
    {
      if (indomain[static_cast<unsigned int>(nodes[p])] == false) break;
    }
    if (p == nodes.size())
    {
      elementtypemesh->get_center(point,*cit);
      if (stimmesh->locate(ci,point))
      {
      
        for (p = 0; p < nodes.size(); p++)
        {
          stimulusparam_type stimitem;
          stimitem.node = static_cast<unsigned int>(nodes[p]);
          stimitem.weight = weight_factor(elementtypemesh.get_rep(),*cit, nodes[p]);     
          stimulustablelist.push_back(stimitem);
        }
      }
    }
    ++cit;    
  }
  
  
  if (isgeomtocomp)
  {
    for (size_t p=0; p < stimulustablelist.size(); p++) 
    {
      stimulustablelist[p].node = geomtocompcc[stimulustablelist[p].node];
    }
  }

  if (stimulustablelist.size() > 0)       
  {  
    std::sort(stimulustablelist.begin(),stimulustablelist.end());

    size_t k = 0;
    for (size_t p=1; p < stimulustablelist.size(); p++) 
    {
      if (stimulustablelist[p].node == stimulustablelist[k].node)
      {
      stimulustablelist[k].weight += stimulustablelist[p].weight;
      stimulustablelist[p].weight = 0.0;
      }
      else
      {
        k++;
        stimulustablelist[k] = stimulustablelist[p];
      }
    }
       
    stimulustablelist.resize(k+1);
  }

  
  // Success:
  return (true);
}


template <class FNODE, class FIELD>
bool BuildStimulusTableFaceAlgoT<FNODE,FIELD>::BuildStimulusTable(ProgressReporter *pr, FieldHandle ElementType, FieldHandle Stimulus, MatrixHandle CompToGeom, double domaintype, bool selectbynode, StimulusTable& stimulustablelist)
{
  FNODE *elementtypefield = dynamic_cast<FNODE*>(ElementType.get_rep());
  FIELD *stimfield = dynamic_cast<FIELD*>(Stimulus.get_rep());

  if (stimfield == 0)
  {
    pr->error("BuildStimulusTable: The Stimulus field is empty");
    return (false);
  }

  if (elementtypefield == 0)
  {
    pr->error("BuildStimulusTable: The domaintype field is empty");
    return (false);
  }

  typename FNODE::mesh_handle_type elementtypemesh = elementtypefield->get_typed_mesh();
  if (elementtypemesh.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The domaintype mesh is empty");
    return (false);
  }

  typename FIELD::mesh_handle_type stimmesh = stimfield->get_typed_mesh();
  if (stimmesh.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The stimmesh is empty");
    return (false);
  }


  typename FNODE::mesh_type::Node::size_type nnodes;
  bool isgeomtocomp = false;
  int  *geomtocomprr = 0;
  int  *geomtocompcc = 0;

  if (CompToGeom.get_rep())
  {
    elementtypemesh->size(nnodes);
    if (nnodes != CompToGeom->nrows())
    {
      pr->error("BuildStimulusTable: The number of rows in CompToGeom is not of the right size");
      return (false);    
    }
    
    SparseRowMatrix* mat = dynamic_cast<SparseRowMatrix *>(CompToGeom.get_rep());
    if (mat == 0)
    {
      pr->error("BuildStimulusTable: CompToGeom is not a sparse matrix");
      return (false);    
    }
  
    geomtocomprr = mat->rows;
    geomtocompcc = mat->columns;
    
    for (int r=0; r<nnodes+1;r++)
    {
      if (geomtocomprr[r] != r)
      {
        pr->error("BuildStimulusTable: CompToGeom is mapping a geometrical node to multiple computational nodes");
        return (false);     
      }
    }
    isgeomtocomp = true;
  }

  typename FNODE::mesh_type::Node::iterator it, it_end;
  typename FNODE::mesh_type::Node::size_type sz;
  typename FNODE::mesh_type::Elem::array_type elems;
  typename FNODE::value_type val, dval;
  typename FIELD::mesh_type::Elem::index_type ci;
  Point point;

  elementtypemesh->size(sz);
  elementtypemesh->begin(it);
  elementtypemesh->end(it_end);  

  std::vector<bool> indomain(sz);

  dval = static_cast<typename FNODE::value_type>(domaintype);
  
  stimmesh->synchronize(Mesh::LOCATE_E|Mesh::FACES_E);

  while (it != it_end)
  {
    indomain[(static_cast<unsigned int>(*it))] = false;

    elementtypemesh->get_center(point,*it);
    if (stimmesh->locate(ci,point))
    {
      elementtypemesh->get_elems(elems,*it);
      if (elems.size() > 0)
      {
        val = elementtypefield->value(elems[0]);
        if (val == dval)
        {
          indomain[(static_cast<unsigned int>(*it))] = true;        
        }
      }
    }
    ++it;
  }

  typename FNODE::mesh_type::Face::iterator cit, cit_end;
  typename FNODE::mesh_type::Node::array_type nodes;  
    
  elementtypemesh->begin(cit);
  elementtypemesh->end(cit_end);  
 
  stimulustablelist.clear();
 
  while (cit != cit_end)
  {
    elementtypemesh->get_nodes(nodes,*cit);
    size_t p = 0;
    for (; p < nodes.size(); p++)
    {
      if (indomain[static_cast<unsigned int>(nodes[p])] == false) break;
    }
    if (p == nodes.size())
    {
      elementtypemesh->get_center(point,*cit);
      if (stimmesh->locate(ci,point))
      {
        for (p = 0; p < nodes.size(); p++)
        {
          stimulusparam_type stimitem;
          stimitem.node = static_cast<unsigned int>(nodes[p]);
          stimitem.weight = weight_factor(elementtypemesh.get_rep(),*cit, nodes[p]);     
          stimulustablelist.push_back(stimitem);
        }
      }
    }
    ++cit;
  }


  if (isgeomtocomp)
  {
    for (size_t p=0; p < stimulustablelist.size(); p++) 
    {
      stimulustablelist[p].node = geomtocompcc[stimulustablelist[p].node];
    }
  }

  std::sort(stimulustablelist.begin(),stimulustablelist.end());

  size_t k = 0;
  for (size_t p=1; p < stimulustablelist.size(); p++) 
  {
    if (stimulustablelist[p].node == stimulustablelist[k].node)
    {
      stimulustablelist[k].weight += stimulustablelist[p].weight;
      stimulustablelist[p].weight = 0.0;
    }
    else
    {
      k++;
      stimulustablelist[k] = stimulustablelist[p];
    }
  }
 
  if (stimulustablelist.size() > 0)       
  {
    stimulustablelist.resize(k+1);
  }
                 
                                             
  // Success:
  return (true);
}



template <class FNODE, class FIELD>
bool BuildStimulusTableEdgeAlgoT<FNODE,FIELD>::BuildStimulusTable(ProgressReporter *pr, FieldHandle ElementType, FieldHandle Stimulus, MatrixHandle CompToGeom, double domaintype, bool selectbynode, StimulusTable& stimulustablelist)
{
  FNODE *elementtypefield = dynamic_cast<FNODE*>(ElementType.get_rep());
  FIELD *stimfield = dynamic_cast<FIELD*>(Stimulus.get_rep());

  if (stimfield == 0)
  {
    pr->error("BuildStimulusTable: The Stimulus field is empty");
    return (false);
  }

  if (elementtypefield == 0)
  {
    pr->error("BuildStimulusTable: The domaintype field is empty");
    return (false);
  }

  typename FNODE::mesh_handle_type elementtypemesh = elementtypefield->get_typed_mesh();
  if (elementtypemesh.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The domaintype mesh is empty");
    return (false);
  }

  typename FIELD::mesh_handle_type stimmesh = stimfield->get_typed_mesh();
  if (stimmesh.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The stimmesh is empty");
    return (false);
  }

  typename FNODE::mesh_type::Node::size_type nnodes;
  bool isgeomtocomp = false;
  int  *geomtocomprr = 0;
  int  *geomtocompcc = 0;

  if (CompToGeom.get_rep())
  {
    elementtypemesh->size(nnodes);
    if (nnodes != CompToGeom->nrows())
    {
      pr->error("BuildStimulusTable: The number of rows in CompToGeom is not of the right size");
      return (false);    
    }
    
    SparseRowMatrix* mat = dynamic_cast<SparseRowMatrix *>(CompToGeom.get_rep());
    if (mat == 0)
    {
      pr->error("BuildStimulusTable: CompToGeom is not a sparse matrix");
      return (false);    
    }
  
    geomtocomprr = mat->rows;
    geomtocompcc = mat->columns;
    
    for (int r=0; r<nnodes+1;r++)
    {
      if (geomtocomprr[r] != r)
      {
        pr->error("BuildStimulusTable: CompToGeom is mapping a geometrical node to multiple computational nodes");
        return (false);     
      }
    }
    isgeomtocomp = true;
  }

  typename FNODE::mesh_type::Eleme::array_type elems;
  typename FNODE::mesh_type::Node::iterator it, it_end;
  typename FNODE::mesh_type::Node::size_type sz;
  typename FIELD::mesh_type::Elem::index_type ci;
  typename FNODE::value_type val, dval;
  Point point;

  elementtypemesh->size(sz);
  elementtypemesh->begin(it);
  elementtypemesh->end(it_end);  

  std::vector<bool> indomain(sz);

  dval = static_cast<typename FNODE::value_type>(domaintype);
  
  stimmesh->synchronize(Mesh::LOCATE_E|Mesh::EDGES_E);

  while (it != it_end)
  {
    indomain[(static_cast<unsigned int>(*it))] = false;
    
    elementtypemesh->get_center(point,*it);
    if (stimmesh->locate(ci,point))
    {
      elementtypemesh->get_elems(elems,*it);
      if (elems.size() > 0)
      {
        val = elementtypefield->value(elems[0]);
        if (val == dval)
        {
          indomain[static_cast<unsigned int>(*it)] = true;        
        }
      }
    }
    ++it;
  }

  typename FNODE::mesh_type::Edge::iterator cit, cit_end;
  typename FNODE::mesh_type::Node::array_type nodes;  
    
  elementtypemesh->begin(cit);
  elementtypemesh->end(cit_end);  
 
  stimulustablelist.clear();
 
  while (cit != cit_end)
  {
    elementtypemesh->get_nodes(nodes,*cit);
    size_t p = 0;
    for (; p < nodes.size(); p++)
    {
      if (indomain[static_cast<unsigned int>(nodes[p])] == false) break;
    }
    if (p == nodes.size())
    {
      elementtypemesh->get_center(point,*cit);
      if (stimmesh->locate(ci,point))
      {    
        for (p = 0; p < nodes.size(); p++)
        {
          stimulusparam_type stimitem;
          stimitem.node = static_cast<unsigned int>(nodes[p]);
          stimitem.weight = weight_factor(elementtypemesh.get_rep(),*cit, nodes[p]);     
          stimulustablelist.push_back(stimitem);
        }
      }
    }
    ++cit;
  }

  if (isgeomtocomp)
  {
    for (size_t p=1; p < stimulustablelist.size(); p++) 
    {
      stimulustablelist[p].node = geomtocompcc[stimulustablelist[p].node];
    }
  }

  std::sort(stimulustablelist.begin(),stimulustablelist.end());

  size_t k = 0;
  for (size_t p=0; p < stimulustablelist.size(); p++) 
  {
    if (stimulustablelist[p].node == stimulustablelist[k].node)
    {
      stimulustablelist[k].weight += stimulustablelist[p].weight;
      stimulustablelist[p].weight = 0.0;
    }
    else
    {
      k++;
      stimulustablelist[k] = stimulustablelist[p];
    }
  }
     
  if (stimulustablelist.size() > 0)       
  {
    stimulustablelist.resize(k+1);
  }

                 
  // Success:
  return (true);
}


template <class FNODE, class FIELD>
bool BuildStimulusTableNodeAlgoT<FNODE,FIELD>::BuildStimulusTable(ProgressReporter *pr, FieldHandle ElementType, FieldHandle Stimulus, MatrixHandle CompToGeom, double domaintype, bool selectbynode, StimulusTable& stimulustablelist)
{
  FNODE *elementtypefield = dynamic_cast<FNODE*>(ElementType.get_rep());
  FIELD *stimfield = dynamic_cast<FIELD*>(Stimulus.get_rep());

  if (stimfield == 0)
  {
    pr->error("BuildStimulusTable: The Stimulus field is empty");
    return (false);
  }

  if (elementtypefield == 0)
  {
    pr->error("BuildStimulusTable: The domaintype field is empty");
    return (false);
  }

  typename FNODE::mesh_handle_type elementtypemesh = elementtypefield->get_typed_mesh();
  if (elementtypemesh.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The domaintype mesh is empty");
    return (false);
  }

  typename FIELD::mesh_handle_type stimmesh = stimfield->get_typed_mesh();
  if (stimmesh.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The stimmesh is empty");
    return (false);
  }

  typename FNODE::mesh_type::Node::size_type nnodes;
  bool isgeomtocomp = false;
  int  *geomtocomprr = 0;
  int  *geomtocompcc = 0;

  if (CompToGeom.get_rep())
  {
    elementtypemesh->size(nnodes);
    if (nnodes != CompToGeom->nrows())
    {
      pr->error("BuildStimulusTable: The number of rows in CompToGeom is not of the right size");
      return (false);    
    }
    
    SparseRowMatrix* mat = dynamic_cast<SparseRowMatrix *>(CompToGeom.get_rep());
    if (mat == 0)
    {
      pr->error("BuildStimulusTable: CompToGeom is not a sparse matrix");
      return (false);    
    }
  
    geomtocomprr = mat->rows;
    geomtocompcc = mat->columns;
    
    for (int r=0; r<nnodes+1;r++)
    {
      if (geomtocomprr[r] != r)
      {
        pr->error("BuildStimulusTable: CompToGeom is mapping a geometrical node to multiple computational nodes");
        return (false);     
      }
    }
    isgeomtocomp = true;
  }

  typename FNODE::mesh_type::Elem::array_type elems;
  typename FNODE::mesh_type::Node::iterator it, it_end;
  typename FNODE::mesh_type::Node::size_type sz;
  typename FIELD::mesh_type::Elem::index_type ci;
  typename FNODE::value_type val, dval;
  Point point;

  elementtypemesh->size(sz);
  elementtypemesh->begin(it);
  elementtypemesh->end(it_end);  

  std::vector<bool> indomain(sz);

  dval = static_cast<typename FNODE::value_type>(domaintype);
  
  stimmesh->synchronize(Mesh::LOCATE_E|Mesh::EDGES_E);

  while (it != it_end)
  {
    indomain[(static_cast<unsigned int>(*it))] = false;
    
    elementtypemesh->get_center(point,*it);
    if (stimmesh->locate(ci,point))
    {
      elementtypemesh->get_elems(elems,*it);
      if (elems.size() > 0)
      {
        val = elementtypefield->value(elems[0]);
        if (val == dval)
        {
          indomain[static_cast<unsigned int>(*it)] = true;        
        }
      }
    }
    ++it;
  }

  typename FNODE::mesh_type::Elem::iterator cit, cit_end;
  typename FNODE::mesh_type::Node::array_type nodes;  
    
  elementtypemesh->begin(cit);
  elementtypemesh->end(cit_end);  
 
  stimulustablelist.clear();
 
  while (cit != cit_end)
  {
    elementtypemesh->get_nodes(nodes,*cit);
    for (size_t p=0; p<nodes.size();p++)
    {
      if (indomain[static_cast<unsigned int>(nodes[p])] == true) 
      {
        stimulusparam_type stimitem;
        stimitem.node = static_cast<unsigned int>(nodes[p]);
        stimitem.weight = weight_factor(elementtypemesh.get_rep(),*cit, nodes[p]);     
        stimulustablelist.push_back(stimitem);
      }
    }
    ++cit;
  }

  if (isgeomtocomp)
  {
    for (size_t p=0; p < stimulustablelist.size(); p++) 
    {
      stimulustablelist[p].node = geomtocompcc[stimulustablelist[p].node];
    }
  }

  std::sort(stimulustablelist.begin(),stimulustablelist.end());

  size_t k = 0;
  for (size_t p=1; p < stimulustablelist.size(); p++) 
  {
    if (stimulustablelist[p].node == stimulustablelist[k].node)
    {
      stimulustablelist[k].weight += stimulustablelist[p].weight;
      stimulustablelist[p].weight = 0.0;
    }
    else
    {
      k++;
      stimulustablelist[k] = stimulustablelist[p];
    }
  }
     
  if (stimulustablelist.size() > 0)       
  {
    stimulustablelist.resize(k+1);
  }
                 
  // Success:
  return (true);
}


} // end namespace ModelCreation

#endif 

