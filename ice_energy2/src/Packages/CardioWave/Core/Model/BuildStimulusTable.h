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

#ifndef CARDIOWAVE_CORE_FIELDS_BUILDSTIMULUSTABLE_H
#define CARDIOWAVE_CORE_FIELDS_BUILDSTIMULUSTABLE_H 1

// The following include file will include all tools needed for doing 
// dynamic compilation and will include all the standard dataflow types
#include <Core/Algorithms/Util/DynamicAlgo.h>

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


typedef std::vector<stimulusparam_type> StimulusTableList;

class BuildStimulusTableAlgo : public DynamicAlgoBase
{
public:
  virtual bool BuildStimulusTable(ProgressReporter *pr, FieldHandle domainnodetype, FieldHandle stimulus, double domaintype, bool selectbynode, StimulusTableList& stimulustablelist);

protected:
  template<class MESH>
  double weight_factor(MESH *mesh, typename MESH::Cell::index_type elemidx, typename MESH::Node::index_type nodeidx)
  {
     return (static<double>(mesh->get_size(elemidx)/mesh->basis()->number_of_vertices()));
  }

  template<class MESH>
  double weight_factor(MESH *mesh, typename MESH::Face::index_type elemidx, typename MESH::Node::index_type nodeidx)
  {
     return (static<double>(mesh->get_size(elemidx)/mesh->basis()->vertices_of_face()));
  }


  template<class MESH>
  double weight_factor(MESH *mesh, typename MESH::Edge::index_type elemidx, typename MESH::Node::index_type nodeidx)
  {
     return (static<double>(mesh->get_size(elemidx)/2));
  }

  template<>
  double weight_factor(QuadSurfMesh<QuadBilinearLgn<Point> >* mesh,
    QuadSurfMesh<QuadBilinearLgn<Point> >::Elem::index_type nodeidx
    QuadSurfMesh<QuadBilinearLgn<Point> >::Node::index_type nodeidx)
  {
     return (static<double>(mesh->get_size(elemidx)/4));
  }

  template<>
  double weight_factor(HexVolMesh<HexTrilinearLgn<Point> >* mesh,
    HexVolMesh<HexTrilinearLgn<Point> >::Elem::index_type nodeidx
    HexVolMesh<HexTrilinearLgn<Point> >::Node::index_type nodeidx)
  {
     return (static<double>(mesh->get_size(elemidx)/8));
  }

  template<>
  double weight_factor(HexVolMesh<HexTrilinearLgn<Point> >* mesh,
    HexVolMesh<HexTrilinearLgn<Point> >::Face::index_type nodeidx
    HexVolMesh<HexTrilinearLgn<Point> >::Node::index_type nodeidx)
  {
     return (static<double>(mesh->get_size(elemidx)/4));
  }

};

template <class FNODE, class FIELD>
class BuildStimulusTableCellAlgoT : public BuildStimulusTableAlgo
{
public:
  virtual bool BuildStimulusTable(ProgressReporter *pr, FieldHandle domainnodetype, FieldHandle stimulus, double domaintype, bool selectbynode, StimulusTableList& stimulustablelist);  
};

template <class FNODE, class FIELD>
class BuildStimulusTableNodeAlgoT : public BuildStimulusTableAlgo
{
public:
  virtual bool BuildStimulusTable(ProgressReporter *pr, FieldHandle domainnodetype, FieldHandle stimulus, double domaintype, bool selectbynode, StimulusTableList& stimulustablelist);  
};

template <class FNODE, class FIELD>
class BuildStimulusTableEdgeAlgoT : public BuildStimulusTableAlgo
{
public:
  virtual bool BuildStimulusTable(ProgressReporter *pr, FieldHandle domainnodetype, FieldHandle stimulus, double domaintype, bool selectbynode, StimulusTableList& stimulustablelist);  
};

template <class FNODE, class FIELD>
class BuildStimulusTableNodeAlgoT : public BuildStimulusTableAlgo
{
public:
  virtual bool BuildStimulusTable(ProgressReporter *pr, FieldHandle domainnodetype, FieldHandle stimulus, double domaintype, bool selectbynode, StimulusTableList& stimulustablelist);  
};

template <class FNODE, class FIELD>
bool BuildStimulusTableCellAlgoT<FNODE,FIELD>::BuildStimulusTable(ProgressReporter *pr, FieldHandle domainnodetype, FieldHandle stimulus, double domaintype, bool selectbynode, StimulusTableList& stimulustablelist)
{
  FNODE *domainfield = dynamic_cast<FNODE*>(domainnodetype.get_rep());
  FIELD *stimfield = dynamic_cast<FIELD*>(stimulus.get_rep());

  if (stimfield.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The stimulus field is empty");
    return (false);
  }

  if (domainfield.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The domaintype field is empty");
    return (false);
  }

  FNODE::mesh_handle_type domainmesh = domainfield->get_typed_mesh();
  if (domainmesh.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The domaintype mesh is empty");
    return (false);
  }

  FIELD::mesh_handle_type stimmesh = stimfield->get_typed_mesh();
  if (stimmesh.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The stimmesh is empty");
    return (false);
  }

  typename FNODE::mesh_type::Node::iterator it, it_end;
  typename FNODE::mesh_type::Node::size_t sz;
  typename FNODE::value_type val, dval;
  Point point;

  domainfield->begin(it);
  domainfield->begin(it_end);  

  std::vector<bool> indomain(sz);
  std::vector<double> voldomain(sz);

  dval = static_cast<typename FNODE::value_type>(domaintype);
  
  stimmesh->synchronize(LOCATE_E|CELL_E);

  while (it != it_end)
  {
    indomain[(static_cast<unsigned int>(*it))] = false;
    voldomain[(static_cast<unsigned int>(*it))] = 0.0;
    
    val = domainfield->value(*it);
    if (val == dval)
    {
      domainmesh->get_center(point,*it);
      if(stimmesh->locate(ci,point))
      {
        indomain[(static_cast<unsigned int>(*it))] = true;
      }
    }
    ++it;
  }

  typename FNODE::mesh_type::Cell::iterator cit, cit_end;
  typename FNODE::mesh_type::Node::array_type nodes;  
    
  domainmesh->begin(cit);
  domainmesh->begin(cit_end);  
 
  stimulustablelist.clear();

  while (cit != cit_end)
  {
    domainmesh->get_nodes(nodes,*cit);
    size_t p = 0;
    for (; p < nodes.size(); p++)
    {
      if (indomain[(static_cast<unsigned int>(nodes[p])] == false) break;
    }
    if (p == nodes.size())
    {
      for (p = 0; p < nodes.size(); p++)
      {
        stimulusparam_type stimitem;
        stimitem.node = static_cast<unsigned int>(nodes[p]);
        stimitem.weight = weight_factor(domainmesh,*cit, nodes[p]);     
        stimulustablelist.push_back(stimitem);
      }
    }
  }
  
  std::sort(stimulustablelist.begin(),stimulustablelist.end());

  size_t k = 0;
  for (size_t p=1; p < stimulustablelist.size(); p++) 
  {
    if (stimulustablelist[p].node == stimulustablelist[k].node)
    {
      stimulustablelist[p].weight += stimulustablelist[k].weight;
    }
    else
    {
      k++;
      stimulustablelist[k] = stimulustablelist[p];
    }
  }
     
  stimulustablelist.resize(k+1);
                 
  // Success:
  return (true);
}


template <class FNODE, class FIELD>
bool BuildStimulusTableFaceAlgoT<FNODE,FIELD>::BuildStimulusTable(ProgressReporter *pr, FieldHandle domainnodetype, FieldHandle stimulus, double domaintype, bool selectbynode, StimulusTableList& stimulustablelist)
{
  FNODE *domainfield = dynamic_cast<FNODE*>(domainnodetype.get_rep());
  FIELD *stimfield = dynamic_cast<FIELD*>(stimulus.get_rep());

  if (stimfield.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The stimulus field is empty");
    return (false);
  }

  if (domainfield.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The domaintype field is empty");
    return (false);
  }

  FNODE::mesh_handle_type domainmesh = domainfield->get_typed_mesh();
  if (domainmesh.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The domaintype mesh is empty");
    return (false);
  }

  FIELD::mesh_handle_type stimmesh = stimfield->get_typed_mesh();
  if (stimmesh.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The stimmesh is empty");
    return (false);
  }

  typename FNODE::mesh_type::Node::iterator it, it_end;
  typename FNODE::mesh_type::Node::size_t sz;
  typename FNODE::value_type val, dval;
  Point point;

  domainfield->begin(it);
  domainfield->begin(it_end);  

  std::vector<bool> indomain(sz);
  std::vector<double> voldomain(sz);

  dval = static_cast<typename FNODE::value_type>(domaintype);
  
  stimmesh->synchronize(LOCATE_E|FACES_E);

  while (it != it_end)
  {
    indomain[(static_cast<unsigned int>(*it))] = false;
    voldomain[(static_cast<unsigned int>(*it))] = 0.0;
    
    val = domainfield->value(*it);
    if (val == dval)
    {
      domainmesh->get_center(point,*it);
      if(stimmesh->locate(ci,point))
      {
        indomain[(static_cast<unsigned int>(*it))] = true;
      }
    }
    ++it;
  }

  typename FNODE::mesh_type::Face::iterator cit, cit_end;
  typename FNODE::mesh_type::Node::array_type nodes;  
    
  domainmesh->begin(cit);
  domainmesh->begin(cit_end);  
 
  stimulustablelist.clear();
 
  while (cit != cit_end)
  {
    domainmesh->get_nodes(nodes,*cit);
    size_t p = 0;
    for (; p < nodes.size(); p++)
    {
      if (indomain[(static_cast<unsigned int>(nodes[p])] == false) break;
    }
    if (p == nodes.size())
    {
      for (p = 0; p < nodes.size(); p++)
      {
        stimulusparam_type stimitem;
        stimitem.node = static_cast<unsigned int>(nodes[p]);
        stimitem.weight = weight_factor(domainmesh,*cit, nodes[p]);     
        stimulustablelist.push_back(stimitem);
      }
    }
  }
  
  std::sort(stimulustablelist.begin(),stimulustablelist.end());

  size_t k = 0;
  for (size_t p=1; p < stimulustablelist.size(); p++) 
  {
    if (stimulustablelist[p].node == stimulustablelist[k].node)
    {
      stimulustablelist[p].weight += stimulustablelist[k].weight;
    }
    else
    {
      k++;
      stimulustablelist[k] = stimulustablelist[p];
    }
  }
     
  stimulustablelist.resize(k+1);
                 
  // Success:
  return (true);
}



template <class FNODE, class FIELD>
bool BuildStimulusTableEdgeAlgoT<FNODE,FIELD>::BuildStimulusTable(ProgressReporter *pr, FieldHandle domainnodetype, FieldHandle stimulus, double domaintype, bool selectbynode, StimulusTableList& stimulustablelist)
{
  FNODE *domainfield = dynamic_cast<FNODE*>(domainnodetype.get_rep());
  FIELD *stimfield = dynamic_cast<FIELD*>(stimulus.get_rep());

  if (stimfield.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The stimulus field is empty");
    return (false);
  }

  if (domainfield.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The domaintype field is empty");
    return (false);
  }

  FNODE::mesh_handle_type domainmesh = domainfield->get_typed_mesh();
  if (domainmesh.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The domaintype mesh is empty");
    return (false);
  }

  FIELD::mesh_handle_type stimmesh = stimfield->get_typed_mesh();
  if (stimmesh.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The stimmesh is empty");
    return (false);
  }

  typename FNODE::mesh_type::Node::iterator it, it_end;
  typename FNODE::mesh_type::Node::size_t sz;
  typename FNODE::value_type val, dval;
  Point point;

  domainfield->begin(it);
  domainfield->begin(it_end);  

  std::vector<bool> indomain(sz);
  std::vector<double> voldomain(sz);

  dval = static_cast<typename FNODE::value_type>(domaintype);
  
  stimmesh->synchronize(LOCATE_E|EDGES_E);

  while (it != it_end)
  {
    indomain[(static_cast<unsigned int>(*it))] = false;
    voldomain[(static_cast<unsigned int>(*it))] = 0.0;
    
    val = domainfield->value(*it);
    if (val == dval)
    {
      domainmesh->get_center(point,*it);
      if(stimmesh->locate(ci,point))
      {
        indomain[(static_cast<unsigned int>(*it))] = true;
      }
    }
    ++it;
  }

  typename FNODE::mesh_type::Edge::iterator cit, cit_end;
  typename FNODE::mesh_type::Node::array_type nodes;  
    
  domainmesh->begin(cit);
  domainmesh->begin(cit_end);  
 
  stimulustablelist.clear();
 
  while (cit != cit_end)
  {
    domainmesh->get_nodes(nodes,*cit);
    size_t p = 0;
    for (; p < nodes.size(); p++)
    {
      if (indomain[(static_cast<unsigned int>(nodes[p])] == false) break;
    }
    if (p == nodes.size())
    {
      for (p = 0; p < nodes.size(); p++)
      {
        stimulusparam_type stimitem;
        stimitem.node = static_cast<unsigned int>(nodes[p]);
        stimitem.weight = weight_factor(domainmesh,*cit, nodes[p]);     
        stimulustablelist.push_back(stimitem);
      }
    }
  }

  std::sort(stimulustablelist.begin(),stimulustablelist.end());

  size_t k = 0;
  for (size_t p=1; p < stimulustablelist.size(); p++) 
  {
    if (stimulustablelist[p].node == stimulustablelist[k].node)
    {
      stimulustablelist[p].weight += stimulustablelist[k].weight;
    }
    else
    {
      k++;
      stimulustablelist[k] = stimulustablelist[p];
    }
  }
     
  stimulustablelist.resize(k+1);
                 
  // Success:
  return (true);
}


template <class FNODE, class FIELD>
bool BuildStimulusTableNodeAlgoT<FNODE,FIELD>::BuildStimulusTable(ProgressReporter *pr, FieldHandle domainnodetype, FieldHandle stimulus, double domaintype, bool selectbynode, StimulusTableList& stimulustablelist)
{
  FNODE *domainfield = dynamic_cast<FNODE*>(domainnodetype.get_rep());
  FIELD *stimfield = dynamic_cast<FIELD*>(stimulus.get_rep());

  if (stimfield.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The stimulus field is empty");
    return (false);
  }

  if (domainfield.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The domaintype field is empty");
    return (false);
  }

  FNODE::mesh_handle_type domainmesh = domainfield->get_typed_mesh();
  if (domainmesh.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The domaintype mesh is empty");
    return (false);
  }

  FIELD::mesh_handle_type stimmesh = stimfield->get_typed_mesh();
  if (stimmesh.get_rep() == 0)
  {
    pr->error("BuildStimulusTable: The stimmesh is empty");
    return (false);
  }

  typename FNODE::mesh_type::Node::iterator it, it_end;
  typename FNODE::mesh_type::Node::size_t sz;
  typename FNODE::value_type val, dval;
  Point point;

  domainfield->begin(it);
  domainfield->begin(it_end);  

  std::vector<bool> indomain(sz);
  std::vector<double> voldomain(sz);

  dval = static_cast<typename FNODE::value_type>(domaintype);
  
  stimmesh->synchronize(LOCATE_E|EDGES_E);

  while (it != it_end)
  {
    indomain[(static_cast<unsigned int>(*it))] = false;
    voldomain[(static_cast<unsigned int>(*it))] = 0.0;
    
    val = domainfield->value(*it);
    if (val == dval)
    {
      domainmesh->get_center(point,*it);
      if(stimmesh->locate(ci,point))
      {
        indomain[(static_cast<unsigned int>(*it))] = true;
      }
    }
    ++it;
  }

  typename FNODE::mesh_type::Elem::iterator cit, cit_end;
  typename FNODE::mesh_type::Node::array_type nodes;  
    
  domainmesh->begin(cit);
  domainmesh->begin(cit_end);  
 
  stimulustablelist.clear();
 
  while (cit != cit_end)
  {
    get_nodes(nodes,*cit);
    for (size_t p=0; p<nodes.size();p++)
    {
      if (indomain[(static_cast<unsigned int>(nodes[p])] == true) 
      {
        stimulusparam_type stimitem;
        stimitem.node = static_cast<unsigned int>(nodes[p]);
        stimitem.weight = weight_factor(domainmesh,*cit, nodes[p]);     
        stimulustablelist.push_back(stimitem);
      }
    }
  }

  std::sort(stimulustablelist.begin(),stimulustablelist.end());

  size_t k = 0;
  for (size_t p=1; p < stimulustablelist.size(); p++) 
  {
    if (stimulustablelist[p].node == stimulustablelist[k].node)
    {
      stimulustablelist[p].weight += stimulustablelist[k].weight;
    }
    else
    {
      k++;
      stimulustablelist[k] = stimulustablelist[p];
    }
  }
     
  stimulustablelist.resize(k+1);
                 
  // Success:
  return (true);
}


} // end namespace ModelCreation

#endif 

