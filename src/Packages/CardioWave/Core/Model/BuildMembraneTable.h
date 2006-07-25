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

#ifndef CARDIOWAVE_CORE_FIELDS_BUILDMEMBRANETABLE_H
#define CARDIOWAVE_CORE_FIELDS_BUILDMEMBRANETABLE_H 1

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


class membraneparam_type {
  public:
    unsigned int node0;       // node in membrane model
    unsigned int node1;       // node in element type on 1st side
    unsigned int node2;       // node in element type on other side
    unsigned int snode;
    double       surface;
    
};

typedef std::vector<membraneparam_type> MembraneTable;


// Make the lsit sortable
inline bool operator==(const membraneparam_type& p1,const membraneparam_type& p2)
{
  if ((p1.node1 == p2.node1)&&(p1.node2 == p2.node2)) return (true);
  return (false);
}    

inline bool operator<(const membraneparam_type& p1, const membraneparam_type& p2)
{
  if (p1.node1 < p2.node1) return(true); 
  if (p1.node1 == p2.node1) if (p1.node2 < p2.node2) return(true);
  return (false);
}


class BuildMembraneTableAlgo : public DynamicAlgoBase
{
public:
  virtual bool BuildMembraneTable(ProgressReporter *pr, FieldHandle ElementType, FieldHandle MembraneModel, MatrixHandle CompToGeom, MatrixHandle NodeLink, MatrixHandle ElemLink, MembraneTable& membranetablelist, MatrixHandle& MappingMatrix);

};

template <class FVOL, class FSURF>
class BuildMembraneTableAlgoT : public BuildMembraneTableAlgo
{
public:
  virtual bool BuildMembraneTable(ProgressReporter *pr, FieldHandle ElementType, FieldHandle MembraneModel, MatrixHandle CompToGeom, MatrixHandle NodeLink, MatrixHandle ElemLink, MembraneTable& membranetablelist, MatrixHandle& MappingMatrix);
  
};


template <class FVOL, class FSURF>
bool BuildMembraneTableAlgoT<FVOL,FSURF>::BuildMembraneTable(ProgressReporter *pr, FieldHandle ElementType, FieldHandle MembraneModel, MatrixHandle CompToGeom, MatrixHandle NodeLink, MatrixHandle ElemLink, MembraneTable& membranetablelist, MatrixHandle& MappingMatrix)
{
  // Start the algorithm with a couple of checks to prevent Segment Violations.
  // If we cannot obtain handlers, we fail and return false
  // 1) Check whether we have an elementtype field 
  FVOL *elementtypefield = dynamic_cast<FVOL *>(ElementType.get_rep());
  if (elementtypefield == 0)
  { 
    pr->error("BuildMembraneTable: Could not obtain element class description field");
    return (false);
  }

  // 2) Check whether we can obtain the mesh of the elementtype field
  typename FVOL::mesh_handle_type elementtypemesh = elementtypefield->get_typed_mesh();
  if (elementtypemesh.get_rep() == 0)
  {
    pr->error("BuildMembraneTable: No mesh associated with element class description field");
    return (false);
  }

  // 3) Check whether the membranefield is valid
  FSURF *membranefield = dynamic_cast<FSURF *>(MembraneModel.get_rep());
  if (membranefield == 0)
  { 
    pr->error("BuildMembraneTable: Could not obtain model description field");
    return (false);
  }

  // 4) Same for the mesh
  typename FSURF::mesh_handle_type membranemesh = membranefield->get_typed_mesh();
  if (membranemesh.get_rep() == 0)
  {
    pr->error("BuildMembraneTable: No mesh associated with model description field");
    return (false);
  }


  //Get the sizes of both fields
  
  if (elementtypemesh->dimensionality() == 3) elementtypemesh->synchronize(Mesh::NODE_NEIGHBORS_E|Mesh::FACES_E);
  if (elementtypemesh->dimensionality() == 2) elementtypemesh->synchronize(Mesh::NODE_NEIGHBORS_E|Mesh::EDGES_E);
  if (elementtypemesh->dimensionality() == 1) elementtypemesh->synchronize(Mesh::NODE_NEIGHBORS_E|Mesh::NODES_E);
  
  typename FVOL::mesh_type::Node::size_type numelementnodes;
  typename FSURF::mesh_type::Node::size_type nummembranenodes;
  typename FVOL::mesh_type::DElem::size_type numelementdelems;

  elementtypemesh->size(numelementdelems);
  elementtypemesh->size(numelementnodes);
  membranemesh->size(nummembranenodes);
  
  
  // If there are no nodes there is no point in persuing this effort...
  if ((numelementnodes == 0)||(nummembranenodes == 0))
  {
    pr->error("BuildMembraneTable: One of the input meshes has no elements");
    return (false);  
  }

  // CompToGeom: Geometric nodes to computational nodes
  // ElemLink: Border elements to Opposing border elements
  // In this case only where we have two different domains in the outerboundary link
  // these need to be membranes as well.

  // Check whether CompToGeom was defined on the input, if not
  // maybe it is a property, else we ignore it...
  // it is an optional parameter
  if (CompToGeom.get_rep() == 0)
  {
    elementtypefield->get_property("CompToGeom",CompToGeom);
  }
  
  // Same for ElemLink
  if (ElemLink.get_rep() == 0)
  {
    elementtypefield->get_property("ElemLink",ElemLink);
  }

  if (NodeLink.get_rep() == 0)
  {
    elementtypefield->get_property("NodeLink",NodeLink);
  }  

  // VVV ALGORITHM STARTS HERE VVV

  // Make sure we have everything the mesh needs, compute all internal structures
  // needed
  

#ifdef HAVE_HASH_MAP
  typedef hash_multimap<unsigned int,typename FVOL::mesh_type::Node::index_type> nodemap_type;
#else
  typedef multimap<unsigned int,typename FVOL::mesh_type::Node::index_type> nodemap_type;
#endif

  int key;
  
  nodemap_type nodemap;
  Point point, point2, point3;

  double xmin = 0.0;
  double xmax = 0.0;
  double ymin = 0.0;
  double ymax = 0.0;
  double zmin = 0.0;
  double zmax = 0.0;
    
  typename FVOL::mesh_type::Node::iterator it, it_end;
  elementtypemesh->begin(it);
  elementtypemesh->end(it_end);
  
  // Compute a bounding box...
  
  // Initiate the parameters in a single loop
  if (it != it_end)
  {
    elementtypemesh->get_center(point,*it);
    xmin = point.x(); xmax = point.x();
    ymin = point.y(); ymax = point.y();
    zmin = point.z(); zmax = point.z();
    ++it;
  }
    
   // now compute the full bounding box.
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

  // TO DO:
  // Need to check whether bounding box is already available in al meshes..
  // It is a small overhead but may help reduce computational times

  // Define multipliers for the grid we are putting on top
  // Unfortunately the internal mesh grid is not accessible for every mesh type
  // hence we have to redo it. (Design error in SCIRun Mesh classes!)
  double xmul = 0.0; if ((xmax-xmin) > 0.0 ) xmul = 250/(xmax-xmin);
  double ymul = 0.0; if ((ymax-ymin) > 0.0 ) ymul = 250/(ymax-ymin);
  double zmul = 0.0; if ((zmax-zmin) > 0.0 ) zmul = 250/(zmax-zmin);

  xmin -= (xmax-xmin)*0.01;
  ymin -= (ymax-ymin)*0.01;
  zmin -= (zmax-zmin)*0.01;
                  
  elementtypemesh->begin(it);
  elementtypemesh->end(it_end);
  
  
  // Compute a key for each node in the mesh, so we can quickly find nodes when
  // we are looking for them. This is memory overhead, but otherwise this procedure
  // is extremely slow and we cannot use the internal locate function as we allow for
  // multiple nodes to be at the same position, hence the accelarator functions
  // in SCIRun are useless here, hence we need to do our own accelation.
  while (it != it_end)
  {
    elementtypemesh->get_center(point,*it);
    
    key = static_cast<int>((point.x()-xmin)*xmul);
    key += (static_cast<int>((point.y()-ymin)*ymul))<<8;
    key += (static_cast<int>((point.z()-zmin)*zmul))<<16;  
  
    nodemap.insert(typename nodemap_type::value_type(key,*it));
    ++it;
  }

  // Process ElemLink: this matrix tells which faces are connected by a 
  // membrane to faces on the opposite side of the mesh

  // Assume we do not have it
  bool iselemlink = false;
  int* elemlinkrr = 0;
  int* elemlinkcc = 0;
  
  
  if (ElemLink.get_rep())
  {
    // We have a ElemLink Matrix
    
    // Do a sanity check, if not return a proper error
    if ((numelementdelems != ElemLink->nrows())&&(numelementdelems != ElemLink->ncols()))
    {
      pr->error("BuildMembraneTable: The ElemLink property is not of the right dimensions");
      return (false);        
    }
    
    // Get the SparseMatrix, if not we will not do this operation
    SparseRowMatrix *spr = dynamic_cast<SparseRowMatrix *>(ElemLink.get_rep());
    if (spr)
    {
      elemlinkrr = spr->rows;
      elemlinkcc = spr->columns;
      iselemlink = true;
    }
    else
    {
      // Inform the user that they tried something ill-fated
      pr->error("BuildMembraneTable: The ElemLink matrix is not a sparse matrix");
      return (false);       
    }
    
  }  


  // Process ElemLink: this matrix tells which faces are connected by a 
  // membrane to faces on the opposite side of the mesh

  // Assume we do not have it
  bool isnodelink = false;
  int* nodelinkrr = 0;
  int* nodelinkcc = 0;
  
  
  if (NodeLink.get_rep())
  {
    // We have a ElemLink Matrix
    
    // Do a sanity check, if not return a proper error
    if ((numelementnodes != NodeLink->nrows())&&(numelementnodes != NodeLink->ncols()))
    {
      pr->error("BuildMembraneTable: The NodeLink property is not of the right dimensions");
      return (false);        
    }
    
    // Get the SparseMatrix, if not we will not do this operation
    SparseRowMatrix *spr = dynamic_cast<SparseRowMatrix *>(NodeLink.get_rep());
    if (spr)
    {
      nodelinkrr = spr->rows;
      nodelinkcc = spr->columns;
      isnodelink = true;
    }
    else
    {
      // Inform the user that they tried something ill-fated
      pr->error("BuildMembraneTable: The NodeLink matrix is not a sparse matrix");
      return (false);       
    }
    
  }  

  // The GeomToComp: Matrix for conversion of node numbers from geometrical domain
  // to computational domain.

  bool isgeomtocomp = false;
  int* geomrr = 0;
  int* geomcc = 0;
  
  if (CompToGeom.get_rep())
  {
    if ((numelementnodes != CompToGeom->nrows()))
    {
      pr->error("BuildMembraneTable: The CompToGeom matrix property is not of the right dimensions");
      return (false);        
    }
    SparseRowMatrix *spr = dynamic_cast<SparseRowMatrix *>(CompToGeom.get_rep());
    if (spr)
    {
      geomrr = spr->rows;
      geomcc = spr->columns;
      isgeomtocomp = true;
    }
    else
    {
      // Inform the user that they tried something ill-fated
      pr->error("BuildMembraneTable: The CompToGeom matrix is not a sparse matrix");
      return (false);       
    }    
    
    // Check whether it is a one-on-one conversion. Each geometrical node has to
    // map to one computational node (the opposite does not need to be true)
    for (int r=0; r<numelementnodes+1; r++)
    {
      if (geomrr[r] != r)
      {
        pr->error("BuildMembraneTable: The CompToGeom matrix property maps a geometric position to multiple computational nodes. This is not allowed");
        return (false);      
      }
    }
  }

  typename FSURF::mesh_type::Elem::iterator eit, eit_end;
  typename FSURF::mesh_type::Node::array_type nodes;  

  membranemesh->begin(eit);
  membranemesh->end(eit_end);  

  membranemesh->get_nodes(nodes,*eit); 
  
  int nodespersurf =nodes.size();
  std::vector<Point> points(nodespersurf);
  std::vector<double> surfaces(nodespersurf);


  bool foundsurf1;
  bool foundsurf2;

  std::pair<typename nodemap_type::iterator,typename nodemap_type::iterator> lit;
  typename FVOL::mesh_type::Node::index_type idx;
  typename FVOL::mesh_type::Elem::array_type elems;
  typename FVOL::mesh_type::Elem::array_type elems2;
  typename FVOL::mesh_type::DElem::array_type delems;
  typename FVOL::mesh_type::DElem::array_type delems2;
  typename FVOL::mesh_type::DElem::array_type delemlist;
  typename FVOL::mesh_type::Elem::array_type elemlist;
  typename FVOL::mesh_type::Node::array_type enodes;
  typename FVOL::mesh_type::Node::array_type enodes2;
  typename FSURF::value_type modeltype;
  typename FVOL::value_type classvalue1,classvalue2;
  double   surface;
  
  int k = 0;
  typename FSURF::mesh_type::Elem::size_type numelems;
  membranemesh->size(numelems);
  MembraneTable membranetablelisttemp; 
  membranetablelisttemp.resize(nodespersurf*numelems);
  
  
  // Loop over all elements in the membrane model
  while (eit != eit_end)
  {
    // We want to find two surfaces in the elementtype model for each element in
    // membrane model. This function is similar to locate(), but will find multiple
    // entries. Since membranes are infinitely thin, meshes touch eachother. And
    // unfortunately locate() was not designed for that purpose. Thus here is an
    // algorithm that avoids the problems with locate()
    
    // Assume we have not found the surfaces yet
    foundsurf1 = false;
    foundsurf2 = false;
    
    // Get the node numbers of the membrane element
    membranemesh->get_nodes(nodes,*eit); 
    
    surface = (membranemesh->get_size(*eit))/(nodes.size());
    
    // As we need to compare physical locations, start with getting the node locations
    for (size_t p =0; p < nodes.size(); p++) 
    {
      membranemesh->get_center(point2,nodes[p]);
      points[p] = point2;
      surfaces[p] = surface; // <<<< NEED TO ALTER THIS
    }

    // Calculate the key for the first point to find the corresponding node
    // in the hash_map
    key = static_cast<int>((points[0].x()-xmin)*xmul);
    key += (static_cast<int>((points[0].y()-ymin)*ymul))<<8;
    key += (static_cast<int>((points[0].z()-zmin)*zmul))<<16;
    
    // Find all nodes that can be used in the elementtype mesh
    // This to make sure we do not need to do an exhaustive search through the
    // whole mesh.
    lit = nodemap.equal_range(key);

    // loop through all possibilities
    while ((lit.first != lit.second)&&(foundsurf2 == false))
    {
      idx = (*(lit.first)).second;
      elementtypemesh->get_center(point2,idx);

      // check 1: target point needs to be same as point we are looking for
      // SearchGrid is of finite dimensions, thus two different points can have same key
      if (point2 == points[0])
      {
        // get all elements it is connected to:
        elementtypemesh->get_elems(elems,idx);
        delemlist.clear();
        elemlist.clear();
        
        // Get all unique faces that connect 
        for (size_t p = 0; p < elems.size(); p++)
        {
          // for each connected element get the faces
          elementtypemesh->get_delems(delems,elems[p]);
          for (size_t r = 0; r < delems.size();  r++)
          {
             // if it is already in the list skip it
             size_t s;
             for (s=0; s<delemlist.size(); s++) if (delems[r] == delemlist[s]) break;
             if (s < delemlist.size()) continue;
             elementtypemesh->get_nodes(enodes,delems[r]);
             size_t u;
             for (u=0;u<enodes.size();u++) if (enodes[u] == idx) break;
             if (u < enodes.size()) { delemlist.push_back(delems[r]); elemlist.push_back(elems[p]); }
          }
        }

        // check whether  the faces correcspond to the face we are looking for
        bool isequal = false;
        for (size_t r = 0; (r < delemlist.size())&&(!isequal);  r++)
        {
          // Get the nodes of the face for testing whether it contains the other locations
          // we are looking for
          elementtypemesh->get_nodes(enodes,delemlist[r]);
      
          isequal = true;
          for (size_t q = 0; q < enodes.size(); q++)
          {
            elementtypemesh->get_center(point2,enodes[q]);
            size_t t=0;
            for (; t< points.size(); t++) { if (point2 == points[t]) break; } 
            // if we did not find the node in this face it was definitely not
            // the one we were looking for.
            if (t == points.size()) 
            {
              isequal = false;
            }
          }

          // in case we found a surface
          if (isequal)
          {
            
            // is it the first one?
            if (foundsurf1 == false)
            {
              // Get the element type do decide the orientation of the membrane model
              elementtypefield->value(classvalue1,elemlist[r]);

              // Add the newly found surface to the list
              for (size_t q = 0; q < enodes.size(); q++)
              {           
                membranetablelisttemp[k+q].node1 = enodes[q];
              }
              foundsurf1 = true;

              // check whether we can figure out whether the link matrices can
              // provide us with the other surface
              if (iselemlink&&isnodelink)
              {    
                // if so elemlink so point to another face
                if (elemlinkrr[delemlist[r]] < elemlinkrr[delemlist[r]+1])
                {
                  // get the index of this other face
                  typename FVOL::mesh_type::DElem::index_type fidx = elemlinkcc[elemlinkrr[delemlist[r]]];
                  
                  // Now the ugly part, we need to access the element type of that face and there is no
                  // function from face to element.
                  // So we need to use the nodes 
                  
                  elementtypemesh->get_nodes(enodes2,fidx);
                  elementtypemesh->get_elems(elems2,enodes2[0]);
                  
                  bool foundelem = false;
                  for (size_t q = 0; (q < elems2.size())&&(!foundelem); q++)
                  {
                    elementtypemesh->get_delems(delems2,elems2[q]);
                    for (size_t u = 0; u < delems2.size(); u++)
                    {
                      if (delems2[u] == fidx) { elementtypefield->value(classvalue2,elems2[q]); foundelem = true; break; }
                    }
                  }
                  
                  // if they are in the same domain we should need to link them  
                  if (classvalue1 != classvalue2)
                  {  
                    // Now find the how the nodes connect over the boundary,
                    // loop through the nodelink matrix and figure out which link
                    // finds the nodes of the faces. This way we can establish the
                    // order as faces may show a different ordering of the nodes
                    for (size_t q = 0; q < enodes.size(); q++)
                    {           
                      bool foundnode = false;
                      for (size_t u=nodelinkrr[enodes[q]]; (u < nodelinkrr[enodes[q]+1])&&(!foundnode); u++)
                      {
                        for (size_t t=0; t < enodes2.size(); t++)
                        {
                          if (enodes2[t] == nodelinkcc[u]) 
                          { 
                            // add the node numbers in the proper order
                            membranetablelisttemp[k+q].node2 = enodes2[t];
                            foundnode = true; 
                            break;
                          }
                        }
                      }
                    }
                    
                    // Now find how it connects to the model surface:
                    // Here we rely on physical point locations as that is the
                    // only way, we did not keep track of node numbers when splitting
                    // out the surfaces. As membranes are subdivided a couple of 
                    // times, keeping track is a GUI nightmare, hence we reconstruct
                    // it here.
                    
                    for (size_t q = 0; q < enodes.size(); q++)
                    {  
                      elementtypemesh->get_center(point2,static_cast<typename FVOL::mesh_type::Node::index_type>(membranetablelisttemp[k+q].node1));
                      size_t s=0;
                      for (; s < points.size(); s++)
                      {
                        // TO DO: at tolerance here
                        if (points[s] == point2) break;
                      }
                      
                      // Surfaces for the connection are also defined based on the 
                      // membrane surface, hence we use this one here as well.
                      membranetablelisttemp[k+q].node0 = nodes[s];
                      membranetablelisttemp[k+q].surface = surfaces[s];                    
                     
                      // In case the membrane should be defined the other way around:
                      // Order matters: for instance a lot of models have an ICS and
                      // ECS side. Similarly gap junctions have different sides
                      // depending on cell type: e.g. fibroblast and myocyte                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                      if (classvalue1 < classvalue2)
                      {
                        unsigned int temp = membranetablelisttemp[k+q].node2;
                        membranetablelisttemp[k+q].node2 = membranetablelisttemp[k+q].node1;
                        membranetablelisttemp[k+q].node1 = temp;
                      }
                    }
                    
                    // finish by saying that we do not need to process second surface
                    foundsurf2 = true; 
                    k += enodes.size();
                  }
                }
              }
            }
            else if (foundsurf2 == false)
            {
              // we already have one surface but the other one is still missing
              // Get teh element value of this surface
              elementtypefield->value(classvalue2,elemlist[r]);
              
              
              // find out how surfaces are connected node wise
              for (size_t q = 0; q < enodes.size(); q++)
              { 
                // get the physical location to match it with the other surface
                // and the model surface
                
                // get physical location of frist face nodes
                elementtypemesh->get_center(point2,static_cast<typename FVOL::mesh_type::Node::index_type>(membranetablelisttemp[k+q].node1));
                
                // compare with currently found face
                size_t t=0;
                for (; t < enodes.size(); t++)
                {
                  elementtypemesh->get_center(point3,enodes[t]);
                  if (point3 == point2) break;
                }
                // t is now relative index on how different the nodes of the face line up

                // compare with membrane model itself
                size_t s=0;
                for (; s < points.size(); s++)
                {
                  if (points[s] == point2) break;
                }
                // s is now relative index
                            
                // depending on order we swap node2 and node1                                                                                                                                                                                  
                if (classvalue1 < classvalue2)
                {
                  membranetablelisttemp[k+q].node2 = enodes[t];
                }
                else
                {
                  membranetablelisttemp[k+q].node2 = membranetablelisttemp[k+q].node1;
                  membranetablelisttemp[k+q].node1 = enodes[t];
                }
                membranetablelisttemp[k+q].node0 = nodes[s];
                membranetablelisttemp[k+q].surface = surfaces[s];
              }
              foundsurf2 = true;
              k += enodes.size();
            }
          }
        }
      }
      ++(lit.first);
    }
      
//    if ((foundsurf1 == false)||(foundsurf2 == false))
//    {
//      pr->error("BuildMembraneTable: Not every surface/curve in the membrane model can be found in the element type mesh");
//      return (false);
//    }
    ++eit;
  }
  
  // we have the complete list
  // now merge it together:
  // we have double entries as we searched per face, adjoining faces will share
  // connections. Remove these to speed up computations in simulation phase


  int *mrr = scinew int[nummembranenodes+1];
  int *mcc = scinew int[nummembranenodes];
  double* mvv = scinew double[nummembranenodes];

  if ((mrr == 0)||(mcc == 0)||(mvv == 0))
  {
    if (mrr) delete[] mrr;
    if (mcc) delete[] mcc;
    if (mvv) delete[] mvv;
    
    pr->error("BuildMembraneTable: Could not generate mapping matrix");
    return (false);
  }

  for (int r=0; r < nummembranenodes; r++) mrr[r] = r;
  
  if (isgeomtocomp)
  {
    for (size_t q=0; q < membranetablelisttemp.size(); q++)
    {                  
      membranetablelisttemp[q].node1 = geomcc[membranetablelisttemp[q].node1];
      membranetablelisttemp[q].node2 = geomcc[membranetablelisttemp[q].node2];          
    }
  }
  
  std::sort(membranetablelisttemp.begin(),membranetablelisttemp.end());
  
  membranetablelist.clear();



  
  // Find unique entries and count them and move surface areas to add surfaces
  // of duplicate entries
  if (membranetablelisttemp.size() > 0)
  {
    size_t tablesize = 1;
    size_t q = 0;
    for (size_t p=1; p < membranetablelisttemp.size(); p++)
    {
      if (membranetablelisttemp[p] == membranetablelisttemp[q])
      {
        membranetablelisttemp[q].surface += membranetablelisttemp[p].surface;
        membranetablelisttemp[p].surface = 0.0;
        if (membranetablelisttemp[q].node0 > membranetablelisttemp[p].node0)
        {
          mrr[membranetablelisttemp[q].node0] = membranetablelisttemp[p].node0;
          membranetablelisttemp[q].node0 = membranetablelisttemp[p].node0;
        }
        else
        {
          mrr[membranetablelisttemp[p].node0] = membranetablelisttemp[q].node0;
        }
      }
      else if (membranetablelisttemp[p].surface != 0.0)
      {
        q = p;
        tablesize++;
      }
    } 


    for (int r=0; r<nummembranenodes; r++)
    {
      int p = r;
      while (mrr[p] != p) p = mrr[p];
      mrr[r] = p;      
    }

    k=0;
    for (int r=0; r<nummembranenodes ; r++)
    {
      if (mrr[r] == r) 
      {
        mrr[r] = k++;
      }
      else
      {
        mrr[r] = mrr[mrr[r]];
      }
    }
    
    // Build the final list with only unique entries
    q = 0;
    membranetablelist.resize(tablesize);
    for (size_t p=0; p < membranetablelisttemp.size(); p++)
    {
      if (membranetablelisttemp[p].surface != 0.0)
      {
        membranetablelist[q] = membranetablelisttemp[p];
        membranetablelist[q].node0 = mrr[membranetablelist[q].node0];
        q++;
      }    
    }
  }
  else
  {
    pr->warning("BuildMembraneTable: The Membrane geometry does not correspond to any of the element faces/edges of the element type field");
    return (true);    
  }    


  for (int r = 0; r < nummembranenodes; r++)
  {
    mcc[r] = mrr[r];
    mrr[r] = r;
    mvv[r] = 1.0;
  }
  mrr[nummembranenodes] = nummembranenodes; // An extra entry goes on the end of rr.

  MappingMatrix = dynamic_cast<Matrix *>(scinew SparseRowMatrix(nummembranenodes, k, mrr, mcc, nummembranenodes, mvv));
  
  if (MappingMatrix.get_rep() == 0)
  {
    pr->error("LinkToCompGrid: Could build geometry to computational mesh mapping matrix");
    return (false);
  }

  // Success:
  return (true);
}


} // end namespace ModelCreation

#endif 

