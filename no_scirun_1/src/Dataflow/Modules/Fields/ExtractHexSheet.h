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


//    File   : ExtractHexSheet.h
//    Author : Jason Shepherd
//    Date   : May 2006

#if !defined(ExtractHexSheet_h)
#define ExtractHexSheet_h

#define GTB_BEGIN_NAMESPACE namespace gtb {
#define GTB_END_NAMESPACE }

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/CurveMesh.h>
#include <vector>
#include <stack>

namespace SCIRun {

using std::copy;
using std::vector;
using std::stack;
    
using namespace std;

class GuiInterface;

class ExtractHexSheetAlgo : public DynamicAlgoBase
{
public:

  virtual void execute( 
      ProgressReporter *reporter, FieldHandle hexfieldh, vector<unsigned int> edges,
      FieldHandle& keptfield, FieldHandle& extractedfield ) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    string ext);
};


template <class FIELD>
class ExtractHexSheetAlgoHex : public ExtractHexSheetAlgo
{
public:
    //! virtual interface. 
  virtual void execute(
      ProgressReporter *reporter, FieldHandle hexfieldh, vector<unsigned int> edges, 
      FieldHandle& keptfield, FieldHandle& extractedfield );

  void get_opposite_edges( 
      typename HexVolMesh<HexTrilinearLgn<Point> >::Edge::index_type &opp_edge1, 
      typename HexVolMesh<HexTrilinearLgn<Point> >::Edge::index_type &opp_edge2, 
      typename HexVolMesh<HexTrilinearLgn<Point> >::Edge::index_type &opp_edge3,
      HexVolMesh<HexTrilinearLgn<Point> > *mesh,
      typename HexVolMesh<HexTrilinearLgn<Point> >::Elem::index_type hex_id, 
      typename HexVolMesh<HexTrilinearLgn<Point> >::Edge::index_type edge_id ); 

  void node_get_edges(typename FIELD::mesh_type *mesh,
                      set<typename FIELD::mesh_type::Edge::index_type> &result,
                      typename FIELD::mesh_type::Node::index_type node );
};


template <class FIELD>
void
ExtractHexSheetAlgoHex<FIELD>::execute(
    ProgressReporter *mod, FieldHandle hexfieldh, vector<unsigned int> edges, 
    FieldHandle& keptfield, FieldHandle& extractedfield )
{
#ifdef HAVE_HASH_MAP
  typedef hash_map<unsigned int,
    typename FIELD::mesh_type::Node::index_type,
    hash<unsigned int>,
    equal_to<unsigned int> > hash_type;
#else
  typedef map<unsigned int,
    typename FIELD::mesh_type::Node::index_type,
    less<unsigned int> > hash_type;
#endif

  typename FIELD::mesh_type *original_mesh =
      dynamic_cast<typename FIELD::mesh_type *>(hexfieldh->mesh().get_rep());

  typename FIELD::mesh_type *mesh_rep = 
      dynamic_cast<typename FIELD::mesh_type *>(hexfieldh->mesh().get_rep());
  typename FIELD::mesh_type *kept_mesh = scinew typename FIELD::mesh_type();
  kept_mesh->copy_properties( mesh_rep );
  typename FIELD::mesh_type *extracted_mesh = scinew typename FIELD::mesh_type();
  extracted_mesh->copy_properties( mesh_rep );

  original_mesh->synchronize( Mesh::EDGES_E | Mesh::NODE_NEIGHBORS_E );
  
  stack<typename FIELD::mesh_type::Edge::index_type> edge_stack;
  set<typename FIELD::mesh_type::Elem::index_type> extracted_hex_set;
  set<typename FIELD::mesh_type::Edge::index_type> used_edge_set;
  
  for( unsigned int i = 0; i < edges.size(); i++ )
  {        
    if( used_edge_set.find( (typename FIELD::mesh_type::Edge::index_type)edges[i] ) != used_edge_set.end() )
        continue;

    edge_stack.push( (typename FIELD::mesh_type::Edge::index_type)edges[i] );
    used_edge_set.insert( (typename FIELD::mesh_type::Edge::index_type)edges[i] );
    
    while( edge_stack.size() != 0 )
    {
      typename FIELD::mesh_type::Edge::index_type edge_id = edge_stack.top();
      edge_stack.pop();
      
      typename FIELD::mesh_type::Elem::array_type hex_array;
      
      original_mesh->get_elems( hex_array, edge_id );
      if( hex_array.size() == 0 )
      {
        cout << "ERROR: Edge " << edge_id << " does not exist in the mesh.\n";
        continue;
      }
      
      for( unsigned int j = 0; j < hex_array.size(); j++ )
      {
        typename FIELD::mesh_type::Edge::index_type opp_edge1, opp_edge2, opp_edge3;
        get_opposite_edges( opp_edge1, opp_edge2, opp_edge3, original_mesh, hex_array[j], edge_id );
        
        if( used_edge_set.find( opp_edge1 ) == used_edge_set.end() )
        {
          edge_stack.push( opp_edge1 );
          used_edge_set.insert( opp_edge1 );
        }
        if( used_edge_set.find( opp_edge2 ) == used_edge_set.end() )
        {
          edge_stack.push( opp_edge2 );
          used_edge_set.insert( opp_edge2 );
        }
        if( used_edge_set.find( opp_edge3 ) == used_edge_set.end() )
        {
          edge_stack.push( opp_edge3 );
          used_edge_set.insert( opp_edge3 );
        }
        
        extracted_hex_set.insert( hex_array[j] );
      }
    }
  }
  
  cout << "Extracting " << extracted_hex_set.size() << " hexes from the mesh.\n";
  
  set<typename FIELD::mesh_type::Node::index_type> affected_node_set;
  typename set<typename FIELD::mesh_type::Edge::index_type>::iterator es = used_edge_set.begin();
  typename set<typename FIELD::mesh_type::Edge::index_type>::iterator ese = used_edge_set.end();
  while( es != ese )
  {
    typename FIELD::mesh_type::Node::array_type two_nodes;
    original_mesh->get_nodes( two_nodes, *es );
    for( int i = 0; i < 2; i++ )
    {
      affected_node_set.insert( two_nodes[i] );
    }
    ++es;
  }

  typename set<typename FIELD::mesh_type::Elem::index_type>::iterator hs = extracted_hex_set.begin();  
  typename set<typename FIELD::mesh_type::Elem::index_type>::iterator hse = extracted_hex_set.end();
  
  hash_type extracted_nodemap;
  while( hs != hse )
  {
    typename FIELD::mesh_type::Node::array_type onodes;
    original_mesh->get_nodes( onodes, *hs );
    typename FIELD::mesh_type::Node::array_type nnodes(onodes.size());
    
    for (unsigned int k = 0; k < onodes.size(); k++)
    {    
      if( extracted_nodemap.find((unsigned int)onodes[k]) == extracted_nodemap.end())
      {
        Point np;
        original_mesh->get_center( np, onodes[k] );
        const typename FIELD::mesh_type::Node::index_type nodeindex =
            extracted_mesh->add_point( np );
        extracted_nodemap[(unsigned int)onodes[k]] = nodeindex;
        nnodes[k] = nodeindex;
      } 
      else
      {
        nnodes[k] = extracted_nodemap[(unsigned int)onodes[k]];
      }
    }
    extracted_mesh->add_elem( nnodes );
    ++hs;
  }

  typename FIELD::mesh_type::Cell::iterator citer;
  original_mesh->begin( citer );
  typename FIELD::mesh_type::Cell::iterator citere;
  original_mesh->end( citere );
  hash_type kept_nodemap;
  while( citer != citere )
  {
    typename FIELD::mesh_type::Elem::index_type hex_id = *citer;
    if( extracted_hex_set.find( hex_id ) == extracted_hex_set.end() )
    {
      typename FIELD::mesh_type::Node::array_type onodes;
      original_mesh->get_nodes( onodes, hex_id );
      typename FIELD::mesh_type::Node::array_type nnodes( onodes.size() );
      
      for (unsigned int k = 0; k < onodes.size(); k++)
      {
        if( kept_nodemap.find( (unsigned int)onodes[k] ) == kept_nodemap.end() )
        {
          if( affected_node_set.find( (unsigned int) onodes[k] ) == affected_node_set.end() )
          {
            Point np;
            original_mesh->get_center( np, onodes[k] );
            const typename FIELD::mesh_type::Node::index_type nodeindex = kept_mesh->add_point( np );
            kept_nodemap[(unsigned int)onodes[k]] = nodeindex;
            nnodes[k] = nodeindex;
          }
          else
          {
            stack<typename FIELD::mesh_type::Node::index_type> node_stack;
            set<typename FIELD::mesh_type::Edge::index_type> edge_string;
            set<typename FIELD::mesh_type::Node::index_type> node_string_set;
            node_string_set.insert( onodes[k] );
            node_stack.push( onodes[k] );
            while( node_stack.size() )
            {
              typename FIELD::mesh_type::Node::index_type stack_node = node_stack.top();
              node_stack.pop();
              
              set<typename FIELD::mesh_type::Edge::index_type> edge_set;
              node_get_edges( original_mesh, edge_set, stack_node );
              
              typename set<typename FIELD::mesh_type::Edge::index_type>::iterator esi = edge_set.begin();  
              typename set<typename FIELD::mesh_type::Edge::index_type>::iterator esie = edge_set.end();
              while( esi != esie )
              {
                if( used_edge_set.find( *esi ) != used_edge_set.end() )
                {
                  typename FIELD::mesh_type::Node::array_type edge_nodes;
                  edge_string.insert( *esi );
                  original_mesh->get_nodes( edge_nodes, *esi );
                  for( int j = 0; j < 2; j++ )
                  {
                    if( edge_nodes[j] != stack_node &&
                        ( node_string_set.find( edge_nodes[j] ) == node_string_set.end() ) )
                    {
                      node_string_set.insert( edge_nodes[j] );
                      node_stack.push( edge_nodes[j] );
                    }
                  }
                }
                ++esi;
              }
            }

              //find the average location of the node_string_set
            typename set<typename FIELD::mesh_type::Node::index_type>::iterator nss = node_string_set.begin();  
            typename set<typename FIELD::mesh_type::Node::index_type>::iterator nsse = node_string_set.end();
            Point np;
            original_mesh->get_center( np, *nss );
            ++nss;
            while( nss != nsse )
            {
              Point temp;
              original_mesh->get_center( temp, *nss );
              np += temp.vector();
              ++nss;
            }
            np /= node_string_set.size();
            
              //create a new point at this location
            const typename FIELD::mesh_type::Node::index_type node_index = kept_mesh->add_point( np );
              //set the kept_nodemap for all nodes in the set to this new point
            nss = node_string_set.begin();
            while( nss != nsse )
            {
              kept_nodemap[(unsigned int)(*nss)] = node_index;
              ++nss;
            }
            
              //add the point to the nnodes array
            nnodes[k] = node_index;
          } 
        }
        else
        {
          nnodes[k] = kept_nodemap[(unsigned int)onodes[k]];
        }
      }
      
      kept_mesh->add_elem( nnodes );
    }
    ++citer;
  }
  
  keptfield = scinew FIELD( kept_mesh );
  extractedfield = scinew FIELD( extracted_mesh );
  keptfield->copy_properties( hexfieldh.get_rep() );
  extractedfield->copy_properties( hexfieldh.get_rep() );
}

template <class FIELD>
void
ExtractHexSheetAlgoHex<FIELD>::get_opposite_edges( 
    typename HexVolMesh<HexTrilinearLgn<Point> >::Edge::index_type &opp_edge1, 
    typename HexVolMesh<HexTrilinearLgn<Point> >::Edge::index_type &opp_edge2, 
    typename HexVolMesh<HexTrilinearLgn<Point> >::Edge::index_type &opp_edge3,
    HexVolMesh<HexTrilinearLgn<Point> > *mesh,
    typename HexVolMesh<HexTrilinearLgn<Point> >::Elem::index_type hex_id, 
    typename HexVolMesh<HexTrilinearLgn<Point> >::Edge::index_type edge_id )
{
  typename FIELD::mesh_type::Edge::array_type all_edges;
  mesh->get_edges( all_edges, hex_id );

//  cout << all_edges[0] << " " << all_edges[1] << " " << all_edges[2] << " " << all_edges[3] << " " << all_edges[4] << " " << all_edges[5] << " " << all_edges[6] << " " << all_edges[7] << " " << all_edges[8] << " " << all_edges[9] << " " << all_edges[10] << " " << all_edges[11] << endl;

  if( edge_id == all_edges[0] )
  {
    opp_edge1 = all_edges[2];
    opp_edge2 = all_edges[4];
    opp_edge3 = all_edges[6];
  }  
  else if( edge_id == all_edges[3] )
  {
    opp_edge1 = all_edges[7];
    opp_edge2 = all_edges[1];
    opp_edge3 = all_edges[5];
  } 
  else if( edge_id == all_edges[8] )
  {
    opp_edge1 = all_edges[11];
    opp_edge2 = all_edges[9];
    opp_edge3 = all_edges[10];
  }
  else if( edge_id == all_edges[2] )
  {
    opp_edge1 = all_edges[0];
    opp_edge2 = all_edges[4];
    opp_edge3 = all_edges[6];
  } 
  else if( edge_id == all_edges[11] )
  {
    opp_edge1 = all_edges[8];
    opp_edge2 = all_edges[9];
    opp_edge3 = all_edges[10];
  }
  else if( edge_id == all_edges[4] )
  {
    opp_edge1 = all_edges[0];
    opp_edge2 = all_edges[2];
    opp_edge3 = all_edges[6];
  } 
  else if( edge_id == all_edges[7] )
  {
    opp_edge1 = all_edges[3];
    opp_edge2 = all_edges[1];
    opp_edge3 = all_edges[5];
  }
  else if( edge_id == all_edges[6] )
  {
    opp_edge1 = all_edges[0];
    opp_edge2 = all_edges[2];
    opp_edge3 = all_edges[4];
  } 
  else if( edge_id == all_edges[1] )
  {
    opp_edge1 = all_edges[3];
    opp_edge2 = all_edges[7];
    opp_edge3 = all_edges[5];
  }
  else if( edge_id == all_edges[9] )
  {
    opp_edge1 = all_edges[8];
    opp_edge2 = all_edges[11];
    opp_edge3 = all_edges[10];
  } 
  else if( edge_id == all_edges[10] )
  {
    opp_edge1 = all_edges[8];
    opp_edge2 = all_edges[11];
    opp_edge3 = all_edges[9];
  }
  else
  {
    opp_edge1 = all_edges[3];
    opp_edge2 = all_edges[7];
    opp_edge3 = all_edges[1];
  }
}
    
template <class FIELD>
void
ExtractHexSheetAlgoHex<FIELD>::node_get_edges(typename FIELD::mesh_type *mesh,
                                              set<typename FIELD::mesh_type::Edge::index_type> &result,
                                              typename FIELD::mesh_type::Node::index_type node )
{
  result.clear();

  typename FIELD::mesh_type::Elem::array_type elems;
  mesh->get_elems( elems, node );

  for (unsigned int i = 0; i < elems.size(); i++)
  {
    typename FIELD::mesh_type::Edge::array_type edges;
    mesh->get_edges( edges, elems[i] );
    for (unsigned int j = 0; j < edges.size(); j++)
    {
      typename FIELD::mesh_type::Node::array_type nodes;
      mesh->get_nodes( nodes, edges[j] );

      for (unsigned int k = 0; k < nodes.size(); k++)
      {
        if (nodes[k] == node)
        {
          result.insert( edges[j] );
          break;
        }
      }
    }
  }
}

} // end namespace SCIRun
#endif // ExtractHexSheet_h
