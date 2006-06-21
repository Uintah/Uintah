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

  original_mesh->synchronize( Mesh::EDGES_E );
  
//NOTE TO JS: Just playing.  Place the correct code here...
  stack<typename FIELD::mesh_type::Edge::index_type> edge_stack;
  set<typename FIELD::mesh_type::Elem::index_type> total_hex_set;
  
  for( unsigned int i = 0; i < edges.size(); i++ )
  {
    edge_stack.push( (typename FIELD::mesh_type::Edge::index_type)edges[i] );
    
    set<typename FIELD::mesh_type::Elem::index_type> hex_set;
    
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
        if( hex_set.find( hex_array[j] ) == hex_set.end() )
        {
          typename FIELD::mesh_type::Edge::index_type opp_edge1, opp_edge2, opp_edge3;
          get_opposite_edges( opp_edge1, opp_edge2, opp_edge3, original_mesh, hex_array[j], edge_id );
//NOTE TO JS:  These should be unique....
          edge_stack.push( opp_edge1 );
          edge_stack.push( opp_edge2 );
          edge_stack.push( opp_edge3 );
          
          hex_set.insert( hex_array[j] );
          total_hex_set.insert( hex_array[j] );
        }
      }
    }
  }
  
    typename set<typename FIELD::mesh_type::Elem::index_type>::iterator hs = total_hex_set.begin();  
    typename set<typename FIELD::mesh_type::Elem::index_type>::iterator hse = total_hex_set.end();
    
    hash_type nodemap;
    while( hs != hse )
    {
      typename FIELD::mesh_type::Node::array_type onodes;
      original_mesh->get_nodes( onodes, *hs );
      typename FIELD::mesh_type::Node::array_type nnodes(onodes.size());
      
      for (unsigned int k = 0; k < onodes.size(); k++)
      {    
        if( nodemap.find((unsigned int)onodes[k]) == nodemap.end())
        {
          Point np;
          original_mesh->get_center( np, onodes[k] );
          const typename FIELD::mesh_type::Node::index_type nodeindex =
              extracted_mesh->add_point( np );
          nodemap[(unsigned int)onodes[k]] = nodeindex;
          nnodes[k] = nodeindex;
        } 
        else
        {
          nnodes[k] = nodemap[(unsigned int)onodes[k]];
        }
      }
      extracted_mesh->add_elem( nnodes );
      ++hs;
    }
//end NOTE TO JS...  

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
    

} // end namespace SCIRun
#endif // ExtractHexSheet_h
