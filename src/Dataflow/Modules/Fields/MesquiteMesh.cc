
#include <Core/Datatypes/Field.h>
#include <Dataflow/Modules/Fields/MesquiteMesh.h>
#include <MsqVertex.hpp>

namespace SCIRun {
    
  // MesquiteMesh constructor
template <class FIELD> 
MesquiteMesh<FIELD>::MesquiteMesh( FIELD* fieldh )
{
    //make sure that we are given an entity
  if( !fieldh.get_rep() )
  {
    return;
  }
  
  mOwner = fieldh;
  
//   DLIList<CubitNode*> node_list;
//   DLIList<MeshEntity*> element_list;
//     //see whether we are working with a surface or volume
//   myDimension = (unsigned char)(entity->dimension());
//     //if neither a surface or a volume, we don't know what to do.
//   if(myDimension<2 || myDimension>3){
//     PRINT_ERROR("Entities of dimesnion %d are not yet handled by MesquiteMesh.\n",myDimension);
//     return;
//   }

    //get the nodes and elements of this entity
//  mOwner->nodes_inclusive(node_list);
//  mOwner->elements(element_list);
  mOwner->mesh()->size( numNodes );
  mOwner->mesh()->size( numElements ); 
  mOwner->mesh()->begin(niterBegin); 
  mOwner->mesh()->end(niterEnd);
  mOwner->mesh()->begin(eiterBegin); 
  mOwner->mesh()->end(eiterEnd);
  
//   numNodes=node_list.size();
//   numElements=element_list.size();
//   nodeArray= new CubitNode*[numNodes];
//   elementArray= new MeshEntity*[numElements];
    //Add each node to the glboal array and add a TDMesquiteFlag to
    //each node.  This TD stores the byte which
    //is setable only by Mesquite, and the index into the global node
    //array (that is, nodeArray) for the given node.

//NOTE TO JS...  Initialize the field values...
  cout << "ERROR: MesquiteMesh::MesquiteMesh byte flags are not initialized." << endl;
  
//   size_t i;
//   node_list.reset();
//   for(i=0;i < ((size_t) numNodes);++i){
//     nodeArray[i]=node_list.get_and_step();
//     TDMesquiteFlag* td_msq_flag = new TDMesquiteFlag(nodeArray[i],i,0);
//     nodeArray[i]->add_TD(td_msq_flag);
//   }

    //Add each element to the global element array (elementArray)
//   element_list.reset();
//   CubitTet* tet_ptr=NULL;
//   CubitHex* hex_ptr=NULL;
  triExists = false;
  quadExists=false;
  tetExists=false;
  hexExists=false;
    //add element pointers to element array and add hex/tet knowing
    // information to nodes 
  const TypeDescription *mtd = mOwner->mesh()->get_type_description();
  if (mtd->get_name().find("TetVolMesh") != string::npos)
  {
//    ext = "Tet";
    tetExists=true;
    myDimension = 3;
  }
  else if (mtd->get_name().find("TriSurfMesh") != string::npos)
  {
//    ext = "Tri";
    triExists=true;
    myDimension = 2;
  }
  else if (mtd->get_name().find("HexVolMesh") != string::npos)
  {
//    ext = "Hex";
    hexExists=true;
    myDimension = 3;
  }
  else if (mtd->get_name().find("QuadSurfMesh") != string::npos)
  {
    quadExists=true;
    myDimension = 2;
  }
  else
  {
//    mod->error("Unsupported mesh type.  This module only works on Tets, Tris, Quads and Hexes.");
    return;
  }
//   for(i=0;i < ((size_t) numElements) ;++i){
//     elementArray[i]=element_list.get_and_step();
//     if(CAST_TO(elementArray[i],CubitTri))
//     {  typename Fld::mesh_type::Node::iterator niter_end;
//       triExists=CUBIT_TRUE;
//     }
//     else if(CAST_TO(elementArray[i],CubitFace))
//     {
//       quadExists=CUBIT_TRUE;
//     }
//     else if((tet_ptr=CAST_TO(elementArray[i],CubitTet)) != NULL)
//     {
//       tet_ptr->add_tet_to_nodes();
//       tetExists=CUBIT_TRUE;
//     }
//     else if((hex_ptr=CAST_TO(elementArray[i],CubitHex)) != NULL)
//     {
//       hex_ptr->add_hex_to_nodes();
//       hexExists=CUBIT_TRUE;
//     }
//     else{
//       PRINT_ERROR("Type not recognized.\n");
//     }
//   }
    
//     //for 'local' type calls, we currently cache the neighboring
//     //entities for a node, because multiple calls are often made
//     //which require this information.  These variables are new'd to
//     //get around some const-ness issues.
//   cachedNode= new CubitNode*;
//   *cachedNode = NULL;
//   cachedEntityList=new DLIList<MeshEntity*>;
}
  //destructor
template <class FIELD>
MesquiteMesh<FIELD>::~MesquiteMesh()
{
//NOTE TO JS: delete the TDMesquiteFlag for each node.
//   int i;
//   for(i=0;i<numNodes;++i){
//     if(nodeArray[i]!=NULL){
//       nodeArray[i]->delete_TD(&TDMesquiteFlag::is_mesquite_flag_node);
//       nodeArray[i]=NULL;
//     }
//   }
   
//   if(tetExists==CUBIT_TRUE){
//     DLIList<CubitTet*> tet_list;
//     mOwner->tets(tet_list);
//     if( tet_list.size() )
//       CubitTet::delete_tets_from_nodes( tet_list );
//   }
//   if(hexExists==CUBIT_TRUE){
//     DLIList<CubitHex*> hex_list;
//     mOwner->hexes(hex_list);
//     if( hex_list.size() )
//       CubitHex::delete_hexes_from_nodes( hex_list );
//   }
    
//     //delete the global arrays and the cached data.
//   delete cachedNode;
//   delete cachedEntityList;
//   delete [] nodeArray;
//   delete [] elementArray;
    
}

  
  /*! We always pass in nodes in with three coordinates.  This may change
    in the future if we want to do smoothing in a parametric space, but
    !for now, we are always in three-dimensions. */
template <class FIELD>
int MesquiteMesh<FIELD>::get_geometric_dimension(
  Mesquite::MsqError &/*err*/)
{
  cout << "WARNING: MesquiteMesh::get_geometric_dimension called." << endl;
  return 3;
}

  //! Returns the number of verticies for the entity.
template <class FIELD>
size_t MesquiteMesh<FIELD>::get_total_vertex_count(
  Mesquite::MsqError &/*err*/) const
{
  cout << "WARNING: MesquiteMesh::get_total_vertex_count called." << endl;
  return (size_t) numNodes;
}
  
  //! Returns the number of elements for the entity.
template <class FIELD>
size_t MesquiteMesh<FIELD>::get_total_element_count(
  Mesquite::MsqError &/*err*/) const
{
  cout << "WARNING: MesquiteMesh::get_total_element_count called." << endl;
  return (size_t) numElements;
}

  //! Fills array with handles to all vertices in the mesh.
template <class FIELD>
void MesquiteMesh<FIELD>::get_all_vertices(
  vector<Mesquite::Mesh::VertexHandle> &vertices,
  Mesquite::MsqError &/*err*/)
{
  cout << "WARNING: MesquiteMesh::get_all_vertices called." << endl;
  
    //otherwise add a handle to each vertex to the given array.
//  int index = 0;
  niterCurrent = niterBegin;

  vertices.clear();

//  while (index<numNodes )
  while( niterCurrent != niterEnd )
  {
//     vertices.push_back( reinterpret_cast<Mesquite::Mesh::VertexHandle>(nodeArray[index] ));
//     ++index;
    vertices.push_back( reinterpret_cast<Mesquite::Mesh::VertexHandle>( *niterCurrent ));
    ++niterCurrent;
  }
}
  //! Fills array with handles to all elements in the mesh.
template <class FIELD>
void MesquiteMesh<FIELD>::get_all_elements(   
  vector<Mesquite::Mesh::ElementHandle> &elements,      
  Mesquite::MsqError &/*err*/ )
{
  cout << "WARNING: MesquiteMesh::get_all_elements called." << endl;
  
    //otherwise add a handle to each element to the given array.
//  int index = 0;
  eiterCurrent = eiterBegin;

  elements.clear();

//  while (index<numElements )
  while( eiterCurrent != eiterEnd )
  {
//    elements.push_back(
//       reinterpret_cast<Mesquite::Mesh::ElementHandle>(elementArray[index]));
//     ++index;
    elements.push_back( reinterpret_cast<Mesquite::Mesh::ElementHandle>(*eiterCurrent));
    ++eiterCurrent;
  }
}

  
//! Returns a pointer to an iterator that iterates over the
//! set of all vertices in this mesh.  The calling code should
//! delete the returned iterator when it is finished with it.
//! If vertices are added or removed from the Mesh after obtaining
//! an iterator, the behavior of that iterator is undefined.
template <class FIELD>
Mesquite::VertexIterator* MesquiteMesh<FIELD>::vertex_iterator(Mesquite::MsqError &/*err*/)
{
  cout << "WARNING: MesquiteMesh::vertex_iterator initialized." << endl;
//  return new MesquiteMesh<FIELD>::VertexIterator(this);
  typename FIELD::mesh_type::Node::iterator niter;
  mOwner->mesh()->begin( niter ); 
  return niter;
}

  
//! Returns a pointer to an iterator that iterates over the
//! set of all top-level elements in this mesh.  The calling code should
//! delete the returned iterator when it is finished with it.
//! If elements are added or removed from the Mesh after obtaining
//! an iterator, the behavior of that iterator is undefined.
template <class FIELD>
Mesquite::ElementIterator* MesquiteMesh<FIELD>::element_iterator(Mesquite::MsqError &/*err*/)
{
  cout << "WARNING: MesquiteMesh::element_iterator initialized." << endl;
//  return new MesquiteMesh<FIELD>::ElementIterator(this);
  typename FIELD::mesh_type::Elem::iterator eiter;
  mOwner->mesh()->begin( eiter ); 
  return eiter;
}

//! Returns true or false, indicating whether the vertex
//! is allowed to moved.
//! Note that this is a read-only
//! property; this flag can't be modified by users of the
//! Mesquite::Mesh interface.
template <class FIELD>
void MesquiteMesh<FIELD>::vertices_get_fixed_flag(
  const Mesquite::Mesh::VertexHandle vert_array[], 
  bool fixed_flag_array[],
  size_t num_vtx, 
  Mesquite::MsqError &err )
{
  cout << "ERROR: MesquiteMesh::vertices_get_fixed_flag has not been implemented." << endl;
  
//   int i;
//   MRefEntity* owner_ptr = reinterpret_cast<MRefEntity*>(mOwner);
//   if(owner_ptr == NULL){
      
//     MSQ_SETERR(err)("MesquiteMesh::vertex_is_on_boundary: Null pointer to owner.", Mesquite::MsqError::INVALID_STATE);
//     PRINT_ERROR(" MesquiteMesh::vertex_is_on_boundary: Null pointer to owner.\n");
//     return;
//   }
    
//   CubitNode* node_ptr = NULL;
    
//   for (i = 0; i < num_vtx; ++i){
//     node_ptr = reinterpret_cast<CubitNode*>(vert_array[i]);
//       //if we've got a null pointer, something is wrong.
//     if(node_ptr==NULL){
//       MSQ_SETERR(err)("MesquiteMesh::vertex_is_on_boundary: Null pointer to vertex.", Mesquite::MsqError::INVALID_STATE);
//       PRINT_ERROR(" MesquiteMesh::vertex_is_on_boundary: Null pointer to vertex.\n");
//       return;
//     }
//       //if the owner is something other than the top-level owner, the node
//       // is on the boundary; otherwise, it isn't.
//     if(owner_ptr==node_ptr->owner() || node_ptr->position_fixed())
//       fixed_flag_array[i] = false;
//     else
//       fixed_flag_array[i] = true;
//   }
}

  
//! Get location of a vertex
template <class FIELD>
void MesquiteMesh<FIELD>::vertices_get_coordinates(
  const Mesquite::Mesh::VertexHandle vert_array[],
  Mesquite::MsqVertex* coordinates,
  size_t num_vtx,
  Mesquite::MsqError &err)
{
  cout << "WARNING: MesquiteMesh::vertices_get_coordinates called." << endl;
  
  int i;
//   CubitNode* node_ptr = NULL;
  for (i = 0; i<num_vtx; ++i)
  {  
//     node_ptr=reinterpret_cast<CubitNode*>(vert_array[i]);
//       //if null pointer, there is a problem somewhere.  We set the vector's
//       // position to (0,0,0) just to avoid un-initialized variable issues.
//     if(node_ptr==NULL) {
//       MSQ_SETERR(err)("MesquiteMesh::vertex_get_coordinates: invalid vertex handle.", Mesquite::MsqError::INVALID_STATE);
//       PRINT_ERROR("MesquiteMesh::vertex_get_coordinates: invalid vertex handle.\n");
//       coordinates[i].set(0.0,0.0,0.0);
//       return;
//     }
      //set coordinates to the vertex's position.
    Point p;
    mOwner->mesh()->get_point( p, *(vert_array[i]) );
    
    coordinates[i].set( p.x(), p.y(), p.z() );
  }
}
  
//! Set the location of a vertex.
template <class FIELD>
void MesquiteMesh<FIELD>::vertex_set_coordinates(
  VertexHandle vertex,
  const Mesquite::Vector3D &coordinates,
  Mesquite::MsqError &err)
{
//   CubitNode* node_ptr=reinterpret_cast<CubitNode*>(vertex);
//     //make sure no null pointer
//   if(node_ptr==NULL) {
//     MSQ_SETERR(err)("MesquiteMesh::vertex_set_coordinates: invalid vertex handle.",Mesquite::MsqError::INVALID_STATE);
//     PRINT_ERROR("MesquiteMesh::vertex_set_coordinates: invalid vertex handle.\n");
//     return;
//   }
//     //set the vertex's position to the given coordinates.
//   node_ptr->set(coordinates[0], coordinates[1], coordinates[2]);
    Point p;
    mOwner->mesh()->get_point( p, *(vertex) );
    cout << "WARNING: Point " << vertex << " moved from <" << p.x() << ", " << p.y() << ", " << p.z() << "> to <" << coordinates[0] << ", " << coordinates[1] << ", " << coordinates[2] << ">." << endl;
    p.x( coordinates[0] );
    p.y( coordinates[1] );
    p.z( coordinates[2] );
    mOwner->mesh->set_point( p, *(vertex) );
}


//! Each vertex has a byte-sized flag that can be used to store
//! flags.  This byte's value is neither set nor used by the mesh
//! implementation.  It is intended to be used by Mesquite algorithms.
//! Until a vertex's byte has been explicitly set, its value is 0.
//! Cubit stores the byte on the TDMesquiteFlag associated with the
//! node.
template <class FIELD>
void MesquiteMesh<FIELD>::vertex_set_byte (VertexHandle vertex,
                                             unsigned char byte,
                                             Mesquite::MsqError &err)
{
  cout << "ERROR: MesquiteMesh::vertex_set_byte has not been implemented." << endl;
  
//   CubitNode* node_ptr=reinterpret_cast<CubitNode*>(vertex);
//     //make sure there isn't a null pointer.
//   if(node_ptr==NULL) {
//     MSQ_SETERR(err)("MesquiteMesh::vertex_set_byte: invalid vertex handle.", Mesquite::MsqError::INVALID_STATE);
//     PRINT_ERROR("MesquiteMesh::vertex_set_byte: invalid vertex handle.\n");
//     return;
//   }
//     //get the TDMesquiteFlag associated with this node.
//   ToolData *td = node_ptr->get_TD(&TDMesquiteFlag::is_mesquite_flag_node);
//   TDMesquiteFlag* td_msq_flag = static_cast<TDMesquiteFlag*>(td);
//     //There should be a TD.  If there isn't, there is a problem.
//   if(td_msq_flag==NULL){
//     PRINT_ERROR("Node does not have the correct tool data.\n");
//   }
//   else{
//     td_msq_flag->set_byte(byte);
//   }
}

//! Set the byte for a given array of vertices.
template <class FIELD>
void MesquiteMesh<FIELD>::vertices_set_byte (
  const VertexHandle *vert_array,
  const unsigned char *byte_array,
  size_t array_size,
  Mesquite::MsqError &err)
{
  cout << "ERROR: MesquiteMesh::vertices_set_byte has not been implemented." << endl;
  
//     //loop over the given vertices and call vertex_set_byte(...).
//   size_t i=0;
//   for(i=0;i<array_size;++i){
//     vertex_set_byte(vert_array[i],byte_array[i],err);
//   }

}
  
//! Retrieve the byte value for the specified vertex or vertices.
//! The byte value is 0 if it has not yet been set via one of the
//! *_set_byte() functions.
template <class FIELD>
void MesquiteMesh<FIELD>::vertex_get_byte(VertexHandle vertex,
                                            unsigned char *byte,
                                            Mesquite::MsqError &err)
{
  cout << "ERROR: MesquiteMesh::vertex_get_byte has not been implemented." << endl;
  
//   CubitNode* node_ptr=reinterpret_cast<CubitNode*>(vertex);
//     //make sure there isn't a null pointer.
//   if(node_ptr==NULL) {
//     MSQ_SETERR(err)("MesquiteMesh::vertex_get_byte: invalid vertex handle.", Mesquite::MsqError::INVALID_STATE);
//     PRINT_ERROR("MesquiteMesh::vertex_get_byte: invalid vertex handle.\n");
//     return;
//   }
//     //get the TDMesquiteFlag
//   ToolData *td = node_ptr->get_TD(&TDMesquiteFlag::is_mesquite_flag_node);
//   TDMesquiteFlag* td_msq_flag = static_cast<TDMesquiteFlag*>(td);
//     //if there isn't a TDMesquiteFlag for this vertex, there's a problem.
//   if(td_msq_flag==NULL){
//     *byte=0;
//   }
//   else{
//     *byte=td_msq_flag->get_byte();
//   }
}

//! get the bytes associated with the vertices in a given array.
template <class FIELD>
void MesquiteMesh<FIELD>::vertices_get_byte(
  const VertexHandle *vertex_array,
  unsigned char *byte_array,
  size_t array_size,
  Mesquite::MsqError &err)
{
  cout << "ERROR: MesquiteMesh::vertices_get_byte has not been implemented." << endl;
  
//     //loop over the given nodes and call vertex_get_byte(...)
//   size_t i=0;
//   for(i=0;i<array_size;++i){
//     vertex_get_byte(vertex_array[i],&byte_array[i],err);
//   }
}

  
//! Gets the elements attached to this vertex.
template <class FIELD>
void MesquiteMesh<FIELD>::vertices_get_attached_elements(
    const VertexHandle* vertex_array,
    size_t num_vertex,
    msq_std::vector<ElementHandle>& elements,
    msq_std::vector<size_t>& offsets,
    Mesquite::MsqError& err )
{
  cout << "ERROR: MesquiteMesh::vertices_get_attached_elements has not been implemented." << endl;
  
// //   PRINT_INFO("ENTERING VERTEX_GET_ATTACHED_ELEMENTS\n");
//     int i=0, j=0;
//     int list_size=0;
//     elements.clear();
//     offsets.clear();
//     CubitNode* node_ptr=NULL;
//     ElementHandle temp_e_handle;
//     size_t offset_counter = 0;
    
//     for(i=0;i<num_vertex;++i){
//       offsets.push_back(offset_counter);
//       node_ptr=reinterpret_cast<CubitNode*>(vertex_array[i]);
//         //make sure there isn't a null pointer
//       if(node_ptr==NULL) {
//         MSQ_SETERR(err)("MesquiteMesh::vertex_get_attached_elements: invalid vertex handle.", Mesquite::MsqError::INVALID_STATE);
//         PRINT_ERROR("MesquiteMesh::vertex_get_attached_elements: invalid vertex handle.\n");
//         return;
//       }
    
//         //make sure the elements on the node have been cached.
//       if(!cache_elements_attached_to_node(node_ptr)){
//         MSQ_SETERR(err)("vertex_get_attached_elements problem creating element cache.", Mesquite::MsqError::INVALID_STATE);
//         return;
//       }
//         //make sure that enough space has been allocated.
//       list_size=cachedEntityList->size();
//         //add element handles to the given array.
//       cachedEntityList->reset();
//       for(j=0;j<list_size;++j){
//         if(cachedEntityList->get()->owner()==mOwner){
//           temp_e_handle =
//             reinterpret_cast<ElementHandle>(cachedEntityList->get());
//           if(!temp_e_handle){
//             MSQ_SETERR(err)("vertex_get_attached_elements invalid elements.", Mesquite::MsqError::INVALID_STATE);
//             return;
//           }
          
//           elements.push_back(temp_e_handle);
//           ++offset_counter;
//         }
//         cachedEntityList->step();
//       }      
//     }
//     offsets.push_back(offset_counter);
}


  
/*! \brief  
  Returns the vertices that are part of the topological definition of each
  element in the "elem_handles" array.  
  
  When this function is called, the
  following must be true:
  -# "elem_handles" points at an array of "num_elems" element handles.
  -# "vert_handles" points at an array of size "sizeof_vert_handles"
  -# "csr_data" points at an array of size "sizeof_csr_data"
  -# "csr_offsets" points at an array of size "num_elems+1"
      
  When this function returns, adjacency information will be stored
  in csr format:
  -# "vert_handles" stores handles to all vertices found in one
  or more of the elements.  Each vertex appears only
  once in "vert_handles", even if it is in multiple elements.
  -# "sizeof_vert_handles" is set to the number of vertex
  handles placed into "vert_handles".
  -# "sizeof_csr_data" is set to the total number of vertex uses (for
  example, sizeof_csr_data = 6 in the case of 2 TRIANGLES, even if
  the two triangles share some vertices).
  -# "csr_offsets" is filled such that csr_offset[i] indicates the location
  of entity i's first adjacency in "csr_data".  The number of vertices
  in element i is equal to csr_offsets[i+1] - csr_offsets[i].  For this
  reason, csr_offsets[num_elems] is set to the new value of
  "sizeof_csr_data".
  -# "csr_data" stores integer offsets which give the location of
  each adjacency in the "vert_handles" array.

  As an example of how to use this data, you can get the handle of the first
  vertex in element #3 like this:
  \code VertexHandle vh = vert_handles[ csr_data[ csr_offsets[3] ] ] \endcode

  and the second vertex of element #3 like this:
  \code VertexHandle vh = vert_handles[ csr_data[ csr_offsets[3]+1 ] ] \endcode
*/
template <class FIELD>
void MesquiteMesh<FIELD>::elements_get_attached_vertices(
  const Mesquite::Mesh::ElementHandle *elem_handles,
  size_t num_elems,
  vector<Mesquite::Mesh::VertexHandle>& vert_handles,
  vector<size_t> &offsets,
  Mesquite::MsqError &err)
{
  cout << "ERROR: MesquiteMesh::elements_get_attached_vertices has not been implemented." << endl;
  
//     // Check for zero element case.
//   vert_handles.clear();
//   offsets.clear();
  
//   if (num_elems == 0)
//   {
//     return;
//   }       
//   size_t i, j;
//   MeshEntity* element_ptr;
//   DLIList<CubitNode*> entity_list;
//     //get a list of all nodes that are in these elements (the elements
//     // in the list will not necessarily be unique).
//   size_t offset_counter = 0;
//   for(i=0; i < ((size_t) num_elems);++i){
//     offsets.push_back(offset_counter);
//     element_ptr=reinterpret_cast<MeshEntity*>(elem_handles[i]);
//     entity_list.clean_out();
//     element_ptr->nodes(entity_list);
//       //now set size_of_vert_handles to the value it should be at return time.
//     int entity_list_size=entity_list.size();
//     entity_list.reset();
//     VertexHandle temp_v_handle = NULL;
//       //loop over the vertices, to add them to the given array.
//     for(j=0;j<entity_list_size;++j){
//       temp_v_handle =
//         reinterpret_cast<VertexHandle>(entity_list.get_and_step());
      
//       if(temp_v_handle==NULL){
//         PRINT_ERROR("Unexpected null pointer.\n");
//         MSQ_SETERR(err)("Unexpected null pointer.",
//                         Mesquite::MsqError::INVALID_STATE);
//         return;
//       }
//       vert_handles.push_back(temp_v_handle);
//       ++offset_counter;
//     }
      
//   }
//   offsets.push_back(offset_counter);
}

//! Returns the topologies of the given entities.  The "entity_topologies"
//! array must be at least "num_elements" in size.
template <class FIELD>
void MesquiteMesh<FIELD>::elements_get_topologies(
  const ElementHandle *element_handle_array,
  Mesquite::EntityTopology *element_topologies,
  size_t num_elements,
  Mesquite::MsqError &err)
{
  cout << "ERROR: MesquiteMesh::elements_get_topologies has not been implemented." << endl;
  
//   MeshEntity *ent=NULL;
//     //loop over the elements
//   for ( ; num_elements--; )
//   {
//     ent= reinterpret_cast<MeshEntity*>(element_handle_array[num_elements]);
//       //add the appropriate EntityType to the element_topologies array
//     if(CAST_TO(ent,CubitTri))
//     {
//       element_topologies[num_elements]=Mesquite::TRIANGLE;
//     }
//     else if(CAST_TO(ent,CubitFace))
//     {
//       element_topologies[num_elements]=Mesquite::QUADRILATERAL;
//     }
//     else if(CAST_TO(ent,CubitTet))
//     {
//       element_topologies[num_elements]= Mesquite::TETRAHEDRON;
//     }
//     else if(CAST_TO(ent,CubitHex))
//     {
//       element_topologies[num_elements]=Mesquite::HEXAHEDRON;
//     }
//     else{
//       PRINT_ERROR("Type not recognized.\n");
//       MSQ_SETERR(err)("Type not recognized.", Mesquite::MsqError::UNSUPPORTED_ELEMENT);
//       return;
//     }
 
//   }//end loop over elements
}

//! Tells the mesh that the client is finished with a given
//! entity handle.  
template <class FIELD>
void MesquiteMesh<FIELD>::release_entity_handles(
  const EntityHandle */*handle_array*/,
  size_t /*num_handles*/,
  Mesquite::MsqError &/*err*/)
{
  cout << "WARNING: MesquiteMesh::release_entity_handles was called." << endl;
    // Do nothing...
}

  
//! Instead of deleting a Mesh when you think you are done,
//! call release().  In simple cases, the implementation could
//! just call the destructor.  More sophisticated implementations
//! may want to keep the Mesh object to live longer than Mesquite
//! is using it.
template <class FIELD>
void MesquiteMesh<FIELD>::release()
{
  cout << "WARNING: MesquiteMesh::release was called." << endl;
    // We allocate on the stack, so don't delete this...
//  delete this;
}

  //***************   Start of Iterator functions ******************


// ********* VertexIterator functions ********
//constructor
template <class FIELD>
MesquiteMesh<FIELD>::VertexIterator::VertexIterator(
  MesquiteMesh* mesh_ptr)
{
  meshPtr=mesh_ptr;
  restart();
}

//! Moves the iterator back to the first
//! entity in the list.
template <class FIELD>
void MesquiteMesh<FIELD>::VertexIterator::restart()
{
//   mIndex=0;
  mIndex = niterBegin;
}

//! *iterator.  Return the handle currently
//! being pointed at by the iterator.
template <class FIELD>
Mesquite::Mesh::EntityHandle MesquiteMesh<FIELD>::VertexIterator::operator*() const
{
  if(!is_at_end())
//     return reinterpret_cast<Mesquite::Mesh::EntityHandle>(meshPtr->get_node_array()[mIndex]);
      return *mIndex;
  return 0;
}


//! ++iterator
template <class FIELD>
void MesquiteMesh<FIELD>::VertexIterator::operator++()
{
  cout << "WARNING: indexing vertex_iterator." << endl;
  ++mIndex;
}

//! iterator++
template <class FIELD>
void MesquiteMesh<FIELD>::VertexIterator::operator++(int)
{
  ++mIndex;
}

//! Returns false until the iterator has
//! been advanced PAST the last entity.
//! Once is_at_end() returns true, *iterator
//! returns 0.
template <class FIELD>
bool MesquiteMesh<FIELD>::VertexIterator::is_at_end() const
{
  cout << "WARNING: at end of vertex_iterator." << endl;
//  if(mIndex >= meshPtr->get_num_nodes())
  if( mIndex == niterEnd )
    return true;
  return false;
}

// ********* ElementIterator functions ********
//constructor
template <class FIELD>
MesquiteMesh<FIELD>::ElementIterator::ElementIterator(MesquiteMesh* mesh_ptr)
{
  meshPtr=mesh_ptr;
  restart();
}

//! Moves the iterator back to the first
//! entity in the list.
template <class FIELD>
void MesquiteMesh<FIELD>::ElementIterator::restart()
{
//  mIndex=0;
  mIndex = eiterBegin;
}

//! *iterator.  Return the handle currently
//! being pointed at by the iterator.
template <class FIELD>
Mesquite::Mesh::EntityHandle MesquiteMesh<FIELD>::ElementIterator::operator*() const
{
  if(!is_at_end())
//     return reinterpret_cast<Mesquite::Mesh::EntityHandle>(meshPtr->get_element_array()[mIndex]);
      return *mIndex;
  return 0;
}

//! ++iterator
template <class FIELD>
void MesquiteMesh<FIELD>::ElementIterator::operator++()
{
  cout << "WARNING: indexing element_iterator." << endl;
  ++mIndex;
}

//! iterator++
template <class FIELD>
void MesquiteMesh<FIELD>::ElementIterator::operator++(int)
{
  ++mIndex;
}

//! Returns false until the iterator has
//! been advanced PAST the last entity.
//! Once is_at_end() returns true, *iterator
//! returns 0.
template <class FIELD>
bool MesquiteMesh<FIELD>::ElementIterator::is_at_end() const
{
  cout << "WARNING: at end of element_iterator." << endl;
//  if(mIndex>=meshPtr->get_num_elements())
  if( mIndex == eiterEnd )
    return true;
  return false;
}

}
