#ifndef MESQUITE_MESH_HPP
#define MESQUITE_MESH_HPP

#include <Core/Datatypes/Field.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/HexVolMesh.h>
#include "/home/sci/jfsheph/Mesquite-and-Verdict/mesquite-1.1.3/include/MeshInterface.hpp"
#include "/home/sci/jfsheph/Mesquite-and-Verdict/mesquite-1.1.3/include/MsqError.hpp"

namespace SCIRun {

template <class FIELD>
class MesquiteMesh : public Mesquite::Mesh
{
public:
  MesquiteMesh( FIELD* fieldh );
  
public:
  virtual ~MesquiteMesh();
  
//************ Operations on entire mesh ****************
    //! Returns whether this mesh lies in a 2D or 3D coordinate system.
  virtual int get_geometric_dimension( Mesquite::MsqError &err );
  
    //! Returns the number of entities of the indicated type.
  virtual size_t get_total_vertex_count( Mesquite::MsqError &err ) const;
  virtual size_t get_total_element_count( Mesquite::MsqError &err ) const;
  
    //! Fills vector with handles to all vertices/elements in the mesh.
  virtual void get_all_vertices(
      vector<Mesquite::Mesh::VertexHandle> &vertices,
    Mesquite::MsqError &err);
  
  virtual void get_all_elements(
    vector<Mesquite::Mesh::ElementHandle> &elements,
    Mesquite::MsqError &err);
  
    //! Returns a pointer to an iterator that iterates over the
    //! set of all vertices in this mesh.  The calling code should
    //! delete the returned iterator when it is finished with it.
    //! If vertices are added or removed from the Mesh after obtaining
    //! an iterator, the behavior of that iterator is undefined.
  virtual Mesquite::VertexIterator* vertex_iterator(
    Mesquite::MsqError &err);
  
    //! Returns a pointer to an iterator that iterates over the
    //! set of all top-level elements in this mesh.  The calling code should
    //! delete the returned iterator when it is finished with it.
    //! If elements are added or removed from the Mesh after obtaining
    //! an iterator, the behavior of that iterator is undefined.
  virtual Mesquite::ElementIterator* element_iterator(
    Mesquite::MsqError &err);
  
  
  virtual void vertices_get_fixed_flag(
    const Mesquite::Mesh::VertexHandle vert_array[], 
    bool fixed_flag_array[],
    size_t num_vtx, 
    Mesquite::MsqError &err );
  
    //! Get/set location of a vertex (vertices)
  virtual void vertices_get_coordinates(
    const Mesquite::Mesh::VertexHandle vert_array[],
    Mesquite::MsqVertex* coordinates,
    size_t num_vtx,
    Mesquite::MsqError &err);
  
  virtual void vertex_set_coordinates(
    Mesquite::Mesh::VertexHandle vertex,
    const Mesquite::Vector3D &coordinates,
    Mesquite::MsqError &err);
  
    //! Each vertex has a byte-sized flag that can be used to store
    //! flags.  This byte's value is neither set nor used by the mesh
    //! implementation.  It is intended to be used by Mesquite algorithms.
    //! Until a vertex's byte has been explicitly set, its value is 0.
  virtual void vertex_set_byte (
    Mesquite::Mesh::VertexHandle vertex,
    unsigned char byte,
    Mesquite::MsqError &err);
  
  virtual void vertices_set_byte (
    const Mesquite::Mesh::VertexHandle *vert_array,
    const unsigned char *byte_array,
    size_t array_size,
    Mesquite::MsqError &err);
  
    //! Retrieve the byte value for the specified vertex or vertices.
    //! The byte value is 0 if it has not yet been set via one of the
    //! *_set_byte() functions.
  virtual void vertex_get_byte(
    const Mesquite::Mesh::VertexHandle vertex,
    unsigned char *byte,
    Mesquite::MsqError &err);
  
  virtual void vertices_get_byte(
    const Mesquite::Mesh::VertexHandle *vertex_array,
    unsigned char *byte_array,
    size_t array_size,
    Mesquite::MsqError &err);
  
// //**************** Vertex Topology *****************    
//     //! Gets the number of elements attached to this vertex.
//     //! Useful to determine how large the "elem_array" parameter
//     //! of the vertex_get_attached_elements() function must be.
    //! Gets the elements attached to this vertex.
  
  virtual void vertices_get_attached_elements(
    const Mesquite::Mesh::VertexHandle* vertex_array,
    size_t num_vertex,
    vector<Mesquite::Mesh::ElementHandle>& elements,
    vector<size_t> &offsets,
    Mesquite::MsqError &err);
 
  virtual void elements_get_attached_vertices(
    const Mesquite::Mesh::ElementHandle *elem_handles,
    size_t num_elems,
    vector<Mesquite::Mesh::VertexHandle>& vert_handles,
    vector<size_t> &offsets,
    Mesquite::MsqError &err);
  
    //! Returns the topologies of the given entities.  The "entity_topologies"
    //! array must be at least "num_elements" in size.
  virtual void elements_get_topologies(
    const Mesquite::Mesh::ElementHandle *element_handle_array,
    Mesquite::EntityTopology *element_topologies,
    size_t num_elements,
    Mesquite::MsqError &err);


    //TAGs (not implemented yet)
  virtual Mesquite::TagHandle tag_create(
    const string& /*tag_name*/,
    Mesquite::Mesh::TagType /*type*/,
    unsigned /*length*/,
    const void* /*default_value*/,
    Mesquite::MsqError &err)
    {MSQ_SETERR(err)("Function not yet implemented.\n",Mesquite::MsqError::NOT_IMPLEMENTED); return NULL;}
  
  virtual void tag_delete(
    Mesquite::TagHandle /*handle*/,
    Mesquite::MsqError& err ) 
    {MSQ_SETERR(err)("Function not yet implemented.\n",Mesquite::MsqError::NOT_IMPLEMENTED);}
  
  virtual Mesquite::TagHandle tag_get(
    const string& /*name*/, 
    Mesquite::MsqError& err ){
    MSQ_SETERR(err)("Function not yet implemented.\n",Mesquite::MsqError::NOT_IMPLEMENTED); return NULL;}
  
  virtual void tag_properties(
    Mesquite::TagHandle /*handle*/,
    string& /*name_out*/,
    Mesquite::Mesh::TagType& /*type_out*/,
    unsigned& /*length_out*/,
    Mesquite::MsqError& err )
    {MSQ_SETERR(err)("Function not yet implemented.\n",Mesquite::MsqError::NOT_IMPLEMENTED);}

  virtual void tag_set_element_data(
    Mesquite::TagHandle /*handle*/,
    size_t /*num_elems*/,
    const Mesquite::Mesh::ElementHandle* /*elem_array*/,
    const void* /*tag_data*/,
    Mesquite::MsqError& err )
    {MSQ_SETERR(err)("Function not yet implemented.\n",Mesquite::MsqError::NOT_IMPLEMENTED);}

   virtual void tag_set_vertex_data (
     Mesquite::TagHandle /*handle*/,
     size_t /*num_elems*/,
     const Mesquite::Mesh::VertexHandle* /*node_array*/,
     const void* /*tag_data*/,
     Mesquite::MsqError& err )
    {MSQ_SETERR(err)("Function not yet implemented.\n",Mesquite::MsqError::NOT_IMPLEMENTED);}
  
  virtual void tag_get_element_data(
    Mesquite::TagHandle /*handle*/,
    size_t /*num_elems*/,
    const Mesquite::Mesh::ElementHandle* /*elem_array*/,
    void* /*tag_data*/,
    Mesquite::MsqError& err )
    {MSQ_SETERR(err)("Function not yet implemented.\n",Mesquite::MsqError::NOT_IMPLEMENTED);}
  
   virtual void tag_get_vertex_data (
     Mesquite::TagHandle /*handle*/,
     size_t /*num_elems*/,
     const Mesquite::Mesh::VertexHandle* /*node_array*/,
     void* /*tag_data*/,
     Mesquite::MsqError& err )
    {MSQ_SETERR(err)("Function not yet implemented.\n",Mesquite::MsqError::NOT_IMPLEMENTED);}
  
    //END TAGS (NOT IMPLEMENTED YET)
  
//**************** Memory Management ****************
    //! Tells the mesh that the client is finished with a given
    //! entity handle.  
  virtual void release_entity_handles(
    const Mesquite::Mesh::EntityHandle *handle_array,
    size_t num_handles,
    Mesquite::MsqError &err);
  
    //! Instead of deleting a Mesh when you think you are done,
    //! call release().  In simple cases, the implementation could
    //! just call the destructor.  More sophisticated implementations
    //! may want to keep the Mesh object to live longer than Mesquite
    //! is using it.
  virtual void release();

//   CubitNode** get_node_array()
//     {return nodeArray;}
//   MeshEntity** get_element_array()
//     {return elementArray;}
  int get_num_nodes()
    {return numNodes;}
  int get_num_elements()
    {return numElements;}
  
private:
  FieldHandle* mOwner;
  
//   DLIList<MeshEntity*>* cachedEntityList;
//   CubitNode** cachedNode;
//     //DLIList<MeshEntity*> elementList;
 
  typename FIELD::mesh_type::Node::iterator niterBegin;
  typename FIELD::mesh_type::Node::iterator niterEnd;
  typename FIELD::mesh_type::Node::iterator niterCurrent;
  typename FIELD::mesh_type::Elem::iterator eiterBegin;
  typename FIELD::mesh_type::Elem::iterator eiterEnd;
  typename FIELD::mesh_type::Elem::iterator eiterCurrent;
  typename FIELD::mesh_type::Node::size_type numNodes;
  typename FIELD::mesh_type::Elem::size_type numElements;
//   CubitNode** nodeArray;
//   MeshEntity** elementArray;
//   int numNodes;
//   int numElements;

    //booleans about what element types we have
  bool triExists;
  bool quadExists;
  bool tetExists;
  bool hexExists;
  
  unsigned char myDimension;

    //This is function to cache the elements attached to a vertex.
//   inline CubitBoolean cache_elements_attached_to_node(CubitNode* node) const;
  
    // Iterator definitions
  class VertexIterator : public Mesquite::EntityIterator
  {
  public:
    VertexIterator (MesquiteMesh* mesh_ptr);
    
    virtual ~VertexIterator()
      {}
    
      //! Moves the iterator back to the first
      //! entity in the list.
    virtual void restart();
    
      //! *iterator.  Return the handle currently
      //! being pointed at by the iterator.
    virtual Mesquite::Mesh::EntityHandle operator*() const;
    
      //! ++iterator
    virtual void operator++();
      //! iterator++
    virtual void operator++(int);
    
      //! Returns false until the iterator has
      //! been advanced PAST the last entity.
      //! Once is_at_end() returns true, *iterator
      //! returns 0.
    virtual bool is_at_end() const;

  private:
    int mIndex;
    MesquiteMesh* meshPtr;
  };
  
  class ElementIterator : public Mesquite::EntityIterator
  {
  public:
    ElementIterator(MesquiteMesh* mesh_ptr);
    
    virtual ~ElementIterator()
      {}
    
      //! Moves the iterator back to the first
      //! entity in the list.
    virtual void restart();
    
      //! *iterator.  Return the handle currently
      //! being pointed at by the iterator.
    virtual Mesquite::Mesh::EntityHandle operator*() const;
    
      //! ++iterator
    virtual void operator++();
      //! iterator++
    virtual void operator++(int);
    
      //! Returns false until the iterator has
      //! been advanced PAST the last entity.
      //! Once is_at_end() returns true, *iterator
      //! returns 0.
    virtual bool is_at_end() const;

  private:
    int mIndex;
    MesquiteMesh* meshPtr;
  };
};


// //!Given a node, we cache the elements attached to it in cachedEntityList.
// CubitBoolean MesquiteMesh::cache_elements_attached_to_node(
//   CubitNode* node) const
// {
//       //if the node wasn't the last node cached, cache it.
//     if(node!= *cachedNode){
//         //clean out the old list and set the cached node 
//       cachedEntityList->clean_out();
//       *cachedNode=node;
//       DLIList<MeshEntity*> entity_list;
//       DLIList<CubitTri*> tri_list;
//       DLIList<CubitFace*> face_list;
//       DLIList<CubitTet*> tet_list;
//       DLIList<CubitHex*> hex_list;
//       MeshEntity* mesh_ent=NULL;
//       int i = 0;
//         //Michael New comment needed :
//       switch(myDimension){
//         case(2):
//           if(triExists){
//             node->tris(tri_list);
//             for(i=tri_list.size(); i-- ; ){
//               mesh_ent=tri_list.get_and_step();
//               if(mesh_ent!=NULL)
//                 entity_list.append(mesh_ent);
//               else
//                 PRINT_ERROR("Problem caching elements\n");
//             }
//           }
//           if(quadExists){
//             node->faces(face_list);
//             for(i=face_list.size(); i-- ; ){
//               mesh_ent=face_list.get_and_step();
//               if(mesh_ent!=NULL)
//                 entity_list.append(mesh_ent);
//               else
//                 PRINT_ERROR("Problem caching elements\n");
//             }
//           }
//           break;
//         case(3):
//           if(tetExists){
//             node->all_tets(tet_list);
//             for(i=tet_list.size(); i-- ; ){
//               mesh_ent=tet_list.get_and_step();
//               if(mesh_ent!=NULL)
//                 entity_list.append(mesh_ent);
//               else
//                 PRINT_ERROR("Problem caching elements\n");
//             }
//           }
//           if(hexExists){
//             node->all_hexes(hex_list);
//             for(i=hex_list.size(); i-- ; ){
//               mesh_ent=hex_list.get_and_step();
//               if(mesh_ent!=NULL)
//                 entity_list.append(mesh_ent);
//               else
//                 PRINT_ERROR("Problem caching elements\n");
//             }
//           }
//           break;
//         default:
//           PRINT_ERROR("MesquiteMesh::cache_elements_attached_to_node: invalid dimension.\n");
//           return CUBIT_FAILURE;
//       }
//       int list_size=entity_list.size();
      
//       for(i=0;i<list_size;++i){
//         if(entity_list.get()->owner()==mOwner)
//           cachedEntityList->append(entity_list.get());
//         entity_list.step();
//       }
//     }
//       //if we make it this far, we assume the function was successful
//     return CUBIT_SUCCESS;
// }

}

#endif //has file been included
