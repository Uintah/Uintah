#ifndef MESQUITE_MESH_HPP
#define MESQUITE_MESH_HPP

#include <Core/Datatypes/Field.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Dataflow/Modules/Fields/FieldBoundary.h>
#include <MeshInterface.hpp>
#include <MsqError.hpp>

namespace SCIRun {

template <class FIELD>
class MesquiteMesh : public Mesquite::Mesh
{
public:
  MesquiteMesh( FIELD* fieldh, ProgressReporter* mod );
  
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

  int get_num_nodes()
    {return numNodes;}
  int get_num_elements()
    {return numElements;}
 
  void update_progress()
      {
        double temp;
        count_++;
        if  ( count_%50 == 0 )
        {
          if( count_ > anticipated_iterations_ )
          {
            count_ = 3*anticipated_iterations_ / 4 ;
            temp = 0.75;
          }
          else
          {
            temp = (double)count_/(double)anticipated_iterations_;
          }
          
          update_window_->update_progress( temp );
        }   
      }
  
private:
  FIELD* mOwner;
  ProgressReporter* update_window_;
  int anticipated_iterations_;
  int count_;
  
  vector<unsigned char> bytes_;
  vector<bool> fixed_;
  
  typename FIELD::mesh_type::Node::iterator niterBegin;
  typename FIELD::mesh_type::Node::iterator niterEnd;
  typename FIELD::mesh_type::Elem::iterator eiterBegin;
  typename FIELD::mesh_type::Elem::iterator eiterEnd;

  typename FIELD::mesh_type::Node::size_type numNodes;
  typename FIELD::mesh_type::Elem::size_type numElements;

    //booleans about what element types we have
  bool triExists;
  bool quadExists;
  bool tetExists;
  bool hexExists;
  
  unsigned char myDimension;

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

    typedef typename FIELD::mesh_type::Node::iterator node_iter_t;   
    node_iter_t node_iter_;
    
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
    typedef typename FIELD::mesh_type::Elem::iterator elem_iter_t;   
    elem_iter_t elem_iter_;

    MesquiteMesh* meshPtr;
  };
};


  // MesquiteMesh constructor
template <class FIELD> 
MesquiteMesh<FIELD>::MesquiteMesh( FIELD* field, ProgressReporter* mod )
{
    //make sure that we are given a field
  if( !field )
  {
    return;
  }
  
  mOwner = field;
  update_window_ = mod;
  
    //get the nodes and elements of this field
  typename FIELD::mesh_handle_type mesh = field->get_typed_mesh();
  mesh->size( numNodes );
  mesh->size( numElements ); 
  mesh->begin(niterBegin); 
  mesh->end(niterEnd);
  mesh->begin(eiterBegin); 
  mesh->end(eiterEnd);

    //setup some needed information storage vectors for MESQUITE
  count_ = 0;
  anticipated_iterations_ = 3*numNodes;
  
  bytes_.resize(numNodes);
  fixed_.resize(numNodes);  
  size_t i;
  for(i = 0; i < ((size_t) numNodes); ++i )
  {
    bytes_[i] = 0;
    fixed_[i] = false;
    //typename FIELD::value_type v;
    //field->value(v, typename FIELD::mesh_type::Node::index_type(i));
    //fixed_[i] = (v > 0.5);
  }

  const TypeDescription *mtd = mOwner->mesh()->get_type_description();
  CompileInfoHandle ci_boundary = FieldBoundaryAlgo::get_compile_info( mtd );
  Handle<FieldBoundaryAlgo> boundary_algo;
  FieldHandle boundary_field_h;
  if( !DynamicCompilation::compile( ci_boundary, boundary_algo, false, mod ) ) return;
  
  MatrixHandle interp1(0);  
  boundary_algo->execute( mod, mOwner->mesh(), boundary_field_h, interp1, 1 );
  
  triExists = false;
  quadExists=false;
  tetExists=false;
  hexExists=false;
  if( mtd->get_name().find("TetVolMesh") != string::npos )
  {
    tetExists=true;
    myDimension = 3;
    
    TriSurfMesh<TriLinearLgn<Point> > *bound_mesh = dynamic_cast<TriSurfMesh<TriLinearLgn<Point> >*>(boundary_field_h->mesh().get_rep());
    TriSurfMesh<TriLinearLgn<Point> >::Node::iterator bound_iter;
    bound_mesh->begin( bound_iter );
    TriSurfMesh<TriLinearLgn<Point> >::Node::iterator bound_itere; 
    bound_mesh->end( bound_itere );
    while( bound_iter != bound_itere )
    {
      TriSurfMesh<TriLinearLgn<Point> >::Node::index_type bi = *bound_iter;
      ++bound_iter;
      
      int size, stride, *cols;
      double *vals;
      interp1->getRowNonzerosNoCopy( bi, size, stride, cols, vals );
      
      fixed_[*cols] = true;
    }
  }
  else if (mtd->get_name().find("TriSurfMesh") != string::npos)
  {
    triExists=true;
    myDimension = 2; 

    CurveMesh<CrvLinearLgn<Point> > *bound_mesh = dynamic_cast<CurveMesh<CrvLinearLgn<Point> >*>(boundary_field_h->mesh().get_rep());
    CurveMesh<CrvLinearLgn<Point> >::Node::iterator bound_iter; 
    bound_mesh->begin( bound_iter );
    CurveMesh<CrvLinearLgn<Point> >::Node::iterator bound_itere; 
    bound_mesh->end( bound_itere );
    while( bound_iter != bound_itere )
    {
      CurveMesh<CrvLinearLgn<Point> >::Node::index_type bi = *bound_iter;
      ++bound_iter;

      int size, stride, *cols;
      double *vals;
      interp1->getRowNonzerosNoCopy( bi, size, stride, cols, vals );

      fixed_[*cols] = true;
    }    
  }
  else if (mtd->get_name().find("HexVolMesh") != string::npos)
  {
    hexExists=true;
    myDimension = 3; 

    QuadSurfMesh<QuadBilinearLgn<Point> > *bound_mesh = dynamic_cast<QuadSurfMesh<QuadBilinearLgn<Point> >*>(boundary_field_h->mesh().get_rep());
    QuadSurfMesh<QuadBilinearLgn<Point> >::Node::iterator bound_iter; 
    bound_mesh->begin( bound_iter );
    QuadSurfMesh<QuadBilinearLgn<Point> >::Node::iterator bound_itere; 
    bound_mesh->end( bound_itere );
    while( bound_iter != bound_itere )
    {
      QuadSurfMesh<QuadBilinearLgn<Point> >::Node::index_type bi = *bound_iter;
      ++bound_iter;

      int size, stride, *cols;
      double *vals;
      interp1->getRowNonzerosNoCopy( bi, size, stride, cols, vals );

      fixed_[*cols] = true;
    }    
  }
  else if (mtd->get_name().find("QuadSurfMesh") != string::npos)
  {
    quadExists=true;
    myDimension = 2;

    CurveMesh<CrvLinearLgn<Point> > *bound_mesh = dynamic_cast<CurveMesh<CrvLinearLgn<Point> >*>(boundary_field_h->mesh().get_rep());
    CurveMesh<CrvLinearLgn<Point> >::Node::iterator bound_iter; 
    bound_mesh->begin( bound_iter );
    CurveMesh<CrvLinearLgn<Point> >::Node::iterator bound_itere; 
    bound_mesh->end( bound_itere );
    while( bound_iter != bound_itere )
    {
      CurveMesh<CrvLinearLgn<Point> >::Node::index_type bi = *bound_iter;
      ++bound_iter;

      int size, stride, *cols;
      double *vals;
      interp1->getRowNonzerosNoCopy( bi, size, stride, cols, vals );

      fixed_[*cols] = true;
    }    
  }
  else
  {
    return;
  }
}


  //destructor
template <class FIELD>
MesquiteMesh<FIELD>::~MesquiteMesh()
{
}

  
  /*! We always pass in nodes in with three coordinates.  This may change
    in the future if we want to do smoothing in a parametric space, but
    !for now, we are always in three-dimensions. */
template <class FIELD>
int
MesquiteMesh<FIELD>::get_geometric_dimension(Mesquite::MsqError &/*err*/)
{
  return 3;
}


  //! Returns the number of verticies for the entity.
template <class FIELD>
size_t
MesquiteMesh<FIELD>::get_total_vertex_count(Mesquite::MsqError &/*err*/) const
{
  return (size_t) numNodes;
}

  
  //! Returns the number of elements for the entity.
template <class FIELD>
size_t
MesquiteMesh<FIELD>::get_total_element_count(Mesquite::MsqError &/*err*/) const
{
  return (size_t) numElements;
}


  //! Fills array with handles to all vertices in the mesh.
template <class FIELD>
void
MesquiteMesh<FIELD>::get_all_vertices( vector<Mesquite::Mesh::VertexHandle> &vertices,
                                       Mesquite::MsqError &/*err*/)
{
  vertices.clear();
  typename FIELD::mesh_type::Node::iterator node_iter = niterBegin;

  while( node_iter != niterEnd )
  {
    unsigned int temp_id = *node_iter;
    VertexHandle temp_v_handle = (void*)temp_id;
    
    vertices.push_back( temp_v_handle );
    ++node_iter;
  }
}


  //! Fills array with handles to all elements in the mesh.
template <class FIELD>
void
MesquiteMesh<FIELD>::get_all_elements( vector<Mesquite::Mesh::ElementHandle> &elements,      
                                       Mesquite::MsqError &/*err*/ )
{
  elements.clear();
  typename FIELD::mesh_type::Elem::iterator elem_iter = eiterBegin;  

  while( elem_iter != eiterEnd )
  {
    unsigned int temp_id = *elem_iter;
    ElementHandle temp_e_handle = (void*)temp_id;
    
    elements.push_back( temp_e_handle );
    ++elem_iter;
  }
}

  
//! Returns a pointer to an iterator that iterates over the
//! set of all vertices in this mesh.  The calling code should
//! delete the returned iterator when it is finished with it.
//! If vertices are added or removed from the Mesh after obtaining
//! an iterator, the behavior of that iterator is undefined.
template <class FIELD>
Mesquite::VertexIterator*
MesquiteMesh<FIELD>::vertex_iterator(Mesquite::MsqError &/*err*/)
{
  return new typename MesquiteMesh<FIELD>::VertexIterator(this);
}

  
//! Returns a pointer to an iterator that iterates over the
//! set of all top-level elements in this mesh.  The calling code should
//! delete the returned iterator when it is finished with it.
//! If elements are added or removed from the Mesh after obtaining
//! an iterator, the behavior of that iterator is undefined.
template <class FIELD>
Mesquite::ElementIterator*
MesquiteMesh<FIELD>::element_iterator(Mesquite::MsqError &/*err*/)
{
  return new typename MesquiteMesh<FIELD>::ElementIterator(this);
}


//! Returns true or false, indicating whether the vertex
//! is allowed to moved.
//! Note that this is a read-only
//! property; this flag can't be modified by users of the
//! Mesquite::Mesh interface.
template <class FIELD>
void
MesquiteMesh<FIELD>::vertices_get_fixed_flag(
  const Mesquite::Mesh::VertexHandle vert_array[], 
  bool fixed_flag_array[],
  size_t num_vtx, 
  Mesquite::MsqError &err )
{
  size_t i;
  for( i = 0; i < num_vtx; ++i )
  {
    unsigned long node_id = (unsigned long)vert_array[i];
    fixed_flag_array[i] = fixed_[node_id];
  }
}

  
//! Get location of a vertex
template <class FIELD>
void
MesquiteMesh<FIELD>::vertices_get_coordinates(
    const Mesquite::Mesh::VertexHandle vert_array[],
    Mesquite::MsqVertex* coordinates,
    size_t num_vtx,
    Mesquite::MsqError &err)
{
  size_t i;
  for( i = 0; i < num_vtx; ++i )
  {
      //set coordinates to the vertex's position.
    Point p;
    typename FIELD::mesh_type::Node::index_type node_id = (unsigned long) vert_array[i];
    mOwner->get_typed_mesh()->get_point( p, node_id );    
    coordinates[i].set( p.x(), p.y(), p.z() );
  }
}

  
//! Set the location of a vertex.
template <class FIELD>
void
MesquiteMesh<FIELD>::vertex_set_coordinates(
    VertexHandle vertex,
    const Mesquite::Vector3D &coordinates,
    Mesquite::MsqError &err)
{
  Point p;
  typename FIELD::mesh_type::Node::index_type node_id = (unsigned long) vertex;
  mOwner->get_typed_mesh()->get_point( p, node_id );
  p.x( coordinates[0] );
  p.y( coordinates[1] );
  p.z( coordinates[2] );
  mOwner->get_typed_mesh()->set_point( p, node_id );
  
  update_progress();
}


//! Each vertex has a byte-sized flag that can be used to store
//! flags.  This byte's value is neither set nor used by the mesh
//! implementation.  It is intended to be used by Mesquite algorithms.
//! Until a vertex's byte has been explicitly set, its value is 0.
//! Cubit stores the byte on the TDMesquiteFlag associated with the
//! node.
template <class FIELD>
void
MesquiteMesh<FIELD>::vertex_set_byte( VertexHandle vertex,
                                      unsigned char byte,
                                      Mesquite::MsqError &err)
{
  unsigned long idx = (unsigned long)vertex;
  bytes_[idx] = byte;
}


//! Set the byte for a given array of vertices.
template <class FIELD>
void
MesquiteMesh<FIELD>::vertices_set_byte (
  const VertexHandle *vert_array,
  const unsigned char *byte_array,
  size_t array_size,
  Mesquite::MsqError &err)
{
    //loop over the given vertices and call vertex_set_byte(...).
  size_t i = 0;
  for( i = 0; i < array_size; ++i )
  {
    vertex_set_byte( vert_array[i], byte_array[i], err );
  }
}

  
//! Retrieve the byte value for the specified vertex or vertices.
//! The byte value is 0 if it has not yet been set via one of the
//! *_set_byte() functions.
template <class FIELD>
void
MesquiteMesh<FIELD>::vertex_get_byte(VertexHandle vertex,
                                     unsigned char *byte,
                                     Mesquite::MsqError &err)
{
  unsigned long idx = (unsigned long)vertex;
  *byte = bytes_[idx];
}


//! get the bytes associated with the vertices in a given array.
template <class FIELD>
void
MesquiteMesh<FIELD>::vertices_get_byte(
  const VertexHandle *vertex_array,
  unsigned char *byte_array,
  size_t array_size,
  Mesquite::MsqError &err)
{
    //loop over the given nodes and call vertex_get_byte(...)
  size_t i = 0;
  for( i = 0; i < array_size; ++i )
  {
    vertex_get_byte( vertex_array[i], &byte_array[i], err );
  }
}

  
//! Gets the elements attached to this vertex.
template <class FIELD>
void
MesquiteMesh<FIELD>::vertices_get_attached_elements(
    const VertexHandle* vertex_array,
    size_t num_vertex,
    msq_std::vector<ElementHandle>& elements,
    msq_std::vector<size_t>& offsets,
    Mesquite::MsqError& err )
{
  mOwner->get_typed_mesh()->synchronize(SCIRun::Mesh::ALL_ELEMENTS_E | 
                                        SCIRun::Mesh::NODE_NEIGHBORS_E);

  size_t i = 0, j = 0;
  size_t offset_counter = 0;
  ElementHandle temp_e_handle;

  elements.clear();
  offsets.clear();
  
  for( i = 0; i < num_vertex; ++i )
  {
    offsets.push_back(offset_counter);
    typename FIELD::mesh_type::Elem::array_type attached_elems;
    typename FIELD::mesh_type::Node::index_type this_node = (unsigned long)vertex_array[i];

    mOwner->get_typed_mesh()->get_elems( attached_elems, this_node );
    for( j = 0; j < attached_elems.size(); ++j ) 
    {
      unsigned int temp_id = attached_elems[j];
      temp_e_handle = (void*)temp_id;
      
      elements.push_back(temp_e_handle);
      ++offset_counter;
    }
  }
  offsets.push_back( offset_counter );
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
void
MesquiteMesh<FIELD>::elements_get_attached_vertices(
  const Mesquite::Mesh::ElementHandle *elem_handles,
  size_t num_elems,
  vector<Mesquite::Mesh::VertexHandle>& vert_handles,
  vector<size_t> &offsets,
  Mesquite::MsqError &err)
{
  mOwner->get_typed_mesh()->synchronize(SCIRun::Mesh::ALL_ELEMENTS_E);

  vert_handles.clear();
  offsets.clear();

    // Check for zero element case.  
  if( num_elems == 0 )
  {
    return;
  }   
    
    //get a list of all nodes that are in these elements (the elements
    // in the list will not necessarily be unique).
  size_t i, j;
  size_t offset_counter = 0;
  for( i = 0; i < ((size_t) num_elems); ++i )
  {
    offsets.push_back( offset_counter );
    
    typename FIELD::mesh_type::Node::array_type nodes;
    typename FIELD::mesh_type::Elem::index_type elem_id = (unsigned long)elem_handles[i];
       
    mOwner->get_typed_mesh()->get_nodes( nodes, elem_id );
    VertexHandle temp_v_handle = NULL;
      //loop over the vertices, to add them to the given array.
    for( j = 0; j < nodes.size(); ++j )
    {
      unsigned int temp_id = nodes[j];
      temp_v_handle = (void*)temp_id;

      vert_handles.push_back( temp_v_handle );
      ++offset_counter;
    }
  }
  offsets.push_back(offset_counter);
}


//! Returns the topologies of the given entities.  The "entity_topologies"
//! array must be at least "num_elements" in size.
template <class FIELD>
void
MesquiteMesh<FIELD>::elements_get_topologies(
  const ElementHandle *element_handle_array,
  Mesquite::EntityTopology *element_topologies,
  size_t num_elements,
  Mesquite::MsqError &err)
{
    //NOTE: this function assumes a homogenous mesh type for the entire mesh.
    //  If hybrid mesh types are allowed, this function will need to be 
    //  modified to remove this assumption...
  
  for ( ; num_elements--; )
  {
    if( tetExists )
    {
      element_topologies[num_elements] = Mesquite::TETRAHEDRON;
    }
    else if( hexExists )
    {
      element_topologies[num_elements] = Mesquite::HEXAHEDRON;
    }
    else if( quadExists )
    {
      element_topologies[num_elements] = Mesquite::QUADRILATERAL;
    }
    else if( triExists )
    {
      element_topologies[num_elements] = Mesquite::TRIANGLE;
    }
    else
    {
      MSQ_SETERR(err)("Type not recognized.", Mesquite::MsqError::UNSUPPORTED_ELEMENT);
      return;
    }
  }
}


//! Tells the mesh that the client is finished with a given
//! entity handle.  
template <class FIELD>
void
MesquiteMesh<FIELD>::release_entity_handles(
  const EntityHandle */*handle_array*/,
  size_t /*num_handles*/,
  Mesquite::MsqError &/*err*/)
{
    // Do nothing...
}
  

//! Instead of deleting a Mesh when you think you are done,
//! call release().  In simple cases, the implementation could
//! just call the destructor.  More sophisticated implementations
//! may want to keep the Mesh object to live longer than Mesquite
//! is using it.
template <class FIELD>
void
MesquiteMesh<FIELD>::release()
{
    // We allocate on the stack, so don't delete this...
}

  //***************   Start of Iterator functions ******************

// ********* VertexIterator functions ********
//constructor
template <class FIELD>
MesquiteMesh<FIELD>::VertexIterator::VertexIterator( MesquiteMesh* mesh_ptr )
{
  meshPtr = mesh_ptr;
  restart();
}


//! Moves the iterator back to the first entity in the list.
template <class FIELD>
void
MesquiteMesh<FIELD>::VertexIterator::restart()
{
  node_iter_ = meshPtr->niterBegin;
}

        
//! *iterator. Return the handle currently being pointed at by the iterator.
template <class FIELD>
Mesquite::Mesh::EntityHandle
MesquiteMesh<FIELD>::VertexIterator::operator*() const
{
  node_iter_t ni = node_iter_;
  unsigned int i = *ni;  
  void *p = (void*)i; 
  if(!is_at_end())
      return reinterpret_cast<Mesquite::Mesh::EntityHandle>(p);
  return 0;
}


//! ++iterator
template <class FIELD>
void
MesquiteMesh<FIELD>::VertexIterator::operator++()
{
  ++node_iter_;
}


//! iterator++
template <class FIELD>
void
MesquiteMesh<FIELD>::VertexIterator::operator++(int)
{
  ++node_iter_;
}


//! Returns false until the iterator has been advanced PAST the last entity.
//! Once is_at_end() returns true, *iterator returns 0.
template <class FIELD>
bool
MesquiteMesh<FIELD>::VertexIterator::is_at_end() const
{
  if( node_iter_ == meshPtr->niterEnd )
  {
    return true;
  }
  return false;
}


// ********* ElementIterator functions ********
//constructor
template <class FIELD>
MesquiteMesh<FIELD>::ElementIterator::ElementIterator( MesquiteMesh* mesh_ptr )
{
  meshPtr=mesh_ptr;
  restart();
}


//! Moves the iterator back to the first entity in the list.
template <class FIELD>
void
MesquiteMesh<FIELD>::ElementIterator::restart()
{
  elem_iter_ = meshPtr->eiterBegin;
}


//! *iterator.  Return the handle currently being pointed at by the iterator.
template <class FIELD>
Mesquite::Mesh::EntityHandle
MesquiteMesh<FIELD>::ElementIterator::operator*() const
{
  elem_iter_t ei = elem_iter_;
  unsigned int i = *ei;  
  void *p = (void*)i; 

  if(!is_at_end())
      return reinterpret_cast<Mesquite::Mesh::EntityHandle>(p);
  return 0;
}


//! ++iterator
template <class FIELD>
void
MesquiteMesh<FIELD>::ElementIterator::operator++()
{
  ++elem_iter_;
}


//! iterator++
template <class FIELD>
void
MesquiteMesh<FIELD>::ElementIterator::operator++(int)
{
  ++elem_iter_;
}


//! Returns false until the iterator has been advanced PAST the last entity.
//! Once is_at_end() returns true, *iterator returns 0.
template <class FIELD>
bool
MesquiteMesh<FIELD>::ElementIterator::is_at_end() const
{
  if( elem_iter_ == meshPtr->eiterEnd )
  {
    return true;
  }
  return false;
}
    
} // end namespace SCIRun

#endif //has file been included
