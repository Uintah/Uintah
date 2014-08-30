/*
 * Copyright (c) 2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 * ----------------------------------------------------------------------------
 * Available debugging flags:
 *
 * 	DEBUG_SF_ALL -- Enable all Spatial Field debugging flags.
 *
 */
//#define DEBUG_SF_ALL

#ifndef SpatialOps_SpatialField_h
#define SpatialOps_SpatialField_h

#include <stdexcept>
#include <sstream>

// Boost includes //
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

#include <spatialops/SpatialOpsConfigure.h>
#include <spatialops/structured/FVStaggeredLocationTypes.h>
#include <spatialops/structured/FieldInfo.h>

/**
 * \file SpatialField.h
 */

namespace SpatialOps{

  /**
   *  \class SingleValueCheck
   *
   *  \brief Abstracts checking if a SingleValueField is correctly sized
   *
   *  \tparam FieldLocation - type traits to describe the location of
   *    this field.  On staggered meshes, this will describe the mesh
   *    this field is associated with.  It also defines whether this
   *    field is on a volume or surface.
   *
   *  \tparam Location type traits to describe the location of field. For
   *          SingleValueCheck, only the type traits, SingleValue, matter.
   *
   */
  template<typename Location>
  struct SingleValueCheck {
    static inline void check( MemoryWindow const & window,
                              GhostData const & ghosts ) {}
  };

  template<>
  struct SingleValueCheck<SingleValue> {
    static inline void check( MemoryWindow const & window,
                              GhostData const & ghosts )
    {
#     ifndef NDEBUG
      if( (window.extent(0) > 1) &&
          (window.extent(1) > 1) &&
          (window.extent(2) > 1) )
      {
        std::ostringstream msg;
        msg << "Single Value Field does not support window extents larger than 1\n"
            << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
        throw(std::runtime_error(msg.str()));
      }

      if( (ghosts.get_minus(0) > 0) &&
          (ghosts.get_minus(1) > 0) &&
          (ghosts.get_minus(2) > 0) &&
          (ghosts.get_plus (0) > 0) &&
          (ghosts.get_plus (1) > 0) &&
          (ghosts.get_plus (2) > 0) )
      {
        std::ostringstream msg;
        msg << "Single Value Field does not support non-zero ghosts\n"
            << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
        throw(std::runtime_error(msg.str()));
      }
#     endif
    }
  };

//------------------------------------------------------------------

  /**
   *  \class SpatialField
   *  \ingroup fields
   *
   *  \brief Abstracts a field.
   *
   *  \tparam FieldLocation type traits to describe the location of
   *    this field.  On staggered meshes, this will describe the mesh
   *    this field is associated with.  It also defines whether this
   *    field is on a volume or surface.
   *
   *  \tparam T the underlying datatype (defaults to \c double)
   *
   *  \par Related classes:
   *   - \ref MemoryWindow
   *   - \ref FieldInfo
   *   - \ref SpatialFieldStore
   *
   *  \par Public Typedefs
   *   - \c field_type - this field's type
   *   - \c Location - the location type traits
   *   - \c value_type  - the type of underlying data being stored in this SpatialField
   *   - \c iterator, \c const_iterator - iterators to the elements in this field
   *   - \c iterator, \c const_iterator - iterators to the interior elements in this field (excludes ghost cells).
   */
  template< typename FieldLocation, typename T=double >
  class SpatialField {
  private:
    bool matchGlobalWindow_;                ///< True if using FieldInfo's window, boundary cell info and ghost info. When true, updates to ghost cells happen globally.
    MemoryWindow localWindow_;              ///< Window for this copy of the field, other copies can have different windows
    BoundaryCellInfo bcInfo_;               ///< Information about boundary conditions for current window
    GhostData localValidGhosts_;            ///< Information about ghosts for current window
    boost::shared_ptr<FieldInfo<T> > info_; ///< A shared pointer for FieldInfo for this field (all copies share the same FieldInfo)

  public:
    typedef SpatialField<FieldLocation,T> field_type;      ///< this field's type
    typedef FieldLocation Location;                        ///< this field's location
    typedef T value_type;                                  ///< the underlying value type for this field (nominally double)
    typedef MemoryWindow memory_window;                    ///< the window type for this field
    typedef FieldIterator<field_type> iterator;            ///< the iterator type
    typedef ConstFieldIterator<field_type> const_iterator; ///< const iterator type

    /**
     *  \brief SpatialField constructor
     *
     * \param window      MemoryWindow for the entire field (including ghost cells)
     * \param bc          BoundaryConditionInfo for field
     * \param ghosts      GhostData for entire field (all possible ghosts)
     * \param fieldValues pointer to memory for ExternalStorage mode (default: NULL)
     * \param mode        either InternalStorage or ExternalStorage (default: InternalStorage)
     * \param devIdx      device index of originally active device (default: CPU_INDEX)
     */
    inline SpatialField( const MemoryWindow window,
                         const BoundaryCellInfo bc,
                         const GhostData ghosts,
                         T* const fieldValues = NULL,
                         const StorageMode mode = InternalStorage,
                         const short int devIdx = CPU_INDEX )
    : matchGlobalWindow_( true ),
      localWindow_( window ),
      bcInfo_( bc.limit_by_extent(window.extent()) ),
      localValidGhosts_( ghosts.limit_by_extent(window.extent()) ),
      info_( new FieldInfo<T>( window, bc, ghosts, fieldValues, mode, devIdx ) )
    {
      SingleValueCheck<Location>::check( localWindow_, ghosts );
#     ifndef NDEBUG
      // ensure that we have a consistent BoundaryCellInfo object
      for( int i=0; i<3; ++i ){
        if( localWindow_.extent(i)>1 && bcInfo_.has_bc(i) ){
          assert( bcInfo_.num_extra(i) == Location::BCExtra::int_vec()[i] );
        }
      }
#     endif // NDEBUG
    }

    /**
     *  \brief SpatialField shallow copy constructor
     *
     * \param other SpatialField to copy
     *
     * This results in two fields that share the same underlying
     * memory and window information.
     */
    inline SpatialField( const SpatialField& other )
    : matchGlobalWindow_( other.matchGlobalWindow_ ),
      localWindow_( other.localWindow_ ),
      bcInfo_( other.bcInfo_ ),
      localValidGhosts_( other.localValidGhosts_ ),
      info_( other.info_ )
    {}

    /**
     *  \brief SpatialField shallow copy constructor with new window
     *
     * \param window memory window to use
     * \param other SpatialField to copy
     *
     * This results in two fields that share the same underlying
     * memory but have window information.
     *
     * If window maintains the underlying field's interior memory window
     * (window without ghost), and is within the field's global window
     * (window with all possible ghosts), then this copy can change the
     * number of global valid ghost cells. (These sort of changes would
     * change the number of valid ghost cells in other copies of this
     * same field that meet the same requirements.)
     */
    inline SpatialField( const MemoryWindow window,
                         const SpatialField& other )
    : matchGlobalWindow_( window.fits_between(other.info_->window_without_ghost(),
                                              other.info_->window_with_all_ghost()) ),
      localWindow_( window ),
      bcInfo_( other.info_->boundary_info() ),
      localValidGhosts_( other.localValidGhosts_.limit_by_extent(window.extent()) ),
      info_( other.info_ )
    {
      SingleValueCheck<Location>::check(window, localValidGhosts_);
    }

    ~SpatialField() {};

    const MemoryWindow& window_with_ghost() const{ return localWindow_; }

    const MemoryWindow window_without_ghost() const {
      if( matchGlobalWindow_ ) return info_->window_without_ghost();
      else                     return localWindow_.remove_ghosts(localValidGhosts_);
    }

    /**
     * \brief return the boundary cell information
     */
    const BoundaryCellInfo& boundary_info() const { return bcInfo_; }

    /**
     * \brief return the number of total ghost cells
     */
    const GhostData& get_ghost_data() const { return info_->get_ghost_data(); }

    /**
     * \brief return the number of current valid ghost cells
     */
    const GhostData& get_valid_ghost_data() const {
      if( matchGlobalWindow_ )
        return info_->get_valid_ghost_data();
      else
        return localValidGhosts_;
    }

    /**
     * \brief set the number of valid ghost cells to given ghosts
     *
     * \param ghosts ghost cells to be made valid
     */
    inline void reset_valid_ghosts( const GhostData& input ){
      GhostData const & ghosts = input.limit_by_extent(localWindow_.extent());
      localWindow_ = localWindow_.reset_ghosts( localValidGhosts_, ghosts );
      localValidGhosts_ = ghosts;
      if( matchGlobalWindow_ ) info_->reset_valid_ghosts(ghosts);
    }

    /**
     * \brief returns the number of allocated bytes (for copying memory)
     */
    unsigned int allocated_bytes() const{ return info_->allocated_bytes(); }

#   ifdef ENABLE_THREADS
    /**
     * \brief set the number of partitions Nebo uses for thread-parallel execution
     *
     * \param count the number of partitions to set
     */
    void set_partition_count( const int count) { info_->set_partition_count(count); }

    /**
     * \brief return the number of partitions Nebo uses for thread-parallel execution
     */
    int get_partition_count() const { return info_->get_partition_count(); }
#   endif

#   ifdef ENABLE_CUDA
    /**
     * \brief set the CUDA stream to for Nebo assignment to this field and for async data transfer
     *
     * \param stream the stream to use for this field
     */
    void set_stream( const cudaStream_t& stream ) { info_->set_stream(stream); }

    /**
     * \brief return the CUDA stream for this field
     */
    cudaStream_t const & get_stream() const { return info_->get_stream(); }
#   endif

    /**
     * \brief wait until the current stream is done with all work
     */
    inline void wait_for_synchronization() { info_->wait_for_synchronization(); }

    /**
     * \brief return the index of the current active device
     */
    short int active_device_index() const { return info_->active_device_index(); }

    /**
     * \brief add device memory to this field for given device
     *  and populate it with values from current active device *SYNCHRONOUS VERSION*
     *
     * \param deviceIndex the index of the device to add
     *
     * If device (deviceIndex) is already available for this field and
     * valid, this function becomes a no op.
     *
     * If device (deviceIndex) is already available for this field but
     * not valid, this function becomes identical to sync_device().
     *
     * Thus, regardless of the status of device (deviceIndex) for this
     * field, this function does the bare minimum to make device available
     * and valid for this field.
     *
     * Note: This operation is guaranteed to be synchronous: The host thread waits
     * until the task is completed (on the GPU).
     *
     * Note: This operation is thread safe.
     */
    inline void add_device( short int deviceIndex ) { info_->add_device( deviceIndex ); }

    /**
     * \brief add device memory to this field for given device
     *  and populate it with values from current active device *ASYNCHRONOUS VERSION*
     *
     * \param deviceIndex the index of the device to add
     *
     * If device (deviceIndex) is already available for this field and
     * valid, this function becomes a no op.
     *
     * If device (deviceIndex) is already available for this field but
     * not valid, this function becomes identical to sync_device().
     *
     * Thus, regardless of the status of device (deviceIndex) for this
     * field, this function does the bare minimum to make device available
     * and valid for this field.
     *
     * Note: This operation is asynchronous: The host thread returns immediately after
     * it launches on the GPU.
     *
     * Note: This operation is thread safe.
     */
    inline void add_device_async( short int deviceIndex ) { info_->add_device_async( deviceIndex ); }

   /**
     * \brief populate memory on the given device (deviceIndex) with values
     *  from the active device *SYNCHRONOUS VERSION*
     *
     * This function performs data-transfers when needed and only when needed.
     *
     * \param deviceIndex index for device to synchronize
     *
     * Note: This operation is guaranteed to be synchronous: The host thread waits
     * until the task is completed (on the GPU).
     */
    inline void validate_device( short int deviceIndex ){
      info_->validate_device( deviceIndex );
    }

   /**
     * \brief populate memory on the given device (deviceIndex) with values
     *  from the active device *ASYNCHRONOUS VERSION*
     *
     * This function performs data-transfers when needed and only when needed.
     *
     * \param deviceIndex index for device to synchronize
     *
     * Note: This operation is asynchronous: The host thread returns immediately after
     * it launches on the GPU.
     */
    inline void validate_device_async( short int deviceIndex ){
      info_->validate_device_async( deviceIndex );
    }

    /**
     * \brief set given device (deviceIndex) as active *SYNCHRONOUS VERSION*
     *
     * Given device must exist and be valid, otherwise an exception is thrown.
     *
     * \param deviceIndex index to device to be made active
     *
     * Note: This operation is guaranteed to be synchronous: The host thread waits
     * until the task is completed (on the GPU).
     */
    inline void set_device_as_active( const short int deviceIndex ){
      info_->set_device_as_active( deviceIndex );
    }

    /**
     * \brief set given device (deviceIndex) as active *ASYNCHRONOUS VERSION*
     *
     * Given device must exist and be valid, otherwise an exception is thrown.
     *
     * \param deviceIndex index to device to be made active
     *
     * Note: This operation is asynchronous: The host thread returns immediately
     * after it launches on the GPU.
     */
    inline void set_device_as_active_async( const short int deviceIndex ){
      info_->set_device_as_active_async( deviceIndex );
    }

    /**
     * \brief check if the device (deviceIndex) is available and valid
     *
     * \param deviceIndex index of device to check
     */
    bool is_valid( const short int deviceIndex ) const{
      return info_->is_valid( deviceIndex );
    }

    /**
     * \brief check if the device (deviceIndex) is available
     *
     * \param deviceIndex index of device to check
     */
    bool is_available( const short int deviceIndex ) const{
      return info_->is_available( deviceIndex );
    }

    /**
     * \brief return a non-constant pointer to memory on the given device
     *
     * Note: This method will invalidate all the other devices apart from the deviceIndex.
     *
     * \param deviceIndex index of device for device memory to return (defaults to CPU_INDEX)
     */
    inline T* field_values( const short int deviceIndex = CPU_INDEX ){
      return info_->field_values( deviceIndex );
    }

    /**
     * \brief return a constant pointer to memory on the given device
     *
     * \param deviceIndex device index for device memory to return (defaults to CPU_INDEX)
     */
    inline const T* field_values( const short int deviceIndex = CPU_INDEX ) const{
      return info_->const_field_values( deviceIndex );
    }

    /**
     * \brief return a constant iterator for CPU with valid ghost cells
     */
    inline const_iterator begin() const{
      return const_iterator( info_->const_field_values( CPU_INDEX ), window_with_ghost() );
    }

    /**
     * \brief return a non-constant iterator for CPU with valid ghost cells
     */
    inline iterator begin(){
      return iterator( info_->field_values( CPU_INDEX ), window_with_ghost() );
    }

    /**
     * \brief return a constant iterator to end for CPU with valid ghost cells
     */
    inline const_iterator end() const{
      const MemoryWindow & w = window_with_ghost();
      return begin() + w.extent(0) * w.extent(1) * w.extent(2);
    }

    /**
     * \brief return a non-constant iterator to end for CPU with valid ghost cells
     */
    inline iterator end(){
      const MemoryWindow & w = window_with_ghost();
      return begin() + w.extent(0) * w.extent(1) * w.extent(2);
    }

    /**
     * \brief return a constant iterator for CPU without ghost cells
     */
    inline const_iterator interior_begin() const{
      return const_iterator( info_->const_field_values( CPU_INDEX ), window_without_ghost() );
    }

    /**
     * \brief return a non-constant iterator for CPU without ghost cells
     */
    inline iterator interior_begin(){
      return iterator( info_->field_values( CPU_INDEX ), window_without_ghost() );
    }

    /**
     * \brief return a constant iterator to end for CPU without ghost cells
     */
    inline const_iterator interior_end() const{
      const MemoryWindow & w = window_without_ghost();
      return interior_begin() + w.extent(0) * w.extent(1) * w.extent(2);
    }

    /**
     * \brief return a non-constant iterator to end for CPU without ghost cells
     */
    inline iterator interior_end(){
      const MemoryWindow & w = window_without_ghost();
      return interior_begin() + w.extent(0) * w.extent(1) * w.extent(2);
    }

    /**
     * \brief return constant reference to cell at given flat index on CPU
     *
     * \param i flat index of cell to return a constant reference to
     *
     * Accessing cells with this function can be slow (depending on
     * access pattern), so this function should be used for testing,
     * not production code.
     *
     * Note that CPU_INDEX must be valid for this function to work.
     */
    const T& operator[](const size_t i) const {
      return info_->const_field_values(CPU_INDEX)[i];
    }

    /**
     * \brief return non-constant reference to cell at given flat index on CPU
     *
     * \param i flat index of cell to return a reference to
     *
     * Accessing cells with this function can be slow (depending on
     * access pattern), so this function should be used for testing,
     * not production code.
     *
     * Note that CPU_INDEX must be active for this function to work.
     */
    inline T& operator[](const size_t i) {
      return info_->field_values(CPU_INDEX)[i];
    }

    /**
     * \brief return non-constant reference to cell at given index (ijk) on CPU
     *
     * \param ijk IntVec coordinate (X,Y,Z) index
     *
     * Accessing cells with this function can be slow (depending on
     * access pattern), so this function should be used for testing,
     * not production code.
     *
     * Note that CPU_INDEX must be active for this function to work.
     */
    T& operator()(const IntVec& ijk) {
      return (*this)[window_with_ghost().flat_index(ijk)];
    }

    /**
     * \brief return constant reference to cell at given index (ijk) on CPU
     *
     * \param ijk IntVec coordinate (X,Y,Z) index
     *
     * Accessing cells with this function can be slow (depending on
     * access pattern), so this function should be used for testing,
     * not production code.
     *
     * Note that CPU_INDEX must be valid for this function to work.
     */
    const T& operator()(const IntVec& ijk) const {
      return (*this)[window_with_ghost().flat_index(ijk)];
    }

    /**
     * \brief return non-constant reference to cell at given index (i,j,k) on CPU
     *
     * \param i X-dimension index
     * \param j Y-dimension index
     * \param k Z-dimension index
     *
     * Accessing cells with this function can be slow (depending on
     * access pattern), so this function should be used for testing,
     * not production code.
     *
     * Note that CPU_INDEX must be active for this function to work.
     */
    T& operator()(const size_t i, const size_t j, const size_t k) {
      return (*this)(IntVec(i,j,k));
    }

    /**
     * \brief return constant reference to cell at given index (i,j,k) on CPU
     *
     * \param i X-dimension index
     * \param j Y-dimension index
     * \param k Z-dimension index
     *
     * Accessing cells with this function can be slow (depending on
     * access pattern), so this function should be used for testing,
     * not production code.
     *
     * Note that CPU_INDEX must be valid for this function to work.
     */
    const T& operator()(const size_t i, const size_t j, const size_t k) const {
      return (*this)(IntVec(i,j,k));
    }

    inline field_type& operator =(const field_type&);
  };  // SpatialField

//==================================================================
//
//                          Implementation
//
//==================================================================

  /**
   * \brief copies the contents of one field into another
   *
   * \param other SpatialField to copy data from
   *
   * The copy happens regardless of what active devices on each field are.
   *
   * This function WILL modify data outside of memory windows.  This limitation
   * is in place for two reasons:
   *  1. Currently, assignment only happens on windows that take up all memory
   *     (or nearly all).
   *  2. This limitation simplifies the implementation (which is easier to
   *     maintain) to use standard libraries (which should be more efficient).
   */
  template<typename Location, typename T>
  SpatialField<Location,T>&
  SpatialField<Location,T>::operator=(const SpatialField& other) {
    //check windows to be equal
    if( info_->window_with_ghost() != other.info_->window_with_ghost() ) {
      std::ostringstream msg;
      msg << "Error : Attempted assignment between fields of unequal size!\n"
          << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
      throw(std::runtime_error(msg.str()));
    }

    short int thisIdx = active_device_index();
    short int otherIdx = other.active_device_index();

#   ifdef ENABLE_CUDA
    ema::cuda::CUDADeviceInterface& CDI = ema::cuda::CUDADeviceInterface::self();
#   endif

    if( IS_CPU_INDEX(thisIdx) ) {
      if( IS_CPU_INDEX(otherIdx) ){
        // CPU -> CPU
        // check for self assignment
        if( field_values(CPU_INDEX) != other.field_values(CPU_INDEX) )
          std::copy( other.field_values(CPU_INDEX),
                     other.field_values(CPU_INDEX) + other.info_->window_with_ghost().glob_npts(),
                     field_values(CPU_INDEX) );
      }
#     ifdef ENABLE_CUDA
      else if( IS_GPU_INDEX(otherIdx) ) {
        // GPU -> CPU
        CDI.memcpy_from( field_values(CPU_INDEX),
                         other.field_values(otherIdx),
                         info_->allocated_bytes(),
                         otherIdx,
                         get_stream() );
      }
#     endif
      else {
        // ??? -> CPU
        std::ostringstream msg;
        msg << "Attempted unsupported copy operation, at n\t"
            << __FILE__ << " : " << __LINE__ << std::endl
            << "\t - "
            << DeviceTypeTools::get_memory_type_description(thisIdx) << " = "
            << DeviceTypeTools::get_memory_type_description(otherIdx) << std::endl;
        throw(std::runtime_error(msg.str()));
      }
    }
#   ifdef ENABLE_CUDA
    else if( IS_GPU_INDEX(thisIdx) ) {
      if( IS_CPU_INDEX(otherIdx) ) {
        // CPU -> GPU
        CDI.memcpy_to( field_values(thisIdx),
                       other.field_values(CPU_INDEX),
                       info_->allocated_bytes(),
                       thisIdx,
                       get_stream() );
      }
      else if( IS_GPU_INDEX( otherIdx ) ){
        // GPU -> GPU
        // Check for self assignment
        if( thisIdx != otherIdx ||
            field_values(otherIdx) != other.field_values(otherIdx) ) {
          CDI.memcpy_peer( field_values(thisIdx),
                           thisIdx,
                           other.field_values(otherIdx),
                           otherIdx,
                           info_->allocated_bytes() );
        }
      }
      else {
        // GPU -> ???
        std::ostringstream msg;
        msg << "Attempted unsupported copy operation, at " << std::endl
            << "\t" << __FILE__ << " : " << __LINE__ << std::endl
            << "\t - " << DeviceTypeTools::get_memory_type_description(thisIdx) << " = "
            << DeviceTypeTools::get_memory_type_description(otherIdx) << std::endl;
        throw( std::runtime_error ( msg.str() ));
      }
    }
#   endif // ENABLE_CUDA
    else{
      // ??? -> ___
      std::ostringstream msg;
      msg << "Attempted unsupported copy operation, at \n\t"
          << __FILE__ << " : " << __LINE__ << std::endl
          << "\t - " << DeviceTypeTools::get_memory_type_description(thisIdx)
          << " = "
          << DeviceTypeTools::get_memory_type_description(otherIdx) << std::endl;
      throw(std::runtime_error(msg.str()));
    }

    return *this;
  }

//------------------------------------------------------------------

  /**
   *  \fn BoundaryCellInfo create_new_boundary_cell_info<FieldType>( const PrototypeType )
   *
   *  \brief create a boundary cell info for a field of type FieldType from a field of type PrototypeType
   *
   *  \param prototype the prototype field
   *
   *  \return the new boundary cell info
   */
  template<typename FieldType, typename PrototypeType>
  inline BoundaryCellInfo create_new_boundary_cell_info( const PrototypeType& prototype ){
    return BoundaryCellInfo::build<FieldType>( prototype.boundary_info().has_bc() );
  }

//------------------------------------------------------------------

  /**
   *  \fn MemoryWindow create_new_memory_window<FieldType>( const PrototypeType )
   *
   *  \brief create a memory window for a field of type FieldType from a field of type PrototypeType
   *
   *  \param prototype the prototype field
   *
   *  \return the new memory window (with correct boundary conditions)
   */
  template<typename FieldType, typename PrototypeType>
  inline MemoryWindow create_new_memory_window( const PrototypeType& prototype )
  {
    const BoundaryCellInfo newBC = create_new_boundary_cell_info<FieldType,PrototypeType>(prototype);
    const MemoryWindow& prototypeWindow = prototype.window_with_ghost();
    const IntVec inc = newBC.has_bc() * Subtract< typename FieldType::Location::BCExtra, typename PrototypeType::Location::BCExtra >::result::int_vec();
    return MemoryWindow( prototypeWindow.glob_dim() + inc,
                         prototypeWindow.offset(),
                         prototypeWindow.extent() + inc );
  }

} // namespace SpatialOps

#endif // SpatialOps_SpatialField_h
