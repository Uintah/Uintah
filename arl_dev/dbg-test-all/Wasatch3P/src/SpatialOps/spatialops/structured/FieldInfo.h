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

#ifndef SPATIALOPS_FIELDINFO_H
#define SPATIALOPS_FIELDINFO_H

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <map>

#include <spatialops/SpatialOpsConfigure.h>

#include <spatialops/structured/ExternalAllocators.h>
#include <spatialops/structured/MemoryTypes.h>
#include <spatialops/structured/MemoryWindow.h>
#include <spatialops/structured/GhostData.h>
#include <spatialops/structured/BoundaryCellInfo.h>
#include <spatialops/structured/MemoryPool.h>

#ifdef ENABLE_THREADS
#include <boost/thread/mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#endif

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace SpatialOps{

  //Forward Declaration
  template <typename T> class Pool;

  /**
   * \enum StorageMode
   * \ingroup fields
   * \brief Specifies how memory should be treated in a SpatialField.
   */
  enum StorageMode
  {
    InternalStorage, ///< memory will be managed internally by SpatialOps
    ExternalStorage  ///< memory will be managed externally
  };

  /**
   * \class FieldInfo
   * \ingroup structured
   * \ingroup fields
   *
   * \brief Abstracts the internals of a field
   *
   * FieldInfo abstracts the low-level details of a field, mainly memory management
   * and synchronization of memory between devices.
   *
   * \tparam T - the underlying data type (defaults to \c double)
   *
   * \par Related classes:
   *  - \ref MemoryWindow
   *  - \ref SpatialFieldStore
   *  - \ref SpatialField
   *
   * \par Public Typedefs
   *  - \c value_type  - the type of underlying data being stored in this SpatialField
   */
  template< typename T=double >
  class FieldInfo {
    /**
     * \class DeviceMemory
     * \ingroup structured
     * \ingroup fields
     *
     * \brief Holds memory on a device along with some meta-data
     *
     * DeviceMemory holds pointer to memory for a single device, whether or not
     * that memory is valid, and whether or not that memory was internally allocated.
     *
     */
    struct DeviceMemory{
      T* field_;        ///< Pointer to memory
      bool isValid_;    ///< Flag true if memory holds valid data
      bool builtField_; ///< Flag true if memory was internally allocated

      /**
       * \brief DeviceMemory constructor
       *
       * \param field pointer to memory
       * \param isValid boolean flag, true if memory contains valid data
       * \param builtField boolean flag, true if memory allocated by SpatialField
       */
      DeviceMemory(T* field, bool isValid, bool builtField)
      : field_(field), isValid_(isValid), builtField_(builtField)
      {}

      /**
       * \brief DeviceMemory default constructor
       *
       * This constructor should only be used to build placeholder DeviceMemory
       * objects. That is, DeviceMemory objects that will be overwritten with actual
       * data soon.
       */
      DeviceMemory()
      : field_(NULL), isValid_(false), builtField_(false)
      {}
    };

    typedef std::map<short int, DeviceMemory>        DeviceMemoryMap;
    typedef typename DeviceMemoryMap::const_iterator ConstMapIter;
    typedef typename DeviceMemoryMap::iterator       MapIter;

    DeviceMemoryMap deviceMap_;     ///< Map from device indices to DeviceMemorys
    MemoryWindow wholeWindow_;      ///< Representation of the largest valid window for this field (includes ghost cells)
    const BoundaryCellInfo bcInfo_; ///< Information about this field's behavior on a boundary
    const GhostData totalGhosts_;   ///< The total number of ghost cells on each face of this field
    GhostData totalValidGhosts_;    ///< The number of valid ghost cells on each face of this field
    short int activeDeviceIndex_;   ///< The device index of the active device

#   ifdef ENABLE_THREADS
    int partitionCount_; // Number of partitions Nebo uses in its thread-parallel backend when assigning to this field
#   endif

#   ifdef ENABLE_CUDA
    cudaStream_t cudaStream_;
#   endif

#   ifdef ENABLE_THREADS
    /**
     *  \class ExecMutex
     *  \brief Scoped lock.
     */
    class ExecMutex {
      const boost::mutex::scoped_lock lock;
      inline boost::mutex& get_mutex() const {static boost::mutex m; return m;}
    public:
      ExecMutex() : lock( get_mutex() ) {}
      ~ExecMutex() {}
    };
#   endif

  public:
    typedef T value_type;

    /**
     * \brief FieldInfo constructor
     *
     * \param window      MemoryWindow for the entire field (including ghost cells)
     * \param bc          BoundaryConditionInfo for field
     * \param ghosts      GhostData for entire field (all possible ghosts)
     * \param fieldValues pointer to memory for ExternalStorage mode (default: NULL)
     * \param mode        either InternalStorage or ExternalStorage (default: InternalStorage)
     * \param devIdx      device index of originally active device (default: CPU_INDEX)
     */
    FieldInfo( const MemoryWindow& window,
               const BoundaryCellInfo& bc,
               const GhostData& ghosts,
               T* const fieldValues = NULL,
               const StorageMode mode = InternalStorage,
               const short int devIdx = CPU_INDEX );

    /**
     * \brief FieldInfo destructor
     *
     * This functions free any memory allocated by the FieldInfo.
     * Memory is allocated either in the constructor or when adding new devices.
     */
    ~FieldInfo();

    /**
     * \brief return the global memory window with valid ghost cells
     */
    const MemoryWindow& window_with_ghost() const { return wholeWindow_; }

    /**
     * \brief return the global memory window with no ghost cells
     */
    const MemoryWindow window_without_ghost() const {
      return window_with_ghost().remove_ghosts( get_valid_ghost_data() );
    }

    /**
     * \brief return the global memory window with all possible ghost cells
     */
    const MemoryWindow window_with_all_ghost() const {
      return window_with_ghost().reset_ghosts( get_valid_ghost_data(),
                                               get_ghost_data() );
    }

    /**
     * \brief return the boundary cell information
     */
    const BoundaryCellInfo& boundary_info() const{ return bcInfo_; }

    /**
     * \brief return the number of total ghost cells
     */
    const GhostData& get_ghost_data() const{ return totalGhosts_; }

    /**
     * \brief return the number of current valid ghost cells
     */
    const GhostData& get_valid_ghost_data() const{ return totalValidGhosts_; }

    /**
     * \brief set the number of valid ghost cells to given ghosts
     *
     * \param ghosts ghost cells to be made valid
     */
    inline void reset_valid_ghosts( const GhostData& ghosts ){
      wholeWindow_ = wholeWindow_.reset_ghosts( totalValidGhosts_,
                                                ghosts );
      totalValidGhosts_ = ghosts;
    }

    /**
     * \brief returns the number of allocated bytes (for copying memory)
     */
    unsigned int allocated_bytes() const { return sizeof(T) * (wholeWindow_.glob_npts()); }

#   ifdef ENABLE_THREADS
    /**
     * \brief set the number of partitions Nebo uses for thread-parallel execution
     *
     * \param count the number of partitions to set
     */
    void set_partition_count(const int count) { partitionCount_ = count; }

    /**
     * \brief return the number of partitions Nebo uses for thread-parallel execution
     */
    int get_partition_count() const { return partitionCount_; }
#   endif

#   ifdef ENABLE_CUDA
    /**
     * \brief set the CUDA stream to for Nebo assignment to this field and for async data transfer
     *
     * \param stream the stream to use for this field
     */
    void set_stream( const cudaStream_t& stream ) { cudaStream_ = stream; }

    /**
     * \brief return the CUDA stream for this field
     */
    cudaStream_t const & get_stream() const { return cudaStream_; }
#   endif

    /**
     * \brief wait until the current stream is done with all work
     */
    inline void wait_for_synchronization() {
#     ifdef ENABLE_CUDA
      ema::cuda::CUDADeviceInterface& CDI = ema::cuda::CUDADeviceInterface::self();
      if( get_stream() != NULL ) CDI.sync_stream( get_stream() );
#     endif
    }

    /**
     * \brief return the index of the current active device
     */
    short int active_device_index() const { return activeDeviceIndex_; }

    /**
     * \brief add device memory to this field for given device
     *  and populate it with values from current active device *SYNCHRONOUS VERSION*
     *
     * \param deviceIndex the index of the device to add
     *
     * If device (deviceIndex) is already available for this field and
     * valid, this fuction becomes a no op.
     *
     * If device (deviceIndex) is already available for this field but
     * not valid, this fuction becomes identical to validate_device().
     *
     * Thus, regardless of the status of device (deviceIndex) for this
     * field, this function does the bare minumum to make device available
     * and valid for this field.
     *
     * Note: This operation is guaranteed to be synchronous: The host thread waits
     * until the task is completed (on the GPU).
     *
     * Note: This operation is thread safe.
     */
    void add_device(short int deviceIndex);

    /**
     * \brief add device memory to this field for given device
     *  and populate it with values from current active device *ASYNCHRONOUS VERSION*
     *
     * \param deviceIndex the index of the device to add
     *
     * If device (deviceIndex) is already available for this field and
     * valid, this fuction becomes a no op.
     *
     * If device (deviceIndex) is already available for this field but
     * not valid, this fuction becomes identical to validate_device_async().
     *
     * Thus, regardless of the status of device (deviceIndex) for this
     * field, this function does the bare minumum to make device available
     * and valid for this field.
     *
     * Note: This operation is asynchronous: The host thread returns immediately after
     * it launches on the GPU.
     *
     * Note: This operation is thread safe.
     */
    void add_device_async( short int deviceIndex );

    /**
     * \brief popluate the memory of the given device (deviceIndex) with values
     *  from the active device *SYNCHRONOUS VERSION*
     *
     * This function performs data-transfers when needed and only when needed.
     *
     * \param deviceIndex index for device to synchronize
     *
     * Note: This operation is guaranteed to be synchronous: The host thread waits
     * until the task is completed (on the GPU).
     */
    void validate_device( short int deviceIndex );

    /**
     * \brief popluate the memory of the given device (deviceIndex) with values
     *  from the active device *ASYNCHRONOUS VERSION*
     *
     * This function performs data-transfers when needed and only when needed.
     *
     * \param deviceIndex index for device to synchronize
     *
     * Note: This operation is asynchronous: The host thread returns immediately after
     * it launches on the GPU.
     */
    void validate_device_async( short int deviceIndex );

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
    void set_device_as_active( short int deviceIndex );


    /**
     * \brief set given device (deviceIndex) as active *ASYNCHRONOUS VERSION*
     *
     * Given device must exist and be valid, otherwise an exception is thrown.
     *
     * \param deviceIndex index to device to be made active
     *
     * Note: This operation is asynchronous: The host thread returns immediately after
     * it launches on the GPU.
     */
    void set_device_as_active_async( short int deviceIndex );

    /**
     * \brief check if the device (deviceIndex) is available and valid
     *
     * \param deviceIndex index of device to check
     */
    bool is_valid( const short int deviceIndex ) const;

    /**
     * \brief check if the device (deviceIndex) is available
     *
     * \param deviceIndex index of device to check
     */
    bool is_available( const short int deviceIndex ) const;

    /**
     * \brief return a non-constant pointer to memory on the given device
     *
     * Note: This method will invalidate all the other devices apart from the deviceIndex.
     *
     * \param deviceIndex index of device for device memory to return (defaults to CPU_INDEX)
     */
    inline       T*       field_values(const short int deviceIndex = CPU_INDEX);

    /**
     * \brief return a constant pointer to memory on the given device
     *
     * NOTE: no check is performed to determine if the memory returned is valid.
     * See the is_valid() method to perform that check.
     *
     * \param deviceIndex device index for device memory to return (defaults to CPU_INDEX)
     */
    inline const T* const_field_values( const short int deviceIndex = CPU_INDEX ) const;
  }; // FieldInfo

//==================================================================
//
//                          Implementation
//
//==================================================================

  template<typename T>
  FieldInfo<T>::FieldInfo(const MemoryWindow& window,
                          const BoundaryCellInfo& bc,
                          const GhostData& ghost,
                          T* const fieldValues,
                          const StorageMode mode,
                          const short int devIdx)
    : wholeWindow_( window ),
      bcInfo_( bc.limit_by_extent(window.extent()) ),
      totalGhosts_( ghost.limit_by_extent(window.extent()) ),
      totalValidGhosts_( ghost.limit_by_extent(window.extent()) ),
      activeDeviceIndex_( devIdx )
      //Determine raw byte count -- this is sometimes required for external device allocation.
#     ifdef ENABLE_THREADS
      , partitionCount_( NTHREADS )
#     endif
#     ifdef ENABLE_CUDA
      , cudaStream_( 0 )
#     endif
    {
      //InternalStorage => we build a new field
      //ExternalStorage => we wrap T*

      //check if active device index is valid:
      if( IS_CPU_INDEX(activeDeviceIndex_)
#         ifdef ENABLE_CUDA
          || IS_GPU_INDEX(activeDeviceIndex_)
#         endif
      ){
        //set up active device in map:
        switch( mode ){
          case InternalStorage:
            deviceMap_[activeDeviceIndex_] = DeviceMemory( Pool<T>::get( activeDeviceIndex_, window.glob_npts() ), true, true );
            break;
          case ExternalStorage:
            if( fieldValues != NULL || window.local_npts() == 0 ){
              // allow NULL pointers so long as the window is
              // empty so that we never dereference the pointer.
              deviceMap_[activeDeviceIndex_] = DeviceMemory( fieldValues, true, false );
            }
            else if( window.local_npts() > 0 ){
              std::ostringstream msg;
              msg << "Attempting to use externally allocated memory in FieldInfo constructor, given NULL"
                  << " \n" << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
              throw(std::runtime_error(msg.str()));
            }
            break;
        } // switch(mode)
      }
      else {
        //not valid device index
        std::ostringstream msg;
        msg << "Attempt to create field on unsupported device ( "
            << DeviceTypeTools::get_memory_type_description(activeDeviceIndex_)
        << " )\n" << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
        throw(std::runtime_error(msg.str()));
      }
    }

//------------------------------------------------------------------

  template<typename T>
  FieldInfo<T>::~FieldInfo()
  {
    for( MapIter iter = deviceMap_.begin(); iter != deviceMap_.end(); ++iter )
      if( iter->second.builtField_ )
        Pool<T>::put(iter->first, iter->second.field_);
  }

//------------------------------------------------------------------

  template<typename T>
  void FieldInfo<T>::add_device( const short int deviceIndex )
  {
#   ifdef DEBUG_SF_ALL
    std::cout << "Call to SpatialField::add_device() for device : "
              << DeviceTypeTools::get_memory_type_description(deviceIndex) << std::endl;
#   endif

    add_device_async(deviceIndex);
    wait_for_synchronization();
  }

//------------------------------------------------------------------

  template<typename T>
  void FieldInfo<T>::add_device_async( const short int deviceIndex )
  {
#   ifdef DEBUG_SF_ALL
    std::cout << "Call to SpatialField::add_device_async() for device : "
              << DeviceTypeTools::get_memory_type_description(deviceIndex) << std::endl;
#   endif

#   ifdef ENABLE_THREADS
    ExecMutex lock;
#   endif

    DeviceTypeTools::check_valid_index(deviceIndex, __FILE__, __LINE__);

    MapIter iter = deviceMap_.find( deviceIndex );

    if(iter != deviceMap_.end()) {
      if(!iter->second.isValid_)
        validate_device_async( deviceIndex );
    }
    else {
      deviceMap_[deviceIndex] = DeviceMemory(Pool<T>::get( deviceIndex, (allocated_bytes()/sizeof(T)) ),
                                             false,
                                             true);
      validate_device_async( deviceIndex );
    }
  }

//------------------------------------------------------------------

  template<typename T>
  void FieldInfo<T>::validate_device( const short int deviceIndex )
  {
#   ifdef DEBUG_SF_ALL
    std::cout << "Call to SpatialField::validate_device() for device : "
              << DeviceTypeTools::get_memory_type_description(deviceIndex) << std::endl;
#   endif

    validate_device_async(deviceIndex);
    wait_for_synchronization();
  }

//------------------------------------------------------------------

  template<typename T>
  void FieldInfo<T>::validate_device_async( const short int deviceIndex )
  {
#   ifdef DEBUG_SF_ALL
    std::cout << "Call to SpatialField::validate_device_async() for device : "
              << DeviceTypeTools::get_memory_type_description(deviceIndex) << std::endl;
#   endif

    DeviceTypeTools::check_valid_index( deviceIndex, __FILE__, __LINE__ );

    //check if deviceIndex exists for field
    MapIter iter = deviceMap_.find( deviceIndex );
    if(iter == deviceMap_.end()) {
      std::ostringstream msg;
      msg << "Error : sync_device() did not find a valid field entry in the map. \n"
          << "Given: " << DeviceTypeTools::get_memory_type_description(deviceIndex) << std::endl
          << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
      throw(std::runtime_error(msg.str()));
    }

    // check that deviceIndex is not already valid for field
    if( deviceIndex != active_device_index() &&
        !deviceMap_[deviceIndex].isValid_ ){

#     ifdef ENABLE_CUDA
      ema::cuda::CUDADeviceInterface& CDI = ema::cuda::CUDADeviceInterface::self();
#     endif

      if( IS_CPU_INDEX(active_device_index()) &&
          IS_CPU_INDEX(deviceIndex) ) {
        // CPU->CPU
        // should not happen
        std::ostringstream msg;
        msg << "Error : sync_device() cannot copy from CPU to CPU did not find a valid field entry in the map. \n"
            << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
        throw( std::runtime_error(msg.str()) );
      }

#     ifdef ENABLE_CUDA

      else if( IS_CPU_INDEX(active_device_index()) &&
               IS_GPU_INDEX(deviceIndex) ){
        // CPU->GPU
#       ifdef DEBUG_SF_ALL
        std::cout << "data transfer from CPU to GPU" << std::endl;
#       endif
        CDI.memcpy_to((void*)deviceMap_[deviceIndex].field_,
                      deviceMap_[active_device_index()].field_,
                      allocated_bytes(),
                      deviceIndex,
                      get_stream());
        deviceMap_[deviceIndex].isValid_ = true;
      }

      else if( IS_GPU_INDEX(active_device_index()) &&
               IS_CPU_INDEX(deviceIndex) ){
        //GPU->CPU
#       ifdef DEBUG_SF_ALL
        std::cout << "data transfer from GPU to CPU" << std::endl;
#       endif
        CDI.memcpy_from((void*)deviceMap_[deviceIndex].field_,
                        deviceMap_[active_device_index()].field_,
                        allocated_bytes(),
                        active_device_index(),
                        get_stream());
        deviceMap_[deviceIndex].isValid_ = true;
      }

      else if( IS_GPU_INDEX(active_device_index()) &&
               IS_GPU_INDEX(deviceIndex) ){
        //GPU->GPU
#       ifdef DEBUG_SF_ALL
        std::cout << "data transfer from GPU to GPU" << std::endl;
#       endif
        CDI.memcpy_peer((void*)deviceMap_[deviceIndex].field_,
                        deviceIndex,
                        deviceMap_[active_device_index()].field_,
                        active_device_index(),
                        allocated_bytes());
        deviceMap_[deviceIndex].isValid_ = true;
      }
#     endif // ENABLE_CUDA

      else{
        std::ostringstream msg;
        msg << "Error : sync_device() called on the field with: "
            << DeviceTypeTools::get_memory_type_description(deviceIndex)
            << "\n\t - " << __FILE__ << " : " << __LINE__ << std::endl;
        throw(std::runtime_error(msg.str()));
      }
    }
  }

//------------------------------------------------------------------

  template<typename T>
  void FieldInfo<T>::set_device_as_active( const short int deviceIndex )
  {
#   ifdef DEBUG_SF_ALL
    std::cout << "Call to SpatialField::set_device_as_active() for device : "
              << DeviceTypeTools::get_memory_type_description(deviceIndex) << std::endl;
#   endif

    set_device_as_active_async( deviceIndex );
    wait_for_synchronization();
  }

//------------------------------------------------------------------

  template<typename T>
  void FieldInfo<T>::set_device_as_active_async( const short int deviceIndex )
  {
#   ifdef DEBUG_SF_ALL
    std::cout << "Call to SpatialField::set_device_as_active() for device : "
              << DeviceTypeTools::get_memory_type_description(deviceIndex) << std::endl;
#   endif

    DeviceTypeTools::check_valid_index( deviceIndex, __FILE__, __LINE__ );

    // make device available and valid, if not already
    if( deviceMap_.find(deviceIndex) == deviceMap_.end() ||
        !deviceMap_[deviceIndex].isValid_ ){
      add_device_async( deviceIndex );
    }
    activeDeviceIndex_ = deviceIndex;
  }

//------------------------------------------------------------------

  template<typename T>
  bool FieldInfo<T>::is_valid( const short int deviceIndex ) const
  {
#   ifdef DEBUG_SF_ALL
    std::cout << "Call to SpatialField::is_valid() for device : "
              << DeviceTypeTools::get_memory_type_description(deviceIndex) << std::endl;
#   endif

    DeviceTypeTools::check_valid_index( deviceIndex, __FILE__, __LINE__ );

    ConstMapIter iter = deviceMap_.find( deviceIndex );
#   ifndef DEBUG_SF_ALL
    if( iter == deviceMap_.end() )
      std::cout << "Field Location " << DeviceTypeTools::get_memory_type_description( deviceIndex )
                << " is not allocated. " << std::endl;
    else if( !iter->second.isValid_ )
      std::cout << "Field Location " << DeviceTypeTools::get_memory_type_description( deviceIndex )
                << " is not valid. " << std::endl;
#   endif
    return ( iter != deviceMap_.end() && iter->second.isValid_ );
  }

//------------------------------------------------------------------

  template<typename T>
  bool FieldInfo<T>::is_available( const short int deviceIndex ) const
  {
#   ifdef DEBUG_SF_ALL
    std::cout << "Call to SpatialField::is_available() for device : "
              << DeviceTypeTools::get_memory_type_description(deviceIndex) << std::endl;
#   endif

    DeviceTypeTools::check_valid_index( deviceIndex, __FILE__, __LINE__ );

    ConstMapIter iter = deviceMap_.find( deviceIndex );
#   ifndef DEBUG_SF_ALL
    if( iter == deviceMap_.end() )
      std::cout << "Field Location " << DeviceTypeTools::get_memory_type_description( deviceIndex )
                << " is not allocated. " << std::endl;
#   endif
    return ( iter != deviceMap_.end() );
  }

//------------------------------------------------------------------

  template<typename T>
  T* FieldInfo<T>::field_values( const short int deviceIndex )
  {
#   ifdef DEBUG_SF_ALL
    std::cout << "Call to non-const SpatialField::field_values() for device : "
              << DeviceTypeTools::get_memory_type_description(deviceIndex) << std::endl;
#   endif

    DeviceTypeTools::check_valid_index( deviceIndex, __FILE__, __LINE__ );

    // check active
    if( active_device_index() == deviceIndex ){
      MapIter iter = deviceMap_.find( deviceIndex );

      // mark all the other devices except active device as invalid
      for(MapIter iter2 = deviceMap_.begin(); iter2 != deviceMap_.end(); ++iter2 ){
        if( !(iter2->first == activeDeviceIndex_) ) iter2->second.isValid_ = false;
      }
      return iter->second.field_;
    }
    else{
      std::ostringstream msg;
      msg << "Request for nonconst field pointer on a nonactive device: "
          << DeviceTypeTools::get_memory_type_description(deviceIndex) << std::endl
          << "Active device is: "
          << DeviceTypeTools::get_memory_type_description(active_device_index()) << std::endl
          << "Please check the arguments passed into the function."
          << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
      throw( std::runtime_error(msg.str()) );
    }
  }

//------------------------------------------------------------------

  template<typename T>
  const T* FieldInfo<T>::const_field_values( const short int deviceIndex ) const
  {
#   ifdef DEBUG_SF_ALL
    std::cout << "Call to const SpatialField::field_values() for device : "
              << DeviceTypeTools::get_memory_type_description(deviceIndex) << std::endl;
#   endif

    DeviceTypeTools::check_valid_index( deviceIndex, __FILE__, __LINE__ );

    ConstMapIter iter = deviceMap_.find( deviceIndex );

    if( iter == deviceMap_.end() ) {
      //device is not available
      std::ostringstream msg;
      msg << "Request for const field pointer on a device for which it has not been allocated\n"
          << DeviceTypeTools::get_memory_type_description(deviceIndex)
          << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
      throw(std::runtime_error(msg.str()));
    }
//    else {
//      //device is available but not valid
//      std::ostringstream msg;
//      msg << "Requested const field pointer on a device is not valid! \n"
//          << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
//      throw(std::runtime_error(msg.str()));
//    }
    return iter->second.field_;
  }

//------------------------------------------------------------------

} // namespace SpatialOps

#endif // SPATIALOPS_FIELDINFO_H
