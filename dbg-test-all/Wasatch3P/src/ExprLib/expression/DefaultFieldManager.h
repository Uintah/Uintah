/**
 * \file DefaultFieldManager.h
 *
 * Copyright (c) 2011 The University of Utah
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
 */

/**
 * Available debugging flags
 *      DEBUG_FM_ALL - enable all debugging flags
 */
//#define DEBUG_FM_ALL

#ifdef DEBUG_FM_ALL
//Enable any/all other field manager debugging flags
#endif

#ifndef Expr_DefaultFieldManager_h
#define Expr_DefaultFieldManager_h

#include <string>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <sstream>
#include <stdio.h>

//-- boost includes --//
#include <boost/shared_ptr.hpp>
#include <boost/ref.hpp>
#include <boost/any.hpp>
#include <boost/foreach.hpp>

//-- expression library includes --//
#include <expression/Tag.h>
#include <expression/ManagerTypes.h>
#include <expression/FieldManagerBase.h>

//-- SpatialOps includes --//
#include <spatialops/structured/SpatialField.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/structured/MemoryTypes.h>
#include <spatialops/structured/ExternalAllocators.h>
#include <spatialops/particles/ParticleFieldTypes.h>
#include <spatialops/pointfield/PointFieldTypes.h>
#include <spatialops/Nebo.h>

namespace Expr {

  /**
   *  \struct FieldAllocInfo
   *  \brief used to get information into FieldAllocator
   */
  struct FieldAllocInfo {
    SpatialOps::IntVec dim;
    int nparticles, nrawpoints;
    bool bcplus[3];

    FieldAllocInfo( const SpatialOps::IntVec d,
                    const int nparticle,
                    const int nrawpoint,
                    const bool bcs[3] )
    {
      dim = d;
      nparticles = nparticle;
      nrawpoints = nrawpoint;
      bcplus[0] = bcs[0];
      bcplus[1] = bcs[1];
      bcplus[2] = bcs[2];
    }

    FieldAllocInfo( const std::vector<int> d,
                    const int nparticle,
                    const int nrawpoint,
                    const bool bcx,
                    const bool bcy,
                    const bool bcz )
    {
      dim = d;
      nparticles = nparticle;
      nrawpoints = nrawpoint;
      bcplus[0] = bcx;
      bcplus[1] = bcy;
      bcplus[2] = bcz;
    }

    FieldAllocInfo( const SpatialOps::IntVec d,
                    const int nparticle,
                    const int nrawpoint,
                    const bool bcx,
                    const bool bcy,
                    const bool bcz )
    {
      dim = d;
      nparticles = nparticle;
      nrawpoints = nrawpoint;
      bcplus[0] = bcx;
      bcplus[1] = bcy;
      bcplus[2] = bcz;
    }

    FieldAllocInfo() : dim(-1,-1,-1), nparticles(-1), nrawpoints(-1)
    {}

  };

  /** @class SpatialFieldAllocator
   *  @author Devin Robison
   *  @brief Provides allocate_field method which returns a spatial
   *	field given a FieldInfo object.  Note: This lets us avoid
   *	problems with partial class template specializations.
   */
  template<typename FieldT>
  struct SpatialFieldAllocator {
    static SpatialOps::SpatFldPtr<FieldT>
    allocate_field( const FieldAllocInfo& info,
                    short int deviceIndex = CPU_INDEX )
    {
      using namespace SpatialOps;
      const GhostData ghosts(1);  // for now, we hard-code one ghost cell for all field types.
      const BoundaryCellInfo bcInfo = BoundaryCellInfo::build<FieldT>( info.bcplus[0], info.bcplus[1], info.bcplus[2] );
      return SpatialFieldStore::get_from_window<FieldT>(
          get_window_with_ghost( info.dim, ghosts, bcInfo ),
          bcInfo, ghosts, deviceIndex );
    }
  };

  //TODO: This is allocated to local ram regardless of which manager is specified
  //      Need to figure out a better way to handle it.
  template<>
  struct SpatialFieldAllocator<SpatialOps::SingleValueField> {
    static SpatialOps::SpatFldPtr<SpatialOps::SingleValueField>
    allocate_field( const FieldAllocInfo& info,
                    short int deviceIndex = CPU_INDEX )
    {
      using namespace SpatialOps;
      int onept[3] = {1,1,1};
      return SpatialFieldStore::get_from_window<SingleValueField>(
          MemoryWindow( onept ),
          BoundaryCellInfo::build<SingleValueField>(false,false,false),
          GhostData(0), deviceIndex );
    }
  };

  template<>
  struct SpatialFieldAllocator<SpatialOps::Particle::ParticleField> {
    static SpatialOps::SpatFldPtr<SpatialOps::Particle::ParticleField>
    allocate_field( const FieldAllocInfo& info,
                    const short int deviceIndex = CPU_INDEX )
    {
      using namespace SpatialOps;
      return SpatialFieldStore::get_from_window<SpatialOps::Particle::ParticleField>(
          MemoryWindow( SpatialOps::IntVec(info.nparticles,1,1) ),
          BoundaryCellInfo::build<SpatialOps::Particle::ParticleField>(false,false,false),
          GhostData(0),
          deviceIndex );
    }
  };

  template<>
  struct SpatialFieldAllocator<SpatialOps::Point::PointField> {
    static SpatialOps::SpatFldPtr<SpatialOps::Point::PointField>
    allocate_field( const FieldAllocInfo& info,
                    short int deviceIndex = CPU_INDEX )
    {
      using namespace SpatialOps;
      return SpatialFieldStore::get_from_window<Point::PointField>(
          MemoryWindow( IntVec(info.nrawpoints,1,1) ),
          BoundaryCellInfo::build<Point::PointField>(false,false,false),
          GhostData(0),
          deviceIndex );
    }
  };


  /**
   *  @class  DefaultFieldManager
   *  @author James C. Sutherland
   */
  template<typename FieldT>
  class DefaultFieldManager: public FieldManagerBase {
  public:
    DefaultFieldManager();
    ~DefaultFieldManager();

    //** @brief FieldMangerBase interface wrapper **//
    FieldID register_field(const Tag&);

    /**
     * \brief Used to free resources associated with an individual field
     */
    inline bool release_field(const Tag&);

    /**
     * \brief Used to lock non-persistent fields, blocking memory from being freed automatically
     */
    inline bool lock_field(const Tag& tag);

    /**
     * \brief Used to unlock non-persistent fields, blocking memory from being freed automatically
     */
    inline bool unlock_field(const Tag& tag);

    /**
     * Allocates copy of the field, for consumption, on the target device.
     *
     * @param tag -- Field identifier
     * @param deviceIndex -- Index of the device type that will consume this field
     */
    inline void prep_field_for_consumption( const Tag& tag,
                                            const short int deviceIndex );

    /**
     * Allocates copy of the field, for consumption, on the target device.
     *
     * @param tag -- Field identifier
     * @param deviceIndex -- Index of the device type that will consume this field
     */
    inline void validate_field_location( const Tag& tag,
                                         const short int deviceIndex );
    /**
     * Allocates copy of the field, for consumption, on the target device.
     *
     * @param tag -- Field identifier
     * @param deviceIndex -- Index of the device type to set it as active
     */
    inline void set_active_field_location( const Tag& tag,
                                           const short int deviceIndex );

    /**
     *  \brief Update the resource manager for a specific field
     *
     *  @param tag -- Field identifier
     *  @param m -- the MemoryManager to set on this field
     *  @param deviceIndex -- Index of the device type that will consume this field
     *
     *  NOTE: Calling set_field_memory_manager while fields are in use
     *  will produce undefined behavior and you should not do it. After
     *  modifying the memory manager for a specific field type, any
     *  function depending on that field MUST make a subsequent call to
     *  field_ref.
     */
    void set_field_memory_manager( const Tag& tag,
                                   const MemoryManager m,
                                   const short int deviceIndex = CPU_INDEX );

    /**
     * @param tag - queried field
     * @return does queried field exist?
     */
    bool has_field( const Tag& tag ) const;

    /**
     *  @brief Retrieve the field with the specified tag.
     */
    const FieldT& field_ref( const Tag& tag ) const;
    FieldT& field_ref( const Tag& tag );

    /**
     *  \brief allocates all of the fields supported by this DefaultFieldManager
     *  \param info - the FieldInfo object.  A pointer to this will be held by the FieldManager, so ensure that it does not die.
     */
    virtual inline void allocate_fields( const boost::any& info );

    inline void deallocate_fields();

    void dump_fields(std::ostream& os) const;

    // this should only be called by the Expression base class for CARRY_FORWARD fields
    inline void copy_field_forward(const Tag& id, FieldT& f) const;

  private:

    struct FieldInfo{
      SpatialOps::SpatFldPtr<FieldT> fieldPtr;
      MemoryManager memoryMgr;
      int deviceIndex;
      bool isAllocated;
      bool isLocked;

      FieldInfo() : fieldPtr(), memoryMgr(MEM_EXTERNAL), deviceIndex(CPU_INDEX), isAllocated(false), isLocked(false)
      {}

      FieldInfo& operator=( const FieldInfo& a ){
        isAllocated = a.isAllocated;
        isLocked    = a.isLocked;
        memoryMgr   = a.memoryMgr;
        deviceIndex = a.deviceIndex;
        if( isAllocated ) fieldPtr = a.fieldPtr;
        else              fieldPtr.detach();
        return *this;
      }

      FieldInfo( const FieldInfo& a ){
        *this = a;
      }
    };

    typedef std::map<Tag, FieldInfo> IDFieldInfoMap; // Map field ID to its describing field structure.

    IDFieldInfoMap idFieldInfoMap_;

    FieldAllocInfo fieldAllocInfo_;
    bool hasAllocated_;

    DefaultFieldManager(const DefaultFieldManager&); // no copying
    DefaultFieldManager& operator=(const DefaultFieldManager&); // no assignment

  };

  //--------------------------------------------------------------------

  template<typename FieldT>
  DefaultFieldManager<FieldT>::DefaultFieldManager()
  {
    hasAllocated_ = false;
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  DefaultFieldManager<FieldT>::~DefaultFieldManager()
  {}

  //------------------------------------------------------------------

  template<typename FieldT>
  FieldID DefaultFieldManager<FieldT>::register_field( const Tag& tag )
  {
    const typename IDFieldInfoMap::iterator i = idFieldInfoMap_.find(tag);
    if( i == idFieldInfoMap_.end() ){
      FieldInfo finfo;
      finfo.isAllocated = false;
      finfo.isLocked    = false;
      finfo.memoryMgr   = MEM_EXTERNAL;
      idFieldInfoMap_[tag] = finfo;
    }
    return tag.id();
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  bool DefaultFieldManager<FieldT>::has_field(const Tag& tag) const {
    return ( idFieldInfoMap_.find(tag) != idFieldInfoMap_.end() );
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  inline void DefaultFieldManager<FieldT>::
  copy_field_forward( const Tag& tag, FieldT& f ) const
  {
    using namespace SpatialOps;
    const Tag tag1( tag.name(), STATE_NONE );
    const Tag tag2( tag.name(), STATE_N    );
    if( has_field(tag1) ){
      f <<= field_ref(tag1);
    }
    else if( has_field(tag2) ){
      f <<= field_ref(tag2);
    }
    else{
      std::ostringstream msg;
      msg << "copy_field_forward() called on \n\t" << tag
          << "\nwhich was not available as STATE_N or STATE_NONE\n\t"
          << __FILE__ << " : " << __LINE__ << std::endl << std::endl;
      throw std::runtime_error( msg.str() );
    }
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  void DefaultFieldManager<FieldT>::
  prep_field_for_consumption( const Tag& tag,
                              const short int deviceIndex )
  {
#   ifdef DEBUG_FM_ALL
    std::cout << "FieldStruct::prep_field_for_consumption on " << tag << std::endl;
#   endif

#   ifdef ENABLE_CUDA
    try{
      typename IDFieldInfoMap::iterator ifldx = idFieldInfoMap_.find(tag);

      FieldInfo& finfo = ifldx->second;

      if( finfo.fieldPtr.isnull() ){
        std::ostringstream msg;
        msg << "Prep_field_for_consumption called on an unallocated field, "
            << tag << std::endl
            << __FILE__ << " : " << __LINE__ << "\n";
        throw std::runtime_error( msg.str() );
      }

      //Attempts to add the requested field to the device
      finfo.fieldPtr->add_device_async( deviceIndex );
    }
    catch( std::runtime_error& e ) {
      std::ostringstream msg;
      msg << "Call to prep_field_for_consumption failed: \n" << e.what() << "\n\t-"
          << " call values => " << tag.name() << " , "
          << deviceIndex << "\n\t-"
          << " at " << __FILE__ << " : " << __LINE__ << std::endl;
      throw std::runtime_error( msg.str() );
    }
#   endif
  }

  //------------------------------------------------------------------


  template<typename FieldT>
  void DefaultFieldManager<FieldT>::
  validate_field_location( const Tag& tag,
                           const short int deviceIndex )
  {
#   ifdef DEBUG_FM_ALL
    std::cout << "DefaultFieldManager::validate_field_location() for " << tag
              << " & field Location : " << deviceIndex << std::endl;
#   endif

#   ifdef ENABLE_CUDA
    try{
      typename IDFieldInfoMap::iterator ifldx = idFieldInfoMap_.find(tag);

      FieldInfo& finfo = ifldx->second;

      if( finfo.fieldPtr.isnull() ){
        std::ostringstream msg;
        msg << "validate_field_location called on an unallocated field, "
            << tag << std::endl
            << __FILE__ << " : " << __LINE__ << "\n";
        throw std::runtime_error( msg.str() );
      }

      // Attempt to validate the requested field to the device
      finfo.fieldPtr->validate_device_async( deviceIndex );
    }
    catch( std::runtime_error& e ) {
      std::ostringstream msg;
      msg << "Call to validate_field_location failed: \n" << e.what() << "\n\t-"
          << " call values => " << tag.name() << " , "
          << deviceIndex << "\n\t-"
          << " at " << __FILE__ << " : " << __LINE__ << std::endl;
      throw std::runtime_error( msg.str() );
    }
#   endif
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  void DefaultFieldManager<FieldT>::
  set_active_field_location( const Tag& tag,
                              const short int deviceIndex )
  {
#   ifdef DEBUG_FM_ALL
    std::cout << "DefaultFieldManager::set_active_field_location() for " << tag
              << " & field Location : " << deviceIndex << std::endl;
#   endif

#   ifdef ENABLE_CUDA
    try{
      typename IDFieldInfoMap::iterator ifldx = idFieldInfoMap_.find(tag);

      FieldInfo& finfo = ifldx->second;

      if( finfo.fieldPtr.isnull() ){
        std::ostringstream msg;
        msg << "set_active_field_location called on an unallocated field, "
            << tag << std::endl
            << __FILE__ << " : " << __LINE__ << "\n";
        throw std::runtime_error( msg.str() );
      }

      //sets given device as active
      finfo.fieldPtr->set_device_as_active( deviceIndex );
    }
    catch( std::runtime_error& e ) {
      std::ostringstream msg;
      msg << "Call to set_active_field_location failed: \n" << e.what() << "\n\t-"
          << " call values => " << tag.name() << " , "
          << deviceIndex << "\n\t-"
          << " at " << __FILE__ << " : " << __LINE__ << std::endl;
      throw std::runtime_error( msg.str() );
    }
#   endif
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  void DefaultFieldManager<FieldT>::
  set_field_memory_manager( const Tag& tag,
                            const MemoryManager m,
                            const short int deviceIndex )
  {
    try{
      typename IDFieldInfoMap::iterator ifldx = idFieldInfoMap_.find(tag);
      if( ifldx == idFieldInfoMap_.end() ){
        std::ostringstream msg;
        msg << __FILE__ << " : " << __LINE__ << std::endl
            << "\tCould not find field information for " << tag << std::endl;
        throw std::invalid_argument( msg.str() );
      }

      FieldInfo& finfo = ifldx->second;

      const MemoryManager to = m;
      const MemoryManager from = finfo.memoryMgr;

#     ifdef DEBUG_FM_ALL
      std::cout << " DefaultFieldManager::set_field_memory_manager called on a field : " << tag
                << " , from (MM) : " << from << " , to (MM) : " << to << "," << deviceIndex << std::endl;
#     endif

      if( to == from ){ return; }

      // reset the field's memory manager and device index
      finfo.memoryMgr = m;
      finfo.deviceIndex = deviceIndex;

      // deal with potential field migration
      switch( from ){

        case MEM_EXTERNAL: {
          // This entire case block poses some interesting questions about
          // whether or not we want to copy existing values into a dynamic
          // field ( which would imply that we create the field... )
          switch (to) {
            case MEM_DYNAMIC: {
              // The default field manager allocates 'externally' from the Spatial field store, safe to just change mm and device index
              // if the field has already been allocated we keep it, but now it is eligible to be released.
#             ifdef DEBUG_FM_ALL
              std::cout << "MemoryManager swap: MEM_EXTERNAL -> MEM_DYNAMIC\n";
#             endif
              break;
            }

#           ifdef ENABLE_CUDA
            case MEM_DYNAMIC_GPU: {
              // Swapping to a dynamically allocated memory location from an external
#             ifdef DEBUG_FM_ALL
              std::cout << "MemoryManager swap: MEM_EXTERNAL -> MEM_DYNAMIC_GPU\n";
#             endif
              // A bit of a paradox here, an external field is guaranteed to be allocated.. so, do we want to copy it to the gpu?
              finfo.fieldPtr.detach();
              break;
            }

            case MEM_STATIC_GPU: {
              // The default field manager allocates 'externally' from the Spatial field store, safe to just change mm and device index
              // if the field has already been allocated we keep it, but now it is eligible to be released.
#             ifdef DEBUG_FM_ALL
              std::cout << "MemoryManager swap: MEM_EXTERNAL -> MEM_STATIC_GPU\n";
#             endif
              // A bit of a paradox here, an external field is guaranteed to be allocated.. so, do we want to copy it to the gpu?
              //finfo.fieldPtr.detach();
              break;
            }
#           endif // ENABLE_CUDA
            default: { // switch(to)
              std::ostringstream msg;
              msg << "Unsupported memory manager conversion requested at, "
                  << __FILE__ << " : " << __LINE__
                  << "\n\t - (" << from << ")" << " -> " << to << std::endl;
              throw(std::runtime_error(msg.str()));
            }
          }
          break;
        } // case MEM_EXTERNAL

        case MEM_DYNAMIC:{ // switch(from)
          switch( to ){
            case MEM_EXTERNAL: {
              // If the field is NOT null, then we just make it 'external', in which case its no longer
              // eligible for release.
              // If it is null, then it hasn't yet been allocated, so we need to grab memory for it
#             ifdef DEBUG_FM_ALL
              std::cout << "MemoryManager swap: MEM_DYNAMIC -> MEM_EXTERNAL\n";
#             endif
              break;
            } // case MEM_EXTERNAL
#           ifdef ENABLE_CUDA
            case MEM_DYNAMIC_GPU: {
#             ifdef DEBUG_FM_ALL
              std::cout << "MemoryManager swap: MEM_DYNAMIC -> MEM_DYNAMIC_GPU\n";
#             endif
              // If the field exists, then the dynamic field exists and we need to migrate it to GPU
              if( finfo.isAllocated ){ // !finfo.fieldPtr.isnull()
#               ifdef DEBUG_FM_ALL
                std::cout << "Field was already allocated, moving to GPU\n";
#               endif
                SpatialOps::SpatFldPtr<FieldT> fptrTmp = finfo.fieldPtr;
                finfo.fieldPtr = SpatialFieldAllocator<FieldT>::allocate_field(fieldAllocInfo_, GPU_INDEX);
                ema::cuda::CUDADeviceInterface& CDI = ema::cuda::CUDADeviceInterface::self();
                CDI.memcpy_to( finfo.fieldPtr->field_values(GPU_INDEX),
                               fptrTmp->field_values(CPU_INDEX),
                               fptrTmp->allocated_bytes(),
                               fptrTmp->active_device_index(),
                               fptrTmp->get_stream() );
              }
              break;
            } // case MEM_DYNAMIC_GPU - switch(from)

            case MEM_STATIC_GPU: {
#             ifdef DEBUG_FM_ALL
              std::cout << "MemoryManager swap: MEM_DYNAMIC -> MEM_STATIC_GPU\n";
#             endif
              // If the field exists, then the dynamic field exists and we need to migrate it to GPU
              if( !finfo.fieldPtr.isnull() ){
#               ifdef DEBUG_FM_ALL
                std::cout << "Field was already allocated, moving to GPU\n";
#               endif
                SpatialOps::SpatFldPtr<FieldT> fptrTmp = finfo.fieldPtr;
                finfo.fieldPtr = SpatialFieldAllocator<FieldT>::allocate_field(fieldAllocInfo_, GPU_INDEX);
                ema::cuda::CUDADeviceInterface& CDI = ema::cuda::CUDADeviceInterface::self();
                CDI.memcpy_to( finfo.fieldPtr->field_values(GPU_INDEX),
                               fptrTmp->field_values(CPU_INDEX),
                               fptrTmp->allocated_bytes(),
                               fptrTmp->active_device_index(),
                               fptrTmp->get_stream() );
              } else {
                finfo.fieldPtr = SpatialFieldAllocator<FieldT>::allocate_field(fieldAllocInfo_, GPU_INDEX);
              }
              break;
            } // case MEM_STATIC_GPU - switch(from)
#           endif // ENABLE_CUDA

            default: { // switch(to)
              std::ostringstream msg;
              msg << "Unsupported memory manager conversion requested at, "
                  << __FILE__ << " : " << __LINE__
                  << "\n\t - (" << from << ")" << " -> " << to << std::endl;
              throw std::runtime_error(msg.str());
            }
          } // switch(to)
          break;
        } // case MEM_DYNAMIC switch(from)

#       ifdef ENABLE_CUDA
        case MEM_DYNAMIC_GPU: {
          switch (to) {
            case MEM_EXTERNAL: {
              //Allocate our new external field
              //If the current field is not null, then we copy from the GPU
#             ifdef DEBUG_FM_ALL
              std::cout << "MemoryManager swap: MEM_DYNAMIC_GPU -> MEM_EXTERNAL\n";
#             endif
              if( !finfo.fieldPtr.isnull() ){
                SpatialOps::SpatFldPtr<FieldT> fptrTmp = finfo.fieldPtr;
                finfo.fieldPtr = SpatialFieldAllocator<FieldT>::allocate_field(fieldAllocInfo_, CPU_INDEX);
                ema::cuda::CUDADeviceInterface& CDI = ema::cuda::CUDADeviceInterface::self();
                CDI.memcpy_from( finfo.fieldPtr->field_values(CPU_INDEX),
                                 fptrTmp->field_values(GPU_INDEX),
                                 fptrTmp->allocated_bytes(),
                                 fptrTmp->active_device_index(),
                                 fptrTmp->get_stream() );
              } else {
                finfo.fieldPtr = SpatialFieldAllocator<FieldT>::allocate_field(fieldAllocInfo_, CPU_INDEX);
              }
              break;
            }

            case MEM_DYNAMIC: {
#             ifdef DEBUG_FM_ALL
              std::cout << "MemoryManager swap: MEM_DYNAMIC_GPU -> MEM_DYNAMIC\n";
#             endif
              if( !finfo.fieldPtr.isnull() ){
                SpatialOps::SpatFldPtr<FieldT> fptrTmp = finfo.fieldPtr;
                finfo.fieldPtr = SpatialFieldAllocator<FieldT>::allocate_field(fieldAllocInfo_, CPU_INDEX);
                ema::cuda::CUDADeviceInterface& CDI = ema::cuda::CUDADeviceInterface::self();
                CDI.memcpy_from( finfo.fieldPtr->field_values(CPU_INDEX),
                                 fptrTmp->field_values(GPU_INDEX),
                                 fptrTmp->allocated_bytes(),
                                 fptrTmp->active_device_index(),
                                 fptrTmp->get_stream() );
              }
              break;
            }

            case MEM_STATIC_GPU: {
#             ifdef DEBUG_FM_ALL
              std::cout << "MemoryManager swap: MEM_DYNAMIC_GPU -> MEM_STATIC_GPU\n";
#             endif
              // If the field exists, then the dynamic field exists and we need to migrate it to GPU
              if( !finfo.fieldPtr.isnull() ){
#               ifdef DEBUG_FM_ALL
                std::cout << "Field was already allocated, moving to GPU\n";
#               endif
                SpatialOps::SpatFldPtr<FieldT> fptrTmp = finfo.fieldPtr;
                finfo.fieldPtr = SpatialFieldAllocator<FieldT>::allocate_field(fieldAllocInfo_, GPU_INDEX);
                ema::cuda::CUDADeviceInterface& CDI = ema::cuda::CUDADeviceInterface::self();
                CDI.memcpy_peer( finfo.fieldPtr->field_values(),
                                 finfo.fieldPtr->active_device_index(),
                                 fptrTmp->field_values(),
                                 fptrTmp->active_device_index(),
                                 fptrTmp->allocated_bytes() );
              } else {
                finfo.fieldPtr = SpatialFieldAllocator<FieldT>::allocate_field(fieldAllocInfo_, GPU_INDEX);
              }
              break;
            }  // case MEM_STATIC_GPU - switch(from)

            default: {
              std::ostringstream msg;
              msg << "Unsupported memory manager conversion requested at, "
                  << __FILE__ << " : " << __LINE__
                  << "\n\t - (" << from << ")" << " -> " << to << std::endl;
              throw std::runtime_error(msg.str());
            }
          }
        break;
        }

        case MEM_STATIC_GPU: {
          switch (to) {
            case MEM_EXTERNAL: {
              //Allocate our new external field
              //If the current field is not null, then we copy from the GPU
#             ifdef DEBUG_FM_ALL
              std::cout << "MemoryManager swap: MEM_STATIC_GPU -> MEM_EXTERNAL\n";
#             endif
              if( !finfo.fieldPtr.isnull() ){
                SpatialOps::SpatFldPtr<FieldT> fptrTmp = finfo.fieldPtr;
                finfo.fieldPtr = SpatialFieldAllocator<FieldT>::allocate_field(fieldAllocInfo_, CPU_INDEX);
                ema::cuda::CUDADeviceInterface& CDI = ema::cuda::CUDADeviceInterface::self();
                CDI.memcpy_from( finfo.fieldPtr->field_values(CPU_INDEX),
                                 fptrTmp->field_values(GPU_INDEX),
                                 fptrTmp->allocated_bytes(),
                                 fptrTmp->active_device_index(),
                                 fptrTmp->get_stream() );
              } else {
                finfo.fieldPtr = SpatialFieldAllocator<FieldT>::allocate_field(fieldAllocInfo_, CPU_INDEX);
              }
              break;
            }

            case MEM_DYNAMIC: {
#             ifdef DEBUG_FM_ALL
              std::cout << "MemoryManager swap: MEM_STATIC_GPU -> MEM_DYNAMIC\n";
#             endif
              if( !finfo.fieldPtr.isnull() ){
                SpatialOps::SpatFldPtr<FieldT> fptrTmp = finfo.fieldPtr;
                finfo.fieldPtr = SpatialFieldAllocator<FieldT>::allocate_field(fieldAllocInfo_, CPU_INDEX);
                ema::cuda::CUDADeviceInterface& CDI = ema::cuda::CUDADeviceInterface::self();
                CDI.memcpy_from( finfo.fieldPtr->field_values(CPU_INDEX),
                                 fptrTmp->field_values(GPU_INDEX),
                                 fptrTmp->allocated_bytes(),
                                 fptrTmp->active_device_index(),
                                 fptrTmp->get_stream() );
              }
              break;
            }

            case MEM_DYNAMIC_GPU: {
#             ifdef DEBUG_FM_ALL
              std::cout << "MemoryManager swap: MEM_STATIC_GPU -> MEM_DYNAMIC_GPU\n";
#             endif
              // If the field exists, then the dynamic field exists and we need to migrate it to GPU
              if( !finfo.fieldPtr.isnull() ){
#               ifdef DEBUG_FM_ALL
                std::cout << "Field was already allocated, moving to GPU\n";
#               endif
                SpatialOps::SpatFldPtr<FieldT> fptrTmp = finfo.fieldPtr;
                finfo.fieldPtr = SpatialFieldAllocator<FieldT>::allocate_field(fieldAllocInfo_, GPU_INDEX);
                ema::cuda::CUDADeviceInterface& CDI = ema::cuda::CUDADeviceInterface::self();
                CDI.memcpy_peer( finfo.fieldPtr->field_values(),
                                 finfo.fieldPtr->active_device_index(),
                                 fptrTmp->field_values(),
                                 fptrTmp->active_device_index(),
                                 fptrTmp->allocated_bytes() );
              } else {
                finfo.fieldPtr = SpatialFieldAllocator<FieldT>::allocate_field(fieldAllocInfo_, GPU_INDEX);
              }
              break;
            }  // case MEM_STATIC_GPU - switch(from)

            default: {
              std::ostringstream msg;
              msg << "Unsupported memory manager conversion requested at, "
                  << __FILE__ << " : " << __LINE__
                  << "\n\t - (" << from << ")" << " -> " << to << std::endl;
              throw std::runtime_error(msg.str());
            }
          }
        break;
        }
#       endif // ENABLE_CUDA

        default: {
          std::ostringstream msg;
          msg
          << "Attempt to set manager type for an existing field with an invalid memory manager, at "
          << __FILE__ << " : " << __LINE__
          << "\n\t - We should never get here...";
          throw std::runtime_error(msg.str());
        }
      }
    }
    catch( std::exception& e ){
      std::ostringstream msg;
      msg << e.what() << std::endl << std::endl
          << "ERROR!  No field " << tag << " has been registered." << std::endl
          << "Registered fields follow. \n";;
      this->dump_fields(msg);
      msg << "\nEror from:\n\t" << __FILE__ << " : " << __LINE__ << std::endl;
      throw std::runtime_error(msg.str());
    }
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  FieldT&
  DefaultFieldManager<FieldT>::field_ref( const Tag& tag )
  {
    if( !hasAllocated_ ){
      std::ostringstream msg;
      msg << "Must call allocate_fields() prior to calling field_ref()\n\t"
          << __FILE__ << " : " << __LINE__ << std::endl;
      throw std::runtime_error(msg.str());
    }

    const typename IDFieldInfoMap::iterator ifinfo = idFieldInfoMap_.find(tag);

    if( ifinfo == idFieldInfoMap_.end() ){
      std::ostringstream msg;
      msg << "ERROR!  No field " << tag << " has been registered." << std::endl
          << "Registered fields follow: " << std::endl << " at " << __FILE__
          << " : " << __LINE__ << std::endl;
      this->dump_fields(msg);
      throw std::runtime_error(msg.str());
    }

    FieldInfo& finfo = ifinfo->second;

#   ifdef DEBUG_FM_ALL
    std::cout << "DefaultFieldManager::field_ref() for " << tag << std::endl;
#   endif
    if( !finfo.isAllocated ){
      switch( finfo.memoryMgr ){
        case MEM_DYNAMIC:
        case MEM_EXTERNAL: // For the default FM, we just allocate from the SFA.
#         ifdef DEBUG_FM_ALL
          std::cout << " -> allocating " << tag << " on " << finfo.memoryMgr << std::endl;
#         endif
          finfo.fieldPtr = SpatialFieldAllocator<FieldT>::allocate_field(fieldAllocInfo_, CPU_INDEX );
          finfo.isAllocated = true;
          break;

#       ifdef ENABLE_CUDA
        case MEM_DYNAMIC_GPU: {
#         ifdef DEBUG_FM_ALL
          std::cout << " -> allocating " << tag << " on " << finfo.memoryMgr << std::endl;
#         endif
          if( !IS_GPU_INDEX(finfo.deviceIndex) ){
            std::ostringstream msg;
            msg
            << "Error : device location : " << finfo.deviceIndex << " passed isn't compatible for memory type " <<  finfo.memoryMgr << ", at "
            << __FILE__ << " : " << __LINE__;
            throw std::runtime_error(msg.str());
          }
          finfo.fieldPtr = SpatialFieldAllocator<FieldT>::allocate_field(fieldAllocInfo_, finfo.deviceIndex);
          finfo.isAllocated = true;
          break;
        }

        case MEM_STATIC_GPU:  {
#         ifdef DEBUG_FM_ALL
          std::cout << " -> allocating " << tag << " on " << finfo.memoryMgr << std::endl;
#         endif
          if( !IS_GPU_INDEX(finfo.deviceIndex) ){
            std::ostringstream msg;
            msg
            << "Error : device location : " << finfo.deviceIndex << " passed isn't compatible for memory type " <<  finfo.memoryMgr << ", at "
            << __FILE__ << " : " << __LINE__;
            throw std::runtime_error(msg.str());
          }
          finfo.fieldPtr = SpatialFieldAllocator<FieldT>::allocate_field(fieldAllocInfo_, finfo.deviceIndex);
          finfo.isAllocated = true;
          break;
        }
#       endif // ENABLE_CUDA

        default:{
          std::ostringstream msg;
          msg << "Error: Request for unsupported memory manager " << finfo.memoryMgr
              << "\n\t" << __FILE__ << " : " << __LINE__ << std::endl;
          throw std::runtime_error(msg.str());
        }
      } // switch( finfo.memoryMgr )
    } // if

    return *finfo.fieldPtr;
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  const FieldT&
  DefaultFieldManager<FieldT>::field_ref( const Tag& tag ) const
  {
    if( !hasAllocated_ ){
      std::ostringstream msg;
      msg << "Must call allocate_fields() prior to calling field_ref()\n\t"
          << __FILE__ << " : " << __LINE__ << std::endl;
      throw std::runtime_error(msg.str());
    }

    const typename IDFieldInfoMap::const_iterator i = idFieldInfoMap_.find(tag);
    if (i == idFieldInfoMap_.end()) {
      std::ostringstream msg;
      msg << "ERROR!  No field " << tag << " has been registered!" << std::endl
          << " at " << __FILE__ << " : " << __LINE__ << std::endl;
      throw std::runtime_error(msg.str());
    }

#   ifdef DEBUG_FM_ALL
    std::cout << "UintahFieldManager::field_ref() const for " << tag << std::endl;
#   endif

    const FieldInfo& finfo = i->second;

    /* Check to see if the user is requesting a ( read only ) field reference to
     * a field that has been released. If so, then this is an implicit error.
     * Note: this is likely caused by a user failing to flag a specific field as
     * persistent and then referencing it later.
     */
    if( !finfo.isAllocated ){
      std::ostringstream msg;
      msg << "ERROR! Requesting a const reference to field \n\t" << tag
          << "\n\twhich is not persistent and has been released!\n"
          << "\tThe most likely cause for this is that the variable was not\n"
          << "\tinitialized prior to this function call.\n\n\t"
          << __FILE__ << " : " << __LINE__ << std::endl;
      throw std::runtime_error( msg.str() );
    }
    return *finfo.fieldPtr;
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  void DefaultFieldManager<FieldT>::allocate_fields( const boost::any& anyinfo )
  {
    try{
      fieldAllocInfo_ = boost::any_cast< boost::reference_wrapper<const FieldAllocInfo> > (anyinfo);
      hasAllocated_ = true;
      /* don't actually do the allocation here. This allows better error trapping
       * in the const field_ref() method.  The non-const field_ref() performs the
       * actual allocation, so a call to the const field_ref() method with an
       * unallocated field indicates a logic error and we can trap that if we
       * don't do the allocation here.
       */
    }
    catch( const boost::bad_any_cast & ){
      std::ostringstream msg;
      msg << __FILE__ << " : " << __LINE__ << std::endl
          << "improper information supplied to DefaultFieldManager::allocate_fields()"
          << std::endl
          << "Be sure to supply an Expr::FieldAllocInfo object to the allocate_fields() method."
          << std::endl;

      throw std::runtime_error( msg.str() );
    }
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  inline bool DefaultFieldManager<FieldT>::release_field( const Tag& tag )
  {
    const typename IDFieldInfoMap::iterator i = idFieldInfoMap_.find(tag);

    //std::cout << "Release field called for " << tag << std::endl;
    if( i == idFieldInfoMap_.end() ){
      std::ostringstream msg;
      msg << "ERROR!  No field " << tag << " has been registered." << std::endl
          << "Registered fields follow." << std::endl << std::endl << " at "
          << __FILE__ << " : " << __LINE__ << std::endl;
      this->dump_fields(msg);
      throw std::runtime_error(msg.str());
    }

    FieldInfo& finfo = i->second;

    if( finfo.isLocked || !finfo.isAllocated ) return false;

#   ifdef DEBUG_FM_ALL
    std::cout << "DefaultFieldManager::release_field() for " << tag << std::endl;
#   endif

    switch ( finfo.memoryMgr ) {
      case UNKNOWN:{
        std::ostringstream msg;
        msg << "ERROR!  " << get_memory_manager_description(finfo.memoryMgr) << " MemoryManager for " << tag
            << std::endl << __FILE__ << " : " << __LINE__ << std::endl;
        throw std::runtime_error( msg.str() );
      }
      case MEM_EXTERNAL   : // fall through
      case MEM_DYNAMIC    : // fall through
      case MEM_STATIC_GPU : // fall through
      case MEM_DYNAMIC_GPU: // fall through
        finfo.isAllocated = false;
        if( !finfo.fieldPtr.isnull() ) finfo.fieldPtr.detach();
    }
    return true;
  }

//------------------------------------------------------------------

  template<typename FieldT>
  inline bool DefaultFieldManager<FieldT>::lock_field(const Tag& tag) {
    const typename IDFieldInfoMap::iterator i = idFieldInfoMap_.find(tag);
    if( i == idFieldInfoMap_.end() )  return false;
    FieldInfo& finfo = i->second;
    finfo.isLocked = true;
    return true;
  }

  //--------------------------------------------------------------------

  template<typename FieldT>
  inline bool DefaultFieldManager<FieldT>::unlock_field(const Tag& tag) {
    const typename IDFieldInfoMap::iterator i = idFieldInfoMap_.find(tag);
    if( i == idFieldInfoMap_.end() ){
      std::ostringstream msg;
      msg << "ERROR!  No field " << tag << " has been registered." << std::endl
          << "Registered fields follow." << std::endl << " at " << __FILE__
          << " : " << __LINE__ << std::endl;
      this->dump_fields(msg);
      throw std::runtime_error(msg.str());
    }
    i->second.isLocked = false;
    return true;
  }

  //--------------------------------------------------------------------

  template<typename FieldT>
  inline void DefaultFieldManager<FieldT>::deallocate_fields()
  {
    BOOST_FOREACH( typename IDFieldInfoMap::value_type& myPair, idFieldInfoMap_ ){
      FieldInfo& finfo = myPair.second;
      if( !finfo.isAllocated ) continue; // go to next loop - skip deallocation.
      finfo.isAllocated = false;
      if( !finfo.fieldPtr.isnull() ) finfo.fieldPtr.detach();
#     ifdef DEBUG_FM_ALL
      std::cout << "DefaultFieldManager::deallocate_fields() for " << myPair.first << std::endl;
#     endif
    }
  }

  //--------------------------------------------------------------------

  template<typename FieldT>
  void DefaultFieldManager<FieldT>::dump_fields(std::ostream& os) const {
    using namespace std;
    os.setf(ios::left);
    os << setw(40) << "Field Name" << setw(12) << "Context" << setw(12) << "Mode"
        << setw(20) << "Field Manager Type" << endl
        << "--------------------------------------------------------------------------------------"
        << endl;
    BOOST_FOREACH( const typename IDFieldInfoMap::value_type& myPair, idFieldInfoMap_ ){
      const FieldInfo& finfo = myPair.second;
      os << setw(40) << myPair.first.name() << setw(12) << myPair.first.context()
         << setw(22) << finfo.memoryMgr << endl;
    }
    os
    << "--------------------------------------------------------------------------------------"
    << endl;
  }

  //--------------------------------------------------------------------

} // namespace Expr

#endif // Expr_DefaultDefaultFieldManager_h
