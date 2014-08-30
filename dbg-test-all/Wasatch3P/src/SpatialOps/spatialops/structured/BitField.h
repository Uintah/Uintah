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
 */

#ifndef SpatialOps_BitField_h
#define SpatialOps_BitField_h


#include <iostream>
#include <cassert>
#include <sstream>
#include <vector>

#include <spatialops/SpatialOpsConfigure.h>

#include <spatialops/structured/IntVec.h>
#include <spatialops/structured/MemoryWindow.h>
#include <spatialops/structured/GhostData.h>
#include <spatialops/structured/MemoryPool.h>

#include <climits>  // for CHAR_BIT below.

#define NEBO_INT_BIT ((signed int)(sizeof(unsigned int) * CHAR_BIT))
#define NEBO_INT_BYTE ((signed int)(sizeof(unsigned int)))
#define NEBO_ROUND_TO_INT(size) ((signed int)((size + NEBO_INT_BIT - 1) / NEBO_INT_BIT))

namespace SpatialOps{

  class ConstMaskIterator : public std::iterator<std::random_access_iterator_tag, bool> {
    typedef ConstMaskIterator MyType;

  public:
  ConstMaskIterator(unsigned int * field_values,
                    const MemoryWindow & w)
    : bitField_(field_values),
      bitPosition_((w.offset(0) +
                    w.glob_dim(0) * (w.offset(1) +
                                     w.glob_dim(1) * w.offset(2)))
                   % NEBO_INT_BIT),
      blockPosition_((w.offset(0) +
                      w.glob_dim(0) * (w.offset(1) +
                                       w.glob_dim(1) * w.offset(2)))
                     / NEBO_INT_BIT),
      count_(0),
      size_(NEBO_ROUND_TO_INT(w.glob_npts())),
      xIndex_(0),
      yIndex_(0),
      zIndex_(0),
      yStep_(w.glob_dim(0) - w.extent(0)),
      zStep_((w.glob_dim(1) - w.extent(1)) * w.glob_dim(0)),
      xExtent_(w.extent(0)),
      yExtent_(w.extent(1)),
      zExtent_(w.extent(2)),
      xyExtent_(w.extent(0) * w.extent(1))
  {}

    //immutable dereference
    inline bool operator*() const {
#     ifndef NDEBUG
      if(bitPosition_ < 0 ||
         bitPosition_ >= NEBO_INT_BIT) {
        std::ostringstream msg;
        msg << __FILE__ << " : " << __LINE__ << std::endl
            << "iterator's bitPosition_ is out of bounds";
        throw std::runtime_error(msg.str());
      }
      if(blockPosition_ < 0 ||
         blockPosition_ >= size_) {
        std::ostringstream msg;
        msg << __FILE__ << " : " << __LINE__ << std::endl
            << "iterator's blockPosition_ is out of bounds";
        throw std::runtime_error(msg.str());
      }
      if(count_ != (xIndex_ +
                    yIndex_ * xExtent_ +
                    zIndex_ * xyExtent_)) {
        std::ostringstream msg;
        msg << __FILE__ << " : " << __LINE__ << std::endl
            << "iterator's internal count is off";
        throw std::runtime_error(msg.str());
      }
      if(xIndex_ >= xExtent_ ||
         yIndex_ >= yExtent_ ||
         zIndex_ >= zExtent_ ||
         xIndex_ < 0 ||
         yIndex_ < 0 ||
         zIndex_ < 0) {
        std::ostringstream msg;
        msg << __FILE__ << " : " << __LINE__ << std::endl
            << "iterator is in an invalid state for dereference";
        throw std::runtime_error(msg.str());
      }
#     endif
      return !!(*(bitField_ + blockPosition_) & (1 << bitPosition_));
    }

    //increment
    inline MyType & operator++() {
      bitPosition_++; //xStep
      count_++;
      xIndex_++;
      if(xIndex_ == xExtent_){
        bitPosition_ += yStep_; //yStep
        xIndex_ = 0;
        yIndex_++;
        if(yIndex_ == yExtent_){
          bitPosition_ += zStep_; //zStep
          yIndex_ = 0;
          zIndex_++;
        }
      }

      //recalculate bitPosition_ and blockPosition_
      update_positions();

      return *this;
    };

    inline MyType operator++(int) { MyType result = *this; ++(*this); return result; };

    //decrement
    inline MyType & operator--() {
      --bitPosition_; //xStep
      --count_;
      --xIndex_;
      if( xIndex_ == -1 ){
        bitPosition_ -= yStep_; //yStep
        xIndex_ = xExtent_ - 1;
        --yIndex_;
        if( yIndex_ == -1 ){
          bitPosition_ -= zStep_; //zStep
          yIndex_ = yExtent_ - 1;
          --zIndex_;
        }
      }

      //recalculate bitPosition_ and blockPosition_
      update_positions();

      return *this;
    }

    inline MyType operator--(int) { MyType result = *this; --(*this); return result; };

    //compound assignment
    inline MyType & operator+=(int change) {
      //small change (only changes xIndex_)
      if( (change > 0 && change < xExtent_ - xIndex_) || //positive change
          (change < 0 && - change < xIndex_) ){ //negative change
        bitPosition_ += change;
        xIndex_ += change;
        count_ += change;
      }
      else { //bigger change (changes yIndex_ and/or zIndex_)
        int new_count = count_ + change;
        int old_count = count_;
        bitPosition_ += (change + //xStep
                         yStep_ * ((new_count / xExtent_) - (old_count / xExtent_)) +
                         zStep_ * ((new_count / xyExtent_) - (old_count /xyExtent_)));
        count_ += change;
        xIndex_ = count_ % xExtent_;
        yIndex_ = (count_ % xyExtent_) / xExtent_;
        zIndex_ = count_ / xyExtent_;
      }

      //recalculate bitPosition_ and blockPosition_
      update_positions();

      return *this;
    }

    inline MyType & operator-=(int change) { return *this += -change; }

    //addition/subtraction
    inline MyType operator+ (int change) const { MyType result = *this; result += change; return result; }
    inline MyType operator- (int change) const { return *this + (-change); };

    //iterator subtraction
    inline ptrdiff_t operator- (MyType const & other) const { return count_ - other.count_; }

    //offset dereference
    inline bool operator[](int change) { MyType result = *this; result += change; return *result; }

    //comparisons
    inline bool operator==(MyType const & other) const { return bitField_ == other.bitField_ && count_ == other.count_; }
    inline bool operator!=(MyType const & other) const { return bitField_ != other.bitField_ || count_ != other.count_; }
    inline bool operator< (MyType const & other) const { return bitField_ == other.bitField_ && count_ <  other.count_; }
    inline bool operator> (MyType const & other) const { return bitField_ == other.bitField_ && count_ >  other.count_; }
    inline bool operator<=(MyType const & other) const { return bitField_ == other.bitField_ && count_ <= other.count_; }
    inline bool operator>=(MyType const & other) const { return bitField_ == other.bitField_ && count_ >= other.count_; }

  private:

    inline void update_positions(void) {
      if(bitPosition_ < 0 ||
         bitPosition_ >= NEBO_INT_BIT) {
        const int flatPosition = blockPosition_ * NEBO_INT_BIT + bitPosition_;
        blockPosition_ = flatPosition / NEBO_INT_BIT;
        bitPosition_ = flatPosition % NEBO_INT_BIT;
      }
    }

    unsigned int * bitField_;
    int bitPosition_;
    int blockPosition_;
    int count_;
    int size_;
    int xIndex_, yIndex_, zIndex_;
    int yStep_;
    int zStep_;
    int xExtent_, yExtent_, zExtent_;
    int xyExtent_;
  };

  /**
   *  \class BitField
   *  \ingroup fields
   *
   *  \brief Implements a mask as a bitfield.
   *
   *  BitField is designed to be used within SpatialMask.
   *  In general, it is better to use SpatialMask than BitField directly.
   *
   *  \par Related classes:
   *   - \ref MemoryWindow
   *   - \ref SpatialMask
   */
  class BitField
  {
  public:

    typedef MemoryWindow memory_window;
    typedef ConstMaskIterator const_iterator;

  private:
    typedef std::map<unsigned short int, unsigned int*> ConsumerMap;

    MemoryWindow maskWindow_;	        ///< Representation of the window for this mask ( does NOT include ghost cells )
    const GhostData ghosts_;            ///< Total possible ghost cells for mask

    const unsigned long int size_;      ///< Stores entire mask size (in terms of unsigned int's)
    const unsigned long int bytes_;      ///< Stores entire mask size (in terms of bites)

    unsigned int * bitValues_;		///< Values associated with this mask in the context of LOCAL_RAM
    unsigned int * bitValuesExtDevice_; ///< External mask pointer ( This pointer will only be valid on the device it was created )
    const bool builtMask_;		///< Indicates whether or not we created this mask ( we could just be wrapping memory )
    bool hasgpuConsumer_;               ///< Indicates whether a GPU consumer exists or not.

    const short int deviceIndex_;       ///< Indicates which device is this mask stored on


    //Note: Presently this is assumed to be a collection of GPU based < index, pointer pairs >;
    //      which is not as general is it likely should be, but GPUs are currently the only external
    //      device we're interested in supporting.
    ConsumerMap consumerBitValues_;	///< Provides the ability to store and track copies of this mask consumed on other devices.
    ConsumerMap myConsumerBitValues_;	///< Provides the ability to correctly delete/release copies of this mask that this mask allocated

    inline void reset_values(void)
    {
      if( deviceIndex_ == CPU_INDEX ){
        for( unsigned int long i = 0; i < size_; ++i )
          bitValues_[i] = 0;
      }
      else{
        std::ostringstream msg;
        msg << "Reset values called for unsupported field type ( "
            << DeviceTypeTools::get_memory_type_description(deviceIndex_) << " )"
            << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
        throw(std::runtime_error(msg.str()));
      }
    }

    inline int find_position(const IntVec & point) const
    {
#     ifndef NDEBUG
      //check that point is in bounds
      assert(point[0] >= -ghosts_.get_minus(0));
      assert(point[0] < (signed int)(maskWindow_.extent(0)) + ghosts_.get_plus(0));
      assert(point[1] >= -ghosts_.get_minus(1));
      assert(point[1] < (signed int)(maskWindow_.extent(1)) + ghosts_.get_plus(1));
      assert(point[2] >= -ghosts_.get_minus(2));
      assert(point[2] < (signed int)(maskWindow_.extent(2)) + ghosts_.get_plus(2));
#     endif

      const int xIndex = maskWindow_.offset(0) + point[0];
      const int yIndex = maskWindow_.offset(1) + point[1];
      const int zIndex = maskWindow_.offset(2) + point[2];

      const int xTotal = maskWindow_.glob_dim(0);
      const int yTotal = maskWindow_.glob_dim(1);

      return xIndex + xTotal * (yIndex + yTotal * zIndex);
    };

    inline int find_block(const int flat) const { return flat / NEBO_INT_BIT; };

    inline int find_bit_position(const int flat) const { return flat % NEBO_INT_BIT; };

    inline void add_point(const IntVec & point)
    {
      if( bitValues_ != NULL ){
        const int position = find_position(point);
        unsigned int * const blockPosition = bitValues_ + find_block(position);
        const int bitPosition = find_bit_position(position);
        //update block with new point
        *blockPosition = (*blockPosition | (1 << bitPosition));
      }
      else{
        std::ostringstream msg;
        msg << "Unsupported attempt to add points to a mask of type ( "
            << DeviceTypeTools::get_memory_type_description(deviceIndex_)
            << " )\n" << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
        throw(std::runtime_error(msg.str()));
      }
    }

    inline void add_points( const std::vector<IntVec> & points )
    {
      for(std::vector<IntVec>::const_iterator i = points.begin(); i != points.end(); i++)
        add_point(*i);
    }

  public:

    /**
     *  \brief Construct a BitField
     *  \param points - the points in the mask
     *  \param window - the window to build
     *  \param ghosts - the ghost information
     */
    BitField(const std::vector<IntVec> & points,
             const MemoryWindow & window,
             const GhostData & ghosts)
      : maskWindow_(window),
        ghosts_(ghosts),
        size_(NEBO_ROUND_TO_INT(window.glob_npts())),
        bytes_(NEBO_ROUND_TO_INT(window.glob_npts()) * NEBO_INT_BYTE),
        bitValues_(Pool<unsigned int>::get(CPU_INDEX, size_)),
        builtMask_(true),
        hasgpuConsumer_(false),
        deviceIndex_(CPU_INDEX)
    {
      reset_values();
      add_points(points);
    }

    /**
     *  \brief Shallow copy constructor.  This results in two masks
     *  that share the same underlying memory.
     */
    BitField(const BitField& other)
      : maskWindow_(other.maskWindow_),
        ghosts_(other.ghosts_),
        size_(other.size_),
        bytes_(other.bytes_),
        bitValues_(other.bitValues_),
        builtMask_(false),
        hasgpuConsumer_(false),
        deviceIndex_(other.deviceIndex_),
        consumerBitValues_(other.consumerBitValues_)
    {}

    ~BitField()
    {
#     ifdef ENABLE_CUDA
      //Release any masks allocated for consumer use
      if( hasgpuConsumer_ ){
        for( ConsumerMap::iterator i = myConsumerBitValues_.begin(); i != myConsumerBitValues_.end(); ++i ){
          Pool<unsigned int>::put( i->first, i->second );
        }
      }
      consumerBitValues_.clear();
      myConsumerBitValues_.clear();
      hasgpuConsumer_ = false;
#     endif // ENABLE_CUDA

      if ( builtMask_ && !hasgpuConsumer_ ) {
        if( deviceIndex_ == CPU_INDEX ){
          Pool<unsigned int>::put( CPU_INDEX, bitValues_ );
          bitValues_ = NULL;
        }
        else{
          std::ostringstream msg;
          msg << "Attempt to release ( "
              << DeviceTypeTools::get_memory_type_description(deviceIndex_)
              << " ) mask type, without supporting libraries -- this likely indicates a serious problem in mask initialization\n"
              << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
          throw( std::runtime_error( msg.str() ) );
        }
      }
    }

    /**
     *  \brief Given an index in this mask, return whether or not index is a mask point.
     *  WARNING: slow!
     *  NOTE: Not supported for external mask types
     */
    inline bool operator()(const IntVec& point) const
    {
      if(bitValues_ != NULL) {
        const int position = find_position(point);
        unsigned int * const block = bitValues_ + find_block(position);
        const int bitPosition = find_bit_position(position);
        //update block with new point
        return !!(*block & (1 << bitPosition));
      }
      else {
        std::ostringstream msg;
        msg << "Unsupported attempt to add points to a mask of type ( "
            << DeviceTypeTools::get_memory_type_description(deviceIndex_)
            << " )\n" << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
        throw(std::runtime_error(msg.str()));
      }
    }

    /**
     * \brief Iterator constructs for traversing memory windows.
     * Note: Iteration is not directly supported for external mask types.
     */
    inline const_iterator begin(const MemoryWindow & window) const {
      if( bitValues_ == NULL) {
        std::ostringstream msg;
        msg << "Mask type ( "
            << DeviceTypeTools::get_memory_type_description(deviceIndex_) << " ) ,"
            << " does not support direct iteration, and has no local consumer mask allocated.\n"
            << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
        throw(std::runtime_error(msg.str()));
      }
      return const_iterator(bitValues_, window);
    }

    inline const_iterator end(const MemoryWindow & window) const
    {
      if(bitValues_ != NULL ) {
        int extent = window.extent(0) * window.extent(1) * window.extent(2);
        const_iterator i(bitValues_, window);
        return i + extent;
      } else {
        std::ostringstream msg;
        msg << __FILE__ << " : " << __LINE__ << std::endl
            << "Unsupported request for const_iterator to mask type ( "
            << DeviceTypeTools::get_memory_type_description(deviceIndex_) << " )"
            << "\t - No consumer allocated." << std::endl;
        throw std::runtime_error( msg.str() );
      }
    }

    inline void add_consumer( const short int consumerDeviceIndex )
    {
      // CPU Masks are always exist and hence create only GPU Masks.
      if( consumerDeviceIndex == CPU_INDEX )  return;
#     ifdef ENABLE_CUDA
      else if( IS_GPU_INDEX(consumerDeviceIndex) ){
        //CPU Mask needs to be available on GPU
        ema::cuda::CUDADeviceInterface& CDI = ema::cuda::CUDADeviceInterface::self();

        //Check to see if GPU mask memory exists
        if(consumerBitValues_.find(consumerDeviceIndex) == consumerBitValues_.end()) {
          consumerBitValues_[consumerDeviceIndex] = Pool<unsigned int>::get(consumerDeviceIndex, size_);
          myConsumerBitValues_[consumerDeviceIndex] = consumerBitValues_[consumerDeviceIndex];
        };

        CDI.memcpy_to((void*)consumerBitValues_[consumerDeviceIndex], bitValues_, bytes_, consumerDeviceIndex);
        hasgpuConsumer_ = true;
        return;
      }
#     endif // ENABLE_CUDA
      else{
        std::ostringstream msg;
        msg << "Failed call to add_consumer on Spatial Mask, unknown destination device type : " << consumerDeviceIndex << std::endl
            << "Ensure that you are compiling spatial ops with the proper end device support\n"
            << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
        throw(std::runtime_error(msg.str()));
      }
    }

    inline bool find_consumer(const short int consumerDeviceIndex) const
    {
      //Check for local allocation
      if( consumerDeviceIndex == CPU_INDEX ) return true;
#     ifdef ENABLE_CUDA
      else if( IS_GPU_INDEX(consumerDeviceIndex) ){
        return consumerBitValues_.find( consumerDeviceIndex ) != consumerBitValues_.end();
      }
#     endif // ENABLE_CUDA
      else
      {
        std::ostringstream msg;
        msg << "Failed call to find_consumer on SpatialMask, unknown destination device type : " << consumerDeviceIndex << std::endl
            << "Ensure that you are compiling spatial ops with the proper end device support\n"
            << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
        throw(std::runtime_error(msg.str()));
      }
    }

    inline bool has_consumers()
    {
      return consumerBitValues_.size() > 0;
    }

    inline short int active_device_index() const { return deviceIndex_; };

    inline const unsigned int * mask_values( const short int consumerDeviceIndex = CPU_INDEX ) const
    {
      if( consumerDeviceIndex == CPU_INDEX ){
#       ifndef NDEBUG
        if( bitValues_ == NULL ){
          std::ostringstream msg;
          msg << "Request for consumer mask pointer on a CPU_INDEX for which it has not been allocated\n"
              << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
          throw(std::runtime_error(msg.str()));
        }
#       endif
        return bitValues_;
      }
      else if( IS_GPU_INDEX(consumerDeviceIndex) ){
#       ifdef ENABLE_CUDA
        ConsumerMap::const_iterator citer = consumerBitValues_.find(consumerDeviceIndex);
        if( citer != consumerBitValues_.end() ){
#         ifndef NDEBUG
          if(citer->second == NULL) {
            std::ostringstream msg;
            msg << "Request for consumer mask pointer on a GPU index for which it has not been allocated\n"
                << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
            throw(std::runtime_error(msg.str()));
          }
#         endif
          return citer->second;
        }
#       endif
      }
      else{
        std::ostringstream msg;
        msg << "Request for consumer mask pointer to unknown or unsupported device\n"
            << "\t - " << __FILE__ << " : " << __LINE__ << std::endl;
        throw(std::runtime_error(msg.str()));
      }
      return NULL;   // should never get here.  This line is just to eliminate compiler warnings.
    }
  };

} // namespace SpatialOps

#endif // SpatialOps_BitField_h
