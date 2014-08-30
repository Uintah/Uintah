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

#ifndef SpatialOps_SpatialMask_h
#define SpatialOps_SpatialMask_h

#include <vector>
#include <cassert>

#include <spatialops/SpatialOpsConfigure.h>

#include <spatialops/structured/BitField.h>
#include <spatialops/structured/SpatialField.h>

namespace SpatialOps{

  /**
   *  \class SpatialMask
   *
   *  \brief Abstracts a mask.
   *
   *  Use SpatialMask when using masks with Nebo.
   *
   *  SpatialMasks can be used in Nebo cond.
   *  See structured/test/testMask.cpp for examples.
   *
   *  Constructing a SpatialMask requires a prototype field and a list of (IntVec) points.
   *
   *  The valid ghost cells of the SpatialMask match its prototype field.
   *
   *  Points are indexed from the interior of the MemoryWindow (does not include ghost cells).
   *  Ghost cells on negative faces therefore have negative indices.
   *  Thus, if there is at least one valid on every face, the point (-1,-1,-1) is valid.
   *
   *  The points in the list become the mask points (return true).
   *  All valid points not in the list are not mask points (return false).
   *
   *  SpatialMask supports Nebo GPU execution.
   *  However, every SpatialMask currently must be constructed on the CPU and *explicitly* copied
   *  to the GPU with the add_consumer() method.
   *
   *  \tparam SpatialField - the underlying fieldtype of the mask
   *
   *  \par Related classes:
   *   - \ref SpatialField
   *   - \ref MemoryWindow
   */
  template<typename FieldType>
  class SpatialMask
  {
  public:

    typedef FieldType field_type;
    typedef SpatialMask<FieldType> mask_type;
    typedef MemoryWindow memory_window;
    typedef ConstMaskIterator const_iterator;

  private:

    MemoryWindow maskWindow_;	        ///< Full representation of the window to the mask ( includes ghost cells )
    MemoryWindow interiorMaskWindow_;  ///< Window representation sans ghost cells.

    const BoundaryCellInfo bcInfo_;     ///< information about this field's behavior on a boundary
    const GhostData ghosts_;          ///< The total number of ghost cells on each face of this field.
    GhostData validGhosts_;           ///< The number of valid ghost cells on each face of this field.

    std::vector<IntVec> points_;
    BitField bitField_;

  public:

    /**
     *  \brief Construct a SpatialMask
     *  \param window the MemoryWindow that specifies this field
     *         including ghost cells.
     *  \param bc information on boundary treatment for this field
     *  \param ghosts information on ghost cells for this field
     *  \param points points in the mask
     */
    SpatialMask(const MemoryWindow & window,
                const BoundaryCellInfo & bc,
                const GhostData & ghosts,
                const std::vector<IntVec> & points)
      : maskWindow_(window),
        interiorMaskWindow_(MemoryWindow(window.glob_dim(),
                                         window.offset() + ghosts.get_minus(),
                                         window.extent() - ghosts.get_minus() - ghosts.get_plus())),
        bcInfo_(bc),
        ghosts_(ghosts),
        validGhosts_(ghosts),
        points_(points),
        bitField_(points_,
                  interiorMaskWindow_,
                  validGhosts_)
    {};

    /**
     *  \brief Construct a SpatialMask
     *  \param prototype field to copy size information from
     *  \param points points in the mask
     */
    SpatialMask(const FieldType & prototype,
                const std::vector<IntVec> & points)
      : maskWindow_(prototype.window_with_ghost()),
        interiorMaskWindow_(prototype.window_without_ghost()),
        bcInfo_(prototype.boundary_info()),
        ghosts_(prototype.get_ghost_data()),
        validGhosts_(prototype.get_valid_ghost_data()),
        points_(points),
        bitField_(points_,
                  interiorMaskWindow_,
                  validGhosts_)
    {};

    /**
     *  \brief Shallow copy constructor.  This results in two fields
     *  that share the same underlying memory.
     */
    SpatialMask(const SpatialMask& other)
      : maskWindow_(other.maskWindow_),
        interiorMaskWindow_(other.interiorMaskWindow_),
        bcInfo_(other.bcInfo_),
        ghosts_(other.ghosts_),
        validGhosts_(other.validGhosts_),
        points_(other.points_),
        bitField_(other.bitField_)
    {};

    /**
     *  \brief Shallow copy constructor with new window.
     */
    SpatialMask(const MemoryWindow& window,
                const SpatialMask& other)
      : maskWindow_(window),
        interiorMaskWindow_(other.interiorMaskWindow_), // This should not be used!
        bcInfo_(other.bcInfo_.limit_by_extent(window.extent())),
        ghosts_(other.ghosts_.limit_by_extent(window.extent())),
        validGhosts_(other.ghosts_.limit_by_extent(window.extent())),
        points_(other.points_),
        bitField_(other.bitField_)
    {
      // ensure that we are doing sane operations with the new window:
#     ifndef NDEBUG
      assert( window.sanity_check() );

      const MemoryWindow& pWindow = other.window_with_ghost();
      for( size_t i=0; i<3; ++i ){
        assert( window.extent(i) + window.offset(i) <= pWindow.glob_dim(i) );
        assert( window.offset(i) < pWindow.glob_dim(i) );
      }
#     endif
    };

    template<typename PrototypeType>
    SpatialMask<FieldType>
    static inline build(const PrototypeType & prototype,
                        const std::vector<IntVec> & points) {
      return SpatialMask(create_new_memory_window<FieldType, PrototypeType>(prototype),
                         create_new_boundary_cell_info<FieldType, PrototypeType>(prototype),
                         prototype.get_valid_ghost_data(),
                         points);
    }

    ~SpatialMask() {};

    /**
     *  \brief return reference to list of points in given list
     *  NOTE: Not supported for external field types
     */
    inline const std::vector<IntVec> & points(void) const
    {
      return points_;
    };

    /**
     *  \brief Given an index in this mask, return whether or not index is a mask point.
     *  WARNING: slow!
     *  NOTE: Not supported for external field types
     */
    inline bool operator()(const size_t i, const size_t j, const size_t k) const
    {
      return operator()(IntVec(i,j,k));
    };

    /**
     *  \brief Given an index in this mask, return whether or not index is a mask point.
     *  WARNING: slow!
     *  NOTE: Not supported for external field types
     */
    inline bool operator()(const IntVec& ijk) const { return bitField_(ijk); };

    /**
     * \brief Iterator constructs for traversing memory windows.
     * Note: Iteration is not directly supported for external field types.
     */
    inline const_iterator begin() const { return bitField_.begin(maskWindow_); };

    inline const_iterator end() const { return bitField_.end(maskWindow_); };

    inline const_iterator interior_begin() const
    {
      return bitField_.begin(interiorMaskWindow_);
    };

    inline const_iterator interior_end() const
    {
      return bitField_.end(interiorMaskWindow_);
    };

    inline void add_consumer(const short int consumerDeviceIndex)
    {
      bitField_.add_consumer(consumerDeviceIndex);
    };

    inline bool find_consumer(const short int consumerDeviceIndex) const
    {
      return bitField_.find_consumer(consumerDeviceIndex);
    };

    inline bool has_consumers() { return bitField_.has_consumers(); };

    inline const BoundaryCellInfo& boundary_info() const{ return bcInfo_; };

    inline const MemoryWindow& window_without_ghost() const { return interiorMaskWindow_; };

    inline const MemoryWindow& window_with_ghost() const { return maskWindow_; };

    inline short int active_device_index() const { return bitField_.active_device_index(); };

    inline const unsigned int * mask_values(const short int consumerDeviceIndex = 0) const
    {
      return bitField_.mask_values(consumerDeviceIndex);
    };

    inline const GhostData& get_ghost_data() const{ return ghosts_; };

    inline const GhostData& get_valid_ghost_data() const{ return validGhosts_; };

    /**
     * @brief Obtain a child field that is reshaped.
     * @param extentModify the amount to modify the extent of the current field by
     * @param shift the number of grid points to shift the current field by
     * @return the reshaped child field
     *
     * The memory is the same as the parent field, but windowed differently.
     * Note that a reshaped field is considered read-only and you cannot obtain
     * interior iterators for these fields.
     */
    inline mask_type reshape(const IntVec& extentModify,
                             const IntVec& shift) const
    {
      MemoryWindow w(maskWindow_.glob_dim(),
                     maskWindow_.offset() + shift,
                     maskWindow_.extent() + extentModify);
      return mask_type(w, *this);
    };
  };

} // namespace SpatialOps

#endif // SpatialOps_SpatialMask_h
