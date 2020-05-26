/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef UINTAH_HOMEBREW_ListOfCellsIterator_H
#define UINTAH_HOMEBREW_ListOfCellsIterator_H

#include <climits>
#include <vector>

#include <Core/Geometry/IntVector.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Grid/Variables/BaseIterator.h>
#include <Core/Grid/Variables/Iterator.h>
#include <signal.h>

#include <sci_defs/kokkos_defs.h>

#if defined( UINTAH_ENABLE_KOKKOS )
#include <Kokkos_Core.hpp>
#endif

#include <Core/Parallel/LoopExecution.hpp>

namespace Uintah {

  /**************************************

    CLASS
    ListOfCellsIterator

    Base class for all iterators

    GENERAL INFORMATION

    ListOfCellsIterator.h

    Justin Luitjens
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


    KEYWORDS
    ListOfCellsIterator

    DESCRIPTION
    Base class for all iterators.

    WARNING

   ****************************************/

  class ListOfCellsIterator : public BaseIterator {

    friend std::ostream& operator<<(std::ostream& out, const Uintah::ListOfCellsIterator& b);

    public:

#if defined( UINTAH_ENABLE_KOKKOS )
    ListOfCellsIterator(int size) : mySize(0), index_(0), listOfCells_("primary_ListIterator_BCview", size+1)
    {
      listOfCells_(mySize) = int_3(INT_MAX, INT_MAX, INT_MAX);
    }
#else
    ListOfCellsIterator(int size) : mySize(0), index_(0), listOfCells_(size+1)
    {
      listOfCells_[mySize] = int_3(INT_MAX, INT_MAX, INT_MAX);
    }
#endif

    ListOfCellsIterator(const ListOfCellsIterator &copy) : mySize(copy.mySize), index_(0), listOfCells_(copy.listOfCells_)
    {
      reset();
    }

    ListOfCellsIterator(Iterator &copy) : mySize(0)
                                        , index_(0)
#if defined( UINTAH_ENABLE_KOKKOS )
                                        , listOfCells_("iterator_copy_ListIterator_BCview", copy.size()+1)
#else
                                        , listOfCells_(copy.size()+1)
#endif
    {
      int i = 0;

      for ( copy.reset(); !copy.done(); copy++ ) {
        listOfCells_[i] = int_3((*copy)[0], (*copy)[1], (*copy)[2]);
        i++;
      }

      mySize = i;
      listOfCells_[i] = int_3(INT_MAX, INT_MAX, INT_MAX);

      reset();
    }

    /**
     * prefix operator to move the iterator forward
     */
    ListOfCellsIterator& operator++() { index_++; return *this; }

    /**
     * postfix operator to move the iterator forward
     * does not return the iterator because of performance issues
     */
    void operator++( int ) { index_++; }

    /**
     * returns true if the iterator is done
     */    
    bool done() const { return index_ == mySize; }

    /**
     * returns the IntVector that the current iterator is pointing at
     */
    IntVector operator*() const { ASSERT(index_<mySize); return IntVector( listOfCells_[index_][0], listOfCells_[index_][1], listOfCells_[index_][2] ); }

    /**
     * Assignment operator - this is expensive as we have to allocate new memory
     */
    inline Uintah::ListOfCellsIterator& operator=( Uintah::Iterator& copy )
    {
      // delete old iterator
      int i = 0;

      // copy iterator into portable container
      for ( copy.reset(); !copy.done(); copy++ ) {
        listOfCells_[i] = int_3((*copy)[0],(*copy)[1],(*copy)[2]);
        i++;
      }

      mySize=i;
      return *this;
    }

    /**
     * Return the first element of the iterator
     */
    inline IntVector begin() const { return IntVector( listOfCells_[0][0], listOfCells_[0][1], listOfCells_[0][2] ); }

    /**
     * Return one past the last element of the iterator
     */
    inline IntVector end() const { return IntVector( listOfCells_[mySize][0], listOfCells_[mySize][1], listOfCells_[mySize][2] ); }
    
    /**
     * Return the number of cells in the iterator
     */
    inline unsigned int size() const { return mySize; }

    /**
     * adds a cell to the list of cells
     */
    inline void add( const IntVector& c )
    {
      // place at back of list
      listOfCells_[mySize] = int_3( c[0], c[1], c[2] );
      mySize++;

      // read sentinal to list
      listOfCells_[mySize]=int_3( INT_MAX, INT_MAX, INT_MAX );
    }

    /**
     * resets the iterator
     */
    inline void reset() { index_ = 0; }

// Special handling of MemSpace to promote UintahSpaces::HostSpace to Kokkos::HostSpace
// UintahSpaces::HostSpace is not supported with Kokkos::OpenMP and/or Kokkos::CUDA builds
#if defined( UINTAH_ENABLE_KOKKOS )
//    template<typename MemSpace>
//    inline typename std::enable_if<std::is_same<MemSpace, UintahSpaces::HostSpace>::value, Kokkos::View<int_3*, Kokkos::HostSpace> >::type
//    get_ref_to_iterator(){ return listOfCells_; }

    template<typename ExecSpace, typename MemSpace>
    inline typename std::enable_if<std::is_same<MemSpace, UintahSpaces::HostSpace>::value, Kokkos::View<int_3*, Kokkos::HostSpace> >::type
    get_ref_to_iterator(ExecutionObject<ExecSpace, MemSpace>& execObj){ return listOfCells_; }
#else
//    template<typename MemSpace>
//    inline typename std::enable_if<std::is_same<MemSpace, UintahSpaces::HostSpace>::value,  std::vector<int_3>&>::type
//    get_ref_to_iterator(){ return listOfCells_; }

    template<typename ExecSpace, typename MemSpace>
    inline typename std::enable_if<std::is_same<MemSpace, UintahSpaces::HostSpace>::value,  std::vector<int_3>&>::type
    get_ref_to_iterator(ExecutionObject<ExecSpace, MemSpace>& execObj){ return listOfCells_; }
#endif

#if defined( _OPENMP ) && defined( KOKKOS_ENABLE_OPENMP )
//    template<typename MemSpace>
//    inline typename std::enable_if<std::is_same<MemSpace, Kokkos::HostSpace>::value, Kokkos::View<int_3*, Kokkos::HostSpace> >::type
//    get_ref_to_iterator(){ return listOfCells_; }

    template<typename ExecSpace, typename MemSpace>
    inline typename std::enable_if<std::is_same<MemSpace, Kokkos::HostSpace>::value, Kokkos::View<int_3*, Kokkos::HostSpace> >::type
    get_ref_to_iterator(ExecutionObject<ExecSpace, MemSpace>& execObj){ return listOfCells_; }
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
//    template<typename MemSpace>
//    inline typename std::enable_if<std::is_same<MemSpace, Kokkos::CudaSpace>::value, Kokkos::View<int_3*, Kokkos::CudaSpace> >::type
//    get_ref_to_iterator() {
//      if ( copied_to_gpu ) {
//        return listOfCells_gpu;
//      }
//      else {
//        listOfCells_gpu = Kokkos::View<int_3*, Kokkos::CudaSpace>( "gpu_listOfCellsIterator", listOfCells_.size() );
//        Kokkos::deep_copy( listOfCells_gpu, listOfCells_ );
//        copied_to_gpu = true;
//        return listOfCells_gpu;
//      }
//    }

    template<typename ExecSpace, typename MemSpace>
    inline typename std::enable_if<std::is_same<MemSpace, Kokkos::CudaSpace>::value, Kokkos::View<int_3*, Kokkos::CudaSpace> >::type
    get_ref_to_iterator(ExecutionObject<ExecSpace, MemSpace>& execObj) {
      if ( copied_to_gpu == 2 ) { //if already copied, return
        return listOfCells_gpu;
      }
      else {
        int cur_val = __sync_val_compare_and_swap(&copied_to_gpu, 0, 1);
        if(cur_val == 0){ //comparison was successful and this is a lucky thread that gets to copy the value.
          listOfCells_gpu = Kokkos::View<int_3*, Kokkos::CudaSpace>( "gpu_listOfCellsIterator", listOfCells_.size() );
          cudaStream_t* stream = static_cast<cudaStream_t*>(execObj.getStream());
          cudaMemcpyAsync(listOfCells_gpu.data(), listOfCells_.data(),  listOfCells_.size() * sizeof(int_3), cudaMemcpyHostToDevice, *stream);
          cudaStreamSynchronize(*stream); //Think how cudaStreamSynchronize can be avoided. No other way to set  copied_to_gpu as of now.

          bool success = __sync_bool_compare_and_swap(&copied_to_gpu, 1, 2);
          if(!success){
            printf("Error in copying values. Possible CPU race condition. %s:%d\n", __FILE__, __LINE__);
            exit(1);
          }
          return listOfCells_gpu;
        }
        else{//some other thread already took care. wait until copy is completed.
          while(copied_to_gpu != 2){std::this_thread::yield();}
          return listOfCells_gpu;
        }
      }
    }
#endif

    protected:

    /**
     * Returns a pointer to a deep copy of the virtual class
     * this should be used only by the Iterator class
     */
    ListOfCellsIterator* clone() const
    {
      return scinew ListOfCellsIterator(*this);

    };

    virtual std::ostream& put(std::ostream& out) const
    {
      out << *this;
      return out;
    }

    virtual std::ostream& limits(std::ostream& out) const
    {
      out << begin() << " " << end();
      return out;
    }

    unsigned int mySize{0};
    unsigned int index_{0}; // index into the iterator

#if defined( UINTAH_ENABLE_KOKKOS )
    Kokkos::View<int_3*, Kokkos::HostSpace> listOfCells_;
#else
    std::vector<int_3> listOfCells_{};
#endif

#if defined( HAVE_CUDA ) && defined( KOKKOS_ENABLE_CUDA )
    Kokkos::View<int_3*, Kokkos::CudaSpace> listOfCells_gpu;
    //bool copied_to_gpu{false};
    volatile int copied_to_gpu{0}; //0: not copied, 1: copying, 2: copied
#endif

    private:

    // This old constructor has a static size for portability reasons. It should be avoided since.
#if defined( UINTAH_ENABLE_KOKKOS )
    ListOfCellsIterator() : mySize(0), index_(0), listOfCells_("priv_ListIterator_BCview", 1000)
    {
      listOfCells_(mySize) = int_3(INT_MAX, INT_MAX, INT_MAX);
      std::cout<< "Unsupported constructor, use at your own risk, in Core/Grid/Variables/ListOfCellsIterator.h \n";
    }
#else
    ListOfCellsIterator() : mySize(0), index_(0), listOfCells_(1000)
    {
      listOfCells_[mySize] = int_3(INT_MAX, INT_MAX, INT_MAX);
      std::cout<< "Unsupported constructor, use at your own risk, in Core/Grid/Variables/ListOfCellsIterator.h \n";
    }
#endif

  }; // end class ListOfCellsIterator

} // end namespace Uintah

#endif
