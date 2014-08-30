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

#ifndef NEBO_BASIC_H
#  define NEBO_BASIC_H

#  include <spatialops/SpatialOpsConfigure.h>
#  include <spatialops/structured/IndexTriplet.h>
#  include <spatialops/structured/GhostData.h>
#  include <spatialops/structured/SpatialField.h>
#  include <spatialops/structured/SpatialMask.h>
#  include <spatialops/structured/FVStaggeredFieldTypes.h>
#  include <cmath>
#  include <math.h>
#  include <limits>
#  include <boost/math/special_functions/erf.hpp>

#  ifdef NEBO_REPORT_BACKEND
#     include <iostream>
#  endif
   /* NEBO_REPORT_BACKEND */

#  ifdef ENABLE_THREADS
#     include <spatialops/SpatialOpsTools.h>
#     include <vector>
#     include <boost/bind.hpp>
#     include <spatialops/ThreadPool.h>
#     include <spatialops/structured/IntVec.h>
#     include <spatialops/Semaphore.h>
#  endif
   /* ENABLE_THREADS */

#  ifdef __CUDACC__
#     include <sstream>
#     include <spatialops/structured/MemoryTypes.h>
#  endif
   /* __CUDACC__ */

   namespace SpatialOps {
      /* Meta-programming compiler flags */
      struct All;
      struct InteriorOnly;

      inline GhostData calculate_actual_ghost(bool const useGhost,
                                                          GhostData const & lhs,
                                                          BoundaryCellInfo const & bc,
                                                          GhostData const & rhs) {
        if(bc.has_bc(0) && rhs.get_plus(0) < bc.has_extra(0)) {
          std::ostringstream msg;
          msg << "Nebo error in " << "Nebo Ghost Checking" << ":\n";
          msg << "Not enough valid extra cells to validate all interior ";
          msg << "cells in the X direction";
          msg << "\n";
          msg << "\t - " << __FILE__ << " : " << __LINE__;
          throw(std::runtime_error(msg.str()));;
        };

        if(bc.has_bc(1) && rhs.get_plus(1) < bc.has_extra(1)) {
          std::ostringstream msg;
          msg << "Nebo error in " << "Nebo Ghost Checking" << ":\n";
          msg << "Not enough valid extra cells to validate all interior ";
          msg << "cells in the Y direction";
          msg << "\n";
          msg << "\t - " << __FILE__ << " : " << __LINE__;
          throw(std::runtime_error(msg.str()));;
        };

        if(bc.has_bc(2) && rhs.get_plus(2) < bc.has_extra(2)) {
          std::ostringstream msg;
          msg << "Nebo error in " << "Nebo Ghost Checking" << ":\n";
          msg << "Not enough valid extra cells to validate all interior ";
          msg << "cells in the Z direction";
          msg << "\n";
          msg << "\t - " << __FILE__ << " : " << __LINE__;
          throw(std::runtime_error(msg.str()));;
        };

        return ((useGhost
                 ? min((lhs + point_to_ghost(bc.has_extra())), rhs)
                 : GhostData(IntVec(0, 0, 0),
                                         bc.has_extra()))
                - point_to_ghost(bc.has_extra()));
      };

      template<typename Type1, typename Type2>
       struct NeboFieldCheck;

      template<typename T>
       struct NeboFieldCheck<SpatialOps::SpatialField<SpatialOps::SingleValue, T>,
                             SpatialOps::SpatialField<SpatialOps::SingleValue, T> > {};

      template<typename Type>
       struct NeboFieldCheck<Type, Type> { Type typedef Result; };

      inline IntVec nebo_find_partition(IntVec const & extent,
                                                    int const thread_count) {
         int x = 1;
         int y = 1;
         int z = 1;

         if(thread_count <= extent[2]) { z = thread_count; }
         else if(thread_count <= extent[1]) { y = thread_count; }
         else if(thread_count <= extent[0]) { x = thread_count; };

         return IntVec(x, y, z);
      };

      inline int nebo_partition_count(IntVec const & split) {
         return split[0] * split[1] * split[2];
      };

      inline void nebo_set_up_extents(IntVec const & current,
                                      IntVec const & split,
                                      GhostData & localLimits,
                                      GhostData const & limits) {

        //full extent indexed from 0 rather than DLow (which is nonpositive - zero or below)
        IntVec const fullExtent(limits.get_plus(0) - limits.get_minus(0),
                                limits.get_plus(1) - limits.get_minus(1),
                                limits.get_plus(2) - limits.get_minus(2));

        //sanity checks
#       ifndef NDEBUG
          for( size_t i=0; i<3; ++i ){
            assert( fullExtent[i] >= split[i] );
            assert( split[i] > 0 );
            assert( current[i] < split[i] );
            assert( current[i] >= 0 );
          }
#       endif

        //extent of a partition
        IntVec const stdExtent = fullExtent / split;

        //number of partitions with an extra cell (to cover what is not covered by stdExtent)
        IntVec const nExtra(fullExtent[0] % split[0],
                            fullExtent[1] % split[1],
                            fullExtent[2] % split[2]);

        //number of previous paritions with an extra cell
        IntVec const pastExtra(current[0] < nExtra[0] ? current[0] : nExtra[0],
                               current[1] < nExtra[1] ? current[1] : nExtra[1],
                               current[2] < nExtra[2] ? current[2] : nExtra[2]);

        //does current partition have an extra cell
        IntVec const currentExtra(current[0] < nExtra[0] ? 1 : 0,
                                  current[1] < nExtra[1] ? 1 : 0,
                                  current[2] < nExtra[2] ? 1 : 0);

        //calculate current partition's low and high
        IntVec const low = stdExtent * current + pastExtra;
        IntVec const high = low + stdExtent + currentExtra;

        //shift back to indexing from DLow rather than zero
        localLimits = GhostData(low[0] + limits.get_minus(0),
                                            high[0] + limits.get_minus(0),
                                            low[1] + limits.get_minus(1),
                                            high[1] + limits.get_minus(1),
                                            low[2] + limits.get_minus(2),
                                            high[2] + limits.get_minus(2));
      };

      inline IntVec nebo_next_partition(IntVec const & current,
                                                    IntVec const & split) {
        IntVec result;

        if(current[2] < split[2] - 1)
          result = IntVec(current[0], current[1], 1 + current[2]);
        else if(current[1] < split[1] - 1)
          result = IntVec(current[0], 1 + current[1], 0);
        else result = IntVec(1 + current[0], 0, 0);

        return result;
      };

      template<typename Operand, typename FieldType>
       struct NeboExpression {
         public:
          FieldType typedef field_type;

          Operand typedef Expression;

          NeboExpression(Operand const & given)
          : expr_(given)
          {}

          inline Operand const & expr(void) const { return expr_; }

         private:
          Operand expr_;
      };

      template<typename Operand, typename T>
       struct NeboSingleValueExpression {
         public:
          SpatialOps::SpatialField<SpatialOps::SingleValue, T> typedef field_type;

          Operand typedef Expression;

          NeboSingleValueExpression(Operand const & given)
          : expr_(given)
          {}

          inline Operand const & expr(void) const { return expr_; }

         private:
          Operand expr_;
      };

      template<typename Operand, typename FieldType>
       struct NeboBooleanExpression {
         public:
          FieldType typedef field_type;

          Operand typedef Expression;

          NeboBooleanExpression(Operand const & given)
          : expr_(given)
          {}

          inline Operand const & expr(void) const { return expr_; }

         private:
          Operand expr_;
      };

      template<typename Operand, typename T>
       struct NeboBooleanSingleValueExpression {
         public:
          SpatialOps::SpatialField<SpatialOps::SingleValue, T> typedef field_type;

          Operand typedef Expression;

          NeboBooleanSingleValueExpression(Operand const & given)
          : expr_(given)
          {}

          inline Operand const & expr(void) const { return expr_; }

         private:
          Operand expr_;
      };

      /* Modes: */
      struct Initial;
#     ifdef ENABLE_THREADS
         struct Resize;
#     endif
      /* ENABLE_THREADS */
      struct SeqWalk;
#     ifdef __CUDACC__
        struct GPUWalk;
#     endif
      /* __CUDACC__ */
   } /* SpatialOps */

#endif
/* NEBO_BASIC_H */
