/**
 * The MIT License
 *
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
 *


\file   field_creation.cpp
\date   Jul 2, 2014
\author "James C. Sutherland"


\page example-field-creation Basics of field creation

# Goal of this example
This example will show how to create a field in SpatialOps

# Key Concepts

 -# A \link SpatialOps::SpatialField SpatialField \endlink is
comprised of a few things:

  - A \link SpatialOps::MemoryWindow MemoryWindow \endlink object
    describes the logical layout of the field in memory.  Among other things, a
    window contains a description of the domain size.

  - A \link SpatialOps::BoundaryCellInfo BoundaryCellInfo \endlink
    object describes the behavior of a field at a boundary.
    For face fields on structured meshes, the (+) face on the side of the
    domain has an extra storage location.
    We will explore this more later.

  - A \link SpatialOps::GhostData GhostData \endlink
    object describes the number of ghost cells on each face of the domain.
    These ghost cells are used to exchange data between patches on different nodes
    and for stencil calculations.
    We will explore this more later.

  - A pointer to the raw block of memory for the field and its
    StorageMode (ExternalStorage or InternalStorage)
    This can either be supplied (for externally managed memory) or it will be
    created internally.  For internally created fields, the supplied memory
    block will be ignored and can be NULL.

  - The location of the field (CPU, GPU, etc.).  Fields can be created on a
    CPU, GPU, etc. and moved between these devices. In fact, a field can be on multiple
    devices at once.

  For now, don't worry about these details - they will be covered in more detail
  in later examples.

 -# A \link SpatialOps::SpatialField SpatialField \endlink is
  strongly typed.  There are several supported field types defined for fields
  associated with structured meshes listed \link fieldtypes here\endlink.
  This will also be explored more in later examples.

 -# The \link SpatialOps::SpatialFieldStore SpatialFieldStore\endlink can be
    used to quickly build fields.  It returns
    \link SpatialOps::SpatFldPtr SpatFldPtr \endlink objects that have pointer
    semantics and are reference-counted.

# Example Code
\c examples/field_creation.cpp
\include field_creation.cpp

*/

#include <spatialops/structured/FVStaggered.h>  // everything required to build fields on structured meshes

using namespace SpatialOps;

// If we are compiling with GPU CUDA support, create fields on the device.
// Otherwise, create them on the host.
#ifdef ENABLE_CUDA
# define LOCATION GPU_INDEX
#else
# define LOCATION CPU_INDEX
#endif

int main()
{
  // SVolField = Scalar Volume Field (non-staggered, cell-centered field)
  typedef SVolField FieldT;

  //----------------------------------------------------------------------------
  // Use default values to create objects required to build a field:

  // Ghost cells are needed by some applications and operations
  // In general, set ghost cells to zero, unless they are needed:
  const GhostData nghost(0);

  // Determine if we have physical boundaries present on each right/positive (+) face.
  const bool bcx=true, bcy=true, bcz=true;
  const BoundaryCellInfo bcInfo = BoundaryCellInfo::build<FieldT>( bcx, bcy, bcz );

  // Define the size of the field (nx,ny,nz)
  const IntVec fieldDim( 10, 9, 8 );

  // Construct a memory window (logical extents of a field) from dimensions,
  //  ghost cell data, and boundary cell information
  const MemoryWindow window( get_window_with_ghost( fieldDim, nghost, bcInfo ) );

  //----------------------------------------------------------------------------
  // Create a field from scratch
  //  parameters:
  //   window          : logical extents for the field
  //   bcInfo          : boundary cell information for the field
  //   nghost          : ghost cell data for the field
  //   NULL            : externally allocated memory (not needed here; hence, NULL)
  //   InternalStorage : internally manage memory (could also be ExternalStorage)
  //   LOCATION        : CPU or GPU memory
  FieldT f( window, bcInfo, nghost, NULL, InternalStorage, LOCATION );

  //----------------------------------------------------------------------------
  // Create a field from a prototype using the "SpatialFieldStore." SpatFldPtr has
  // regular pointer semantics but is a reference-counted pointer.
  SpatFldPtr<FieldT> f2 = SpatialFieldStore::get<FieldT>(f); // field with same layout as "f"

  return 0;
}


