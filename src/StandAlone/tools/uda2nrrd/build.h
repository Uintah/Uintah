/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UDA2NRRD_BUILD_FUNCTIONS_H
#define UDA2NRRD_BUILD_FUNCTIONS_H

#include <StandAlone/tools/uda2nrrd/QueryInfo.h>

#include <StandAlone/tools/uda2nrrd/Args.h>

#include <Core/Datatypes/Field.h>

using namespace SCIRun;
using namespace Uintah;

// Force build_field() to NOT be inlined... so that the symbols will
// actually be created so that they can be used during linking

#define NO_INLINE __attribute__((noinline))

template <class T, class VarT, class FIELD>
NO_INLINE
void
build_field( QueryInfo &qinfo,
             IntVector& offset,
             T& /* data_type */,
             VarT& /*var*/,
             FIELD *sfield,
             const Args & args );

GridP
build_minimal_patch_grid( GridP oldGrid );

template<class T, class VarT, class FIELD>
void build_patch_field( QueryInfo& qinfo,
                        const Patch* patch,
                        IntVector& offset,
                        FIELD* field,
                        const Args & args );

template <class T, class VarT, class FIELD, class FLOC>
NO_INLINE
void
build_combined_level_field( QueryInfo &qinfo,
                            IntVector& offset,
                            FIELD *sfield,
                            const Args & args );

#endif // UDA2NRRD_BUILD_FUNCTIONS_H
