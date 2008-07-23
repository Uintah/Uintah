#ifndef UDA2NRRD_BUILD_FUNCTIONS_H
#define UDA2NRRD_BUILD_FUNCTIONS_H

#include <Packages/Uintah/StandAlone/tools/uda2vis/QueryInfo.h>

#include <Packages/Uintah/StandAlone/tools/uda2vis/Args.h>

#include <Core/Datatypes/Field.h>

using namespace SCIRun;
using namespace Uintah;

#ifdef __GNUC__
// Force build_field() to NOT be inlined... so that the symbols will
// actually be created so that they can be used during linking

#  define NO_INLINE __attribute__((noinline))
#else
#  define NO_INLINE
#endif

template <class T, class VarT, class FIELD>
NO_INLINE
void
build_field( QueryInfo &qinfo,
             IntVector& offset,
             T& /* data_type */,
             VarT& /*var*/,
             FIELD *sfield,
             const Args & args,
			 int patchNo );

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
