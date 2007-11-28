#ifndef UDA2NRRD_GET_FUNCTIONS_H
#define UDA2NRRD_GET_FUNCTIONS_H

#include <StandAlone/tools/uda2nrrd/Args.h>
#include <StandAlone/tools/uda2nrrd/QueryInfo.h>

template<class T>                           void handleVariable( QueryInfo &    qinfo,
                                                                 IntVector &    low,
                                                                 IntVector &    hi,
                                                                 IntVector &    range,
                                                                 BBox &         box,
                                                                 const string & filename,
                                                                 const Args   & args );

template <class T, class VarT, class FIELD> void handlePatchData( QueryInfo & qinfo,
                                                                  IntVector& offset,
                                                                  FIELD* sfield,
                                                                  const Patch* patch,
                                                                  const Args & args );

#endif // UDA2NRRD_GET_FUNCTIONS_H


