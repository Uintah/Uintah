/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


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


