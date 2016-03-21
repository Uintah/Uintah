/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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


#include <Core/Grid/Variables/Vector5.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Util/Endian.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>
#include <sci_defs/mpi_defs.h> // For MPIPP_H on SGI

using namespace Uintah;

namespace Uintah {

  static MPI_Datatype makeMPI_Vector5()
  {
    ASSERTEQ(sizeof(Vector5), sizeof(double)*7);
    MPI_Datatype mpitype;
    MPI::Type_vector(1, 5, 5, MPI_DOUBLE, &mpitype);
    MPI::Type_commit(&mpitype);
    return mpitype;
  }
  
  const TypeDescription* fun_getTypeDescription(Vector5*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = new TypeDescription(TypeDescription::Vector5,
                                  "Vector5", true, 
                                  &makeMPI_Vector5);
    }
    return td;
  }
  
  void swapbytes( Vector5& a) {
    SWAP_8(a.rho);
    SWAP_8(a.momX);
    SWAP_8(a.momY);
    SWAP_8(a.momZ);
    SWAP_8(a.eng);
  }

  std::ostream & operator << (std::ostream &out, const Uintah::Vector5 &a) {
    out << "A.rho: "   << a.rho   << " A.momX: " << a.momX << " A.momY: " << a.momY 
        << " A.momZ: " << a.momZ  << " A.eng: "  << a.eng;
    return out;
  }


} // namespace Uintah

