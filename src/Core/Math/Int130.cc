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

//  Int130.cc

#include <Core/Math/Int130.h>
#include <Core/Math/CubeRoot.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <Core/Util/Endian.h>


#include <cstdlib>

using namespace Uintah;
using namespace std;


// Added for compatibility with Core types
namespace Uintah {

using std::string;

template<> const string find_type_name(Int130*)
{
  static const string name = "Int130";
  return name;
}

 
// needed for bigEndian/littleEndian conversion
void swapbytes( Uintah::Int130& s){
  int *p = (int *)(&s);
  SWAP_4(*p);   SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
  SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p); SWAP_4(*++p);
}

} // namespace Uintah

namespace Uintah {
MPI_Datatype makeMPI_Int130()
{
   ASSERTEQ(sizeof(Int130), sizeof(int)*400);

   MPI_Datatype mpitype;
   Uintah::MPI::Type_vector(1, 400, 400, MPI_SHORT, &mpitype);
   Uintah::MPI::Type_commit(&mpitype);

   return mpitype;
}

const TypeDescription* fun_getTypeDescription(Int130*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription(TypeDescription::Int130, "Int130", true,
                                &makeMPI_Int130);
  }
  return td;
}

} // End namespace Uintah

