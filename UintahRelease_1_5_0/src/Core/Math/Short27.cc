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

//  Short27.cc

#include <Core/Math/Short27.h>
#include <Core/Math/CubeRoot.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <Core/Util/Endian.h>


#include <cstdlib>

using namespace Uintah;

const string& 
Short27::get_h_file_path() {
  static const string path(SCIRun::TypeDescription::cc_to_h(__FILE__));
  return path;
}

// Added for compatibility with Core types
namespace SCIRun {

using std::string;

template<> const string find_type_name(Short27*)
{
  static const string name = "Short27";
  return name;
}

const TypeDescription* get_type_description(Short27*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("Short27", Short27::get_h_file_path(), "Uintah");
  }
  return td;
}

void
Pio(Piostream& stream, Short27& s)
{
    stream.begin_cheap_delim();
    Pio(stream, s[0]); Pio(stream, s[1]); Pio(stream, s[2]);
    Pio(stream, s[3]); Pio(stream, s[4]); Pio(stream, s[5]);
    Pio(stream, s[6]); Pio(stream, s[7]); Pio(stream, s[8]);
    Pio(stream, s[9]); Pio(stream,s[10]); Pio(stream,s[11]);
    Pio(stream,s[12]); Pio(stream,s[13]); Pio(stream,s[14]);
    Pio(stream,s[15]); Pio(stream,s[16]); Pio(stream,s[17]);
    Pio(stream,s[18]); Pio(stream,s[19]); Pio(stream,s[20]);
    Pio(stream,s[21]); Pio(stream,s[22]); Pio(stream,s[23]);
    Pio(stream,s[24]); Pio(stream,s[25]); Pio(stream,s[26]);
    stream.end_cheap_delim();
}

// needed for bigEndian/littleEndian conversion
void swapbytes( Uintah::Short27& s){
  short *p = (short *)(&s);
  SWAP_2(*p); SWAP_2(*++p); SWAP_2(*++p);
  SWAP_2(*++p); SWAP_2(*++p); SWAP_2(*++p);
  SWAP_2(*++p); SWAP_2(*++p); SWAP_2(*++p);
  SWAP_2(*++p); SWAP_2(*++p); SWAP_2(*++p);
  SWAP_2(*++p); SWAP_2(*++p); SWAP_2(*++p);
  SWAP_2(*++p); SWAP_2(*++p); SWAP_2(*++p);
  SWAP_2(*++p); SWAP_2(*++p); SWAP_2(*++p);
  SWAP_2(*++p); SWAP_2(*++p); SWAP_2(*++p);
  SWAP_2(*++p); SWAP_2(*++p); SWAP_2(*++p);
}

} // namespace SCIRun

namespace Uintah {
MPI_Datatype makeMPI_Short27()
{
   ASSERTEQ(sizeof(Short27), sizeof(short)*27);

   MPI_Datatype mpitype;
   MPI_Type_vector(1, 27, 27, MPI_SHORT, &mpitype);
   MPI_Type_commit(&mpitype);

   return mpitype;
}

const TypeDescription* fun_getTypeDescription(Short27*)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription(TypeDescription::Short27, "Short27", true,
                                &makeMPI_Short27);
  }
  return td;
}

} // End namespace Uintah

