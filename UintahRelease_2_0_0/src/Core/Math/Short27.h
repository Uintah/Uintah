/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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


#ifndef __SHORT27_H__
#define __SHORT27_H__

#include <Core/Geometry/Vector.h>
#include <Core/Util/Assert.h>
#include <Core/Disclosure/TypeUtils.h> // for get_funTD(Short27) prototype

#include <cmath>
#include <iosfwd>
#include <string>
#include <vector>

#include <Core/Math/TntJama/tnt.h>

namespace Uintah {

class Short27 {

 private:
  short s[27];

 public:
  // constructors
  inline Short27();
  inline ~Short27();

  // access operator
  inline short operator[] (int i) const;
  inline short & operator[] (int i);
  static const std::string& get_h_file_path();
};

inline Short27::Short27()
{
  for(int i=0;i<27;i++){
        s[i] = 0;
  }
}

inline Short27::~Short27()
{
}

inline short Short27::operator [] (int i) const
{
  // Access the i component
  return s[i];
}

inline short & Short27::operator [] (int i)
{
  // Access the i component
  return s[i];
}

} // End namespace Uintah

// Added for compatibility with core types
#include <Core/Datatypes/TypeName.h>

namespace Uintah {

              void                    swapbytes( Uintah::Short27 & s );
  template<>  const std::string       find_type_name( Uintah::Short27 * );
              const TypeDescription * get_type_description( Uintah::Short27 * );


} // namespace Uintah

#endif  // __SHORT27_H__
