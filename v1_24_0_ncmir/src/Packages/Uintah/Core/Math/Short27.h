//  Short27.h
//  class Short27
//    Short27 data type -- holds 27 shorts

#ifndef __SHORT27_H__
#define __SHORT27_H__

#include <Core/Geometry/Vector.h>
#include <Core/Util/Assert.h>
#include <Core/share/share.h>

#include <math.h>
#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
  class TypeDescription;
  class Piostream;
}
namespace Uintah {
using namespace SCIRun;

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
  static const string& get_h_file_path();
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

inline short &Short27::operator [] (int i)
{
  // Access the i component
  return s[i];
}

} // End namespace Uintah

// Added for compatibility with core types
#include <Core/Datatypes/TypeName.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>
namespace SCIRun {
  using namespace Uintah;
void swapbytes( Uintah::Short27& s );
template<> const string find_type_name(Short27*);
const TypeDescription* get_type_description(Short27*);
void Pio( Piostream&, Uintah::Short27& );

} // namespace SCIRun

#endif  // __SHORT27_H__
