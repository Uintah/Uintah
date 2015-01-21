//  Short27.h
//  class Short27
//    Short27 data type -- holds 27 shorts

#ifndef __SHORT27_H__
#define __SHORT27_H__

#include <SCIRun/Core/Geometry/Vector.h>
#include <SCIRun/Core/Util/Assert.h>
#include <Core/Disclosure/TypeUtils.h> // for get_funTD(Short27) prototype

#include <math.h>
#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
  class TypeDescription;
  class Piostream;
}
#include <Core/Math/TntJama/tnt.h>
#include <Core/Math/uintahshare.h>
namespace Uintah {
using namespace SCIRun;

class UINTAHSHARE Short27 {

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
#include <SCIRun/Core/Datatypes/TypeName.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>
namespace SCIRun {
  using namespace Uintah;
UINTAHSHARE void swapbytes( Uintah::Short27& s );
template<> UINTAHSHARE const string find_type_name(Short27*);
UINTAHSHARE const TypeDescription* get_type_description(Short27*);
UINTAHSHARE void Pio( Piostream&, Uintah::Short27& );

} // namespace SCIRun

#endif  // __SHORT27_H__
