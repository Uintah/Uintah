#ifndef UINTAH_HOMEBREW_TypeUtils_H
#define UINTAH_HOMEBREW_TypeUtils_H

class Matrix3;

namespace SCICore {
   namespace Geometry {
      class Point;
      class Vector;
   }
}

namespace Uintah {
   class TypeDescription;
   const TypeDescription* fun_getTypeDescription(double*);
   const TypeDescription* fun_getTypeDescription(bool*);
   const TypeDescription* fun_getTypeDescription(int*);
   const TypeDescription* fun_getTypeDescription(long*);
   const TypeDescription* fun_getTypeDescription(SCICore::Geometry::Point*);
   const TypeDescription* fun_getTypeDescription(SCICore::Geometry::Vector*);
   const TypeDescription* fun_getTypeDescription(Matrix3*);
   
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/07/11 19:58:47  kuzimmer
// modified so that CCVariables and Matrix3 subtype can be recognized in archive reader
//
// Revision 1.2  2000/06/02 17:22:15  guilkey
// Added long_type to the the TypeDescription and TypeUtils.
//
// Revision 1.1  2000/05/20 08:09:29  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
//

#endif

