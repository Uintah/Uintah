#ifndef UINTAH_HOMEBREW_TypeUtils_H
#define UINTAH_HOMEBREW_TypeUtils_H

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
   const TypeDescription* fun_getTypeDescription(SCICore::Geometry::Point*);
   const TypeDescription* fun_getTypeDescription(SCICore::Geometry::Vector*);
   
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/05/20 08:09:29  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
//

#endif

