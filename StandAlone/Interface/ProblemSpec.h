#ifndef UINTAH_HOMEBREW_ProblemSpec_H
#define UINTAH_HOMEBREW_ProblemSpec_H

#include <Uintah/Grid/Handle.h>
#include <Uintah/Grid/RefCounted.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <string>

#ifdef __sgi
#define IRIX
#pragma set woff 1375
#endif
#include <util/PlatformUtils.hpp>
#include <parsers/DOMParser.hpp>
#include <dom/DOM_Node.hpp>
#include <dom/DOM_NamedNodeMap.hpp>
#ifdef __sgi
#pragma reset woff 1375
#endif

namespace SCICore {
   namespace Geometry {
      class IntVector;
      class Vector;
      class Point;
   }
}

namespace Uintah {
   class TypeDescription;

// This is the "base" problem spec.  There should be ways of breaking
// this up

/**************************************

CLASS
   ProblemSpec
   
   Short description...

GENERAL INFORMATION

   ProblemSpec.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Problem_Specification

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   
   class ProblemSpec : public RefCounted {
   public:
      ProblemSpec(const DOM_Node& node);
      virtual ~ProblemSpec();
      
      ProblemSpecP findBlock(const std::string& name) const;
      ProblemSpecP findBlock() const;
      ProblemSpecP findNextBlock(const std::string& name) const;
      ProblemSpecP findNextBlock() const;
      
      std::string getNodeName() const;
      
      void require(const std::string& name, double& value);
      void require(const std::string& name, int& value);
      void require(const std::string& name, bool& value);
      void require(const std::string& name, std::string& value);
      void require(const std::string& name, SCICore::Geometry::IntVector& value);
      void require(const std::string& name, SCICore::Geometry::Vector& value);
      void require(const std::string& name, SCICore::Geometry::Point& value);

   // Get any optional attributes associated with a tag

   void requireOptional(const std::string& name, std::string& value);
   ProblemSpecP getOptional(const std::string& name, std::string& value);

      
      ProblemSpecP get(const std::string& name, double& value);
      ProblemSpecP get(const std::string& name, int& value);
      ProblemSpecP get(const std::string& name, bool& value);
      ProblemSpecP get(const std::string& name, std::string& value);
      ProblemSpecP get(const std::string& name, 
		       SCICore::Geometry::IntVector& value);
      ProblemSpecP get(const std::string& name, 
		       SCICore::Geometry::Vector& value);
      ProblemSpecP get(const std::string& name, 
		       SCICore::Geometry::Point& value);
      
      static const TypeDescription* getTypeDescription();
      
      DOM_Node getNode() const {
	 return d_node;
      }
   private:
      ProblemSpec(const ProblemSpec&);
      ProblemSpec& operator=(const ProblemSpec&);
      
      DOM_Node d_node;
      
   };
   

} // end namespace Uintah

//
// $Log$
// Revision 1.14  2000/05/20 08:09:39  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.13  2000/05/15 19:39:53  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.12  2000/04/27 00:28:38  jas
// Can now read the attributes field of a tag.
//
// Revision 1.11  2000/04/26 06:49:11  sparker
// Streamlined namespaces
//
// Revision 1.10  2000/04/20 22:37:18  jas
// Fixed up the GeometryObjectFactory.  Added findBlock() and findNextBlock()
// to ProblemSpec stuff.  This will iterate through all of the nodes (hopefully).
//
// Revision 1.9  2000/04/12 23:01:55  sparker
// Implemented more of problem spec - added Point and IntVector readers
//
// Revision 1.8  2000/04/12 15:33:49  jas
// Can now read a Vector type [num,num,num] from the ProblemSpec.
//
// Revision 1.7  2000/04/06 02:33:33  jas
// Added findNextBlock which will find all of the tags named name within a
// given block.
//
// Revision 1.6  2000/03/30 20:23:43  sparker
// Fixed compile on SGI
//
// Revision 1.5  2000/03/29 23:48:00  jas
// Filled in methods for extracting data from the xml tree (requires and get)
// and storing the result in a variable.
//
// Revision 1.4  2000/03/29 01:59:59  jas
// Filled in the findBlock method.
//
// Revision 1.3  2000/03/23 20:42:24  sparker
// Added copy ctor to exception classes (for Linux/g++)
// Helped clean up move of ProblemSpec from Interface to Grid
//
// Revision 1.2  2000/03/23 20:00:17  jas
// Changed the include files, namespace, and using statements to reflect the
// move of ProblemSpec from Grid/ to Interface/.
//
// Revision 1.1  2000/03/23 19:47:55  jas
// Moved the ProblemSpec stuff from Grid/ to Interface.
//
// Revision 1.4  2000/03/22 23:41:27  sparker
// Working towards getting arches to compile/run
//
// Revision 1.3  2000/03/21 18:52:11  sparker
// Prototyped header file for new problem spec functionality
//
// Revision 1.2  2000/03/16 22:08:00  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
