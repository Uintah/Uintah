#ifndef UINTAH_HOMEBREW_ProblemSpec_H
#define UINTAH_HOMEBREW_ProblemSpec_H

#include <Uintah/Grid/Handle.h>
#include <Uintah/Grid/RefCounted.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <string>

#include <util/PlatformUtils.hpp>
#include <parsers/DOMParser.hpp>
#include <dom/DOM_Node.hpp>
#include <dom/DOM_NamedNodeMap.hpp>

namespace Uintah {
    namespace Grid {
	class TypeDescription;
    }
namespace Interface {

using Uintah::Grid::RefCounted;
using Uintah::Grid::TypeDescription;
using Uintah::Interface::ProblemSpecP;


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
    ProblemSpec();
    virtual ~ProblemSpec();
    void setDoc(const DOM_Document& doc);
    void setNode(const DOM_Node& node);

    ProblemSpecP findBlock(const std::string& name) const;
    DOM_Node findNode(const std::string &name, DOM_Node node) const;

    void require(const std::string& name, double& value);
    void require(const std::string& name, int& value);
    void require(const std::string& name, bool& value);
    void require(const std::string& name, std::string& value);

    ProblemSpecP get(const std::string& name, double& value);
    ProblemSpecP get(const std::string& name, int& value);
    ProblemSpecP get(const std::string& name, bool& value);
    ProblemSpecP get(const std::string& name, std::string& value);

    static const TypeDescription* getTypeDescription();
private:
    ProblemSpec(const ProblemSpec&);
    ProblemSpec& operator=(const ProblemSpec&);

    DOM_Document d_doc;
    DOM_Node d_node;
 
};

} // end namespace Interface
} // end namespace Uintah

//
// $Log$
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
