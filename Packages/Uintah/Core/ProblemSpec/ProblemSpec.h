#ifndef UINTAH_HOMEBREW_ProblemSpec_H
#define UINTAH_HOMEBREW_ProblemSpec_H

#include <Packages/Uintah/Core/ProblemSpec/Handle.h>
#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

#include <string>
#include <vector>
#include <map>

class DOMNode;

namespace SCIRun {
  class IntVector;
  class Vector;
  class Point;
}

namespace Uintah {

class TypeDescription;

using std::string;
using std::vector;
using std::map;
using namespace SCIRun;

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
      ProblemSpec(const DOMNode* node, bool doWrite=true);
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
      void require(const std::string& name, IntVector& value);
      void require(const std::string& name, Vector& value);
      void require(const std::string& name, Point& value);
      void require(const std::string& name, vector<double>& value);
      void require(const std::string& name, vector<int>& value); 

   // Get any optional attributes associated with a tag

      void requireOptional(const std::string& name, std::string& value);
      ProblemSpecP getOptional(const std::string& name, std::string& value);

      
      // these functions look for the value of tag named name and passes
      // it back into value.  Returns 'this' if found, otherwise null
      ProblemSpecP get(const std::string& name, double& value);
      ProblemSpecP get(const std::string& name, int& value);
      ProblemSpecP get(const std::string& name, bool& value);
      ProblemSpecP get(const std::string& name, std::string& value);
      ProblemSpecP get(const std::string& name, IntVector& value);
      ProblemSpecP get(const std::string& name, Vector& value);
      ProblemSpecP get(const std::string& name, Point& value);
      ProblemSpecP get(const std::string& name, vector<double>& value);   
      ProblemSpecP get(const std::string& name, vector<int>& value); 

      // these functions look for the value of tag named name and passes
      // it back into value.  If the value isn't there it will create a 
      // node and insert it based on the default value
      ProblemSpecP getWithDefault(const std::string& name, double& value, double defaultVal);
      ProblemSpecP getWithDefault(const std::string& name, int& value, int defaultVal);
      ProblemSpecP getWithDefault(const std::string& name, bool& value, bool defaultVal);
      ProblemSpecP getWithDefault(const std::string& name, std::string& value, std::string defaultVal);
      ProblemSpecP getWithDefault(const std::string& name, IntVector& value, IntVector defaultVal);
      ProblemSpecP getWithDefault(const std::string& name, Vector& value, Vector& defaultVal);
      ProblemSpecP getWithDefault(const std::string& name, Point& value, Point& defaultVal);
      ProblemSpecP getWithDefault(const std::string& name, vector<double>& value, vector<double> defaultVal);   
      ProblemSpecP getWithDefault(const std::string& name, vector<int>& value, vector<int> defaultVal); 

      void getAttributes(std::map<std::string,std::string>& value);
      bool getAttribute(const std::string& value, std::string& result);

      static const TypeDescription* getTypeDescription();
      
      DOMNode* getNode() const {
	 return d_node;
      }

      void writeMessages(bool doWrite) {
        d_write = doWrite;
      }

      bool doWriteMessages() const
      { return d_write; }
   private:
      ProblemSpec(const ProblemSpec&);
      ProblemSpec& operator=(const ProblemSpec&);
      
      DOMNode* d_node;
      bool d_write;
   };
   
} // End namespace Uintah

#endif
