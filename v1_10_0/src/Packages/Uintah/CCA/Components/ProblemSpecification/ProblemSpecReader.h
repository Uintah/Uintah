#ifndef __PROBLEM_SPEC_READER_H__ 
#define __PROBLEM_SPEC_READER_H__

#include <Packages/Uintah/CCA/Ports/ProblemSpecInterface.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <string>

namespace Uintah {
      
   class ProblemSpecReader : public ProblemSpecInterface {
   public:
      ProblemSpecReader(const std::string& filename);
      ~ProblemSpecReader();

      // be sure to call releaseDocument on this ProblemSpecP
      virtual ProblemSpecP readInputFile();

      // replaces <include> tag with xml file tree
      static void resolveIncludes(ProblemSpecP params);
   private:
      ProblemSpecReader(const ProblemSpecReader&);
      ProblemSpecReader& operator=(const ProblemSpecReader&);
      
      std::string filename;
   };
} // End namespace Uintah

#endif // __PROBLEM_SPEC_READER_H__
