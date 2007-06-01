#ifndef __PROBLEM_SPEC_READER_H__ 
#define __PROBLEM_SPEC_READER_H__

#include <CCA/Ports/ProblemSpecInterface.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <CCA/Components/ProblemSpecification/uintahshare.h>
namespace Uintah {
      
   class UINTAHSHARE ProblemSpecReader : public ProblemSpecInterface {
   public:
      ProblemSpecReader(const std::string& filename);
      ~ProblemSpecReader();

      // be sure to call releaseDocument on this ProblemSpecP
      virtual ProblemSpecP readInputFile();

      virtual std::string getInputFile() { return d_filename; }
      // replaces <include> tag with xml file tree
      void resolveIncludes(ProblemSpecP params);
   private:
      ProblemSpecReader(const ProblemSpecReader&);
      ProblemSpecReader& operator=(const ProblemSpecReader&);
      
      std::string d_filename;
      ProblemSpecP d_xmlData;

   };
} // End namespace Uintah

#endif // __PROBLEM_SPEC_READER_H__
