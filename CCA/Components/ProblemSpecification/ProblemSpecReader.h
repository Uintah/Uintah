#ifndef __PROBLEM_SPEC_READER_H__ 
#define __PROBLEM_SPEC_READER_H__

#include <Packages/Uintah/CCA/Ports/ProblemSpecInterface.h>
#include <string>

namespace Uintah {
      
   class ProblemSpecReader : public ProblemSpecInterface {
   public:
      ProblemSpecReader(const std::string& filename);
      ~ProblemSpecReader();
      
      virtual ProblemSpecP readInputFile();
      
   private:
      ProblemSpecReader(const ProblemSpecReader&);
      ProblemSpecReader& operator=(const ProblemSpecReader&);
      
      std::string filename;
   };
} // End namespace Uintah

#endif // __PROBLEM_SPEC_READER_H__
