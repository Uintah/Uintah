#ifndef __PROBLEM_SPEC_READER_H__ 
#define __PROBLEM_SPEC_READER_H__

#include <Uintah/Interface/ProblemSpec.h>

using Uintah::Interface::ProblemSpecP;
using Uintah::Interface::ProblemSpec;

class ProblemSpecReader {

public:
  ProblemSpecReader();
  ~ProblemSpecReader();
   
  ProblemSpecP readInputFile(const std::string name);

private:
  ProblemSpecReader(const ProblemSpecReader&);
  ProblemSpecReader& operator=(const ProblemSpecReader&);
  
  ProblemSpecP d_prob_spec; 
};

#endif // __PROBLEM_SPEC_READER_H__
