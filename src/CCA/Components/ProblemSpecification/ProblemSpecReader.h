/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __PROBLEM_SPEC_READER_H__ 
#define __PROBLEM_SPEC_READER_H__

#include <CCA/Ports/ProblemSpecInterface.h>

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <string>
#include <vector>

namespace Uintah {
      
  struct Tag;

  class ProblemSpecReader : public ProblemSpecInterface {
  public:
    ProblemSpecReader();
    ~ProblemSpecReader();

    // Reads in an XML problem spec file.  (See 2nd version below, too).

    virtual ProblemSpecP readInputFile( const std::string      & filename,
                                        const bool               validate = false );

    // This function is designed to remove <Patch> sections of the xml input 'filename' before
    // creating the XML DOM object, where the patches passed in are just the ones assigned to
    // this processor and thus all other <Patch>s are excluded from the DOM.  However, I
    // don't think anyone is using this at this point, and I'm not convinced it is the
    // correct file for this 'fix', as most xml files read by readInputFile() are not the
    // timestep.xml file which has the <Patch>s in it.  Once things settle down and the
    // rest of the XML memory usage problems are fixed, this should be revisited and probably
    // just remove the streaming parsing in this fucntion.

    virtual ProblemSpecP readInputFile( const std::string      & filename,
                                        const std::vector<int> & patches,
                                        const bool               validate = false );

    // Returns the main xml file name.
    virtual std::string getInputFile() { return *d_upsFilename[0]; }

  private:

    ProblemSpecReader(const ProblemSpecReader&);
    ProblemSpecReader& operator=(const ProblemSpecReader&);
    
    ////////////////////////////////////////////////////////////////////////////////
    // Variables:

    // d_upsFilename[0] is the main file... but each subsequent string
    // is the name of an <include>d file.
    std::vector< std::string * > d_upsFilename;

    ProblemSpecP d_xmlData;

    ////////////////////////////////////////////////////////////////////////////////
    // Functions:

    void         parseValidationFile();
    void         validateProblemSpec( ProblemSpecP & prob_spec );
    std::string* findFileNamePtr( const std::string & filename );

    // Replaces <include> tags with xml file tree.
    void resolveIncludes( xmlNode * child, xmlNode * parent, int depth = 0 );

   };

} // End namespace Uintah

#endif // __PROBLEM_SPEC_READER_H__
