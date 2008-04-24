#ifndef __PROBLEM_SPEC_READER_H__ 
#define __PROBLEM_SPEC_READER_H__

#include <Packages/Uintah/CCA/Components/ProblemSpecification/uintahshare.h>
#include <Packages/Uintah/CCA/Ports/ProblemSpecInterface.h>

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <Core/XMLUtil/XMLUtil.h>

#include <sgi_stl_warnings_off.h>
#include   <string>
#include <sgi_stl_warnings_on.h>


namespace Uintah {
      
  class UINTAHSHARE ProblemSpecReader : public ProblemSpecInterface {
  public:
    ProblemSpecReader( const std::string& upsFilename );
    ~ProblemSpecReader();

    // be sure to call releaseDocument on this ProblemSpecP
    virtual ProblemSpecP readInputFile();

    virtual std::string getInputFile() { return d_upsFilename; }
    // replaces <include> tag with xml file tree
    void resolveIncludes(ProblemSpecP params);

  private:

    ProblemSpecReader(const ProblemSpecReader&);
    ProblemSpecReader& operator=(const ProblemSpecReader&);
    
    std::string d_upsFilename;
    ProblemSpecP d_xmlData;

    /////////////////////////////////////////////

    struct AttributeAndTagBase;
    struct Tag;
    struct Attribute;

    // Currently all ProblemSpecReader's share the validation data...
    // (Pragmatically I use this to not parse the DW created files,
    //  and only parse the original .ups...)
    static Tag * uintahSpec_;
    static Tag * commonTags_;

    // Functions:
    void        parseValidationFile();
    void        validateProblemSpec( ProblemSpecP & prob_spec );

    void        validate( Tag * root, const ProblemSpec * ps, unsigned int level = 0 );

    void        parseTag( Tag * parent, const xmlNode * xmlTag );

    void        validateAttribute( Tag * root, xmlAttr * attr );
    void        validateText( AttributeAndTagBase * root, const string & text );
    bool        validateString( const string & value, const vector<string> & validValues );
    bool        validateBoolean( const string & value );
    void        validateDouble( double value, const vector<string> & validValues );

    Attribute * findAttribute( Tag * root, const string & attrName );
    Tag *       findSubTag( Tag * root, const string & tagName );
   };

} // End namespace Uintah

#endif // __PROBLEM_SPEC_READER_H__
