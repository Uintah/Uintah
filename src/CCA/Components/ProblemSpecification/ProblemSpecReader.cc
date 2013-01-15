/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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
#include <CCA/Components/ProblemSpecification/ProblemSpecReader.h>

#include <Core/Parallel/Parallel.h> // Only used for MPI cerr
#include <Core/Parallel/ProcessorGroup.h> // process determination
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <Core/Containers/StringUtil.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FileUtils.h>

#include <Core/Util/Environment.h> // for SCIRUN_SRCDIR

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <list>

#include <cstdio>
#include <cstring>
#include <unistd.h>

#include <libxml/tree.h>
#include <libxml/parser.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

//////////////////////////////////////////////////////////////////////////////////////////////

namespace Uintah {

  struct Attribute;
  struct ChildRequirement;

  typedef Handle<Tag>              TagP;
  typedef Handle<Attribute>        AttributeP;
  typedef Handle<ChildRequirement> ChildRequirementP;
}

//////////////////////////////////////////////////////////////////////////////////////////////

static DebugStream dbg( "ProblemSpecReader", false );
static DebugStream inc_dbg( "ProblemSpecReaderIncludes", false );

//////////////////////////////////////////////////////////////////////////////////////////////
// Utility Functions:

// Prints out 2 spaces for each level of indentation.
static
void
indent( ostream & out, unsigned int depth )
{
  // out << depth << " ";
  for( unsigned int pos = 0; pos < depth; pos++ ) {
    out << "    ";
  }
}

string
getErrorInfo( const xmlNode * node )
{
  ostringstream error;

  if( node->_private == NULL ) {
    // All nodes of the ups_spec.xml will have _private set, but nodes coming from the 
    // .ups file being validated may not.  However, they will have a doc pointer that
    // has the file information.
    //
    // Both ATTRIBUTE and TEXT node line numbers aren't part of those
    // type of 'node's... we have to look at their parents to get the
    // real value.  (This occurs because we cast all nodes into
    // generic xmlNodes before they get to this portion of the code,
    // and this is why we need to do this test.)
    if( node->type == XML_ATTRIBUTE_NODE || node->type == XML_TEXT_NODE ) {
      error << "See file: " << (const char *)( node->doc->URL ) << " (line #" << node->parent->line << ")";
    }
    else {
      error << "See file: " << (const char *)( node->doc->URL ) << " (line #" << node->line << ")";
    }
  }
  else {
    string file = *(string*)(node->_private);
    error << "See file: " << file << " (line #" << node->line << ")";;

  }
  
  return error.str();
}

//////////////////////////////////////////////////////////////////////////////////////////////
// The following section holds structures used in validating the problem spec file.

namespace Uintah {

/////////////////////////////////////////////////////////////////
//
// need_e, type_e, Element, and Tag are all used for validation of XML
//

// MULTIPLE = 0 or more occurrences
enum need_e { OPTIONAL, REQUIRED, MULTIPLE, INVALID_NEED };
// VECTORs are specified as [0.0, 0.0, 0.0] (ie: 3 numbers: used for IntVector)... Note, IntVectors are integer vectors, so
//   probably should have a separate validation that 'integers' are used, but don't do that now...
enum type_e { DOUBLE, INTEGER, STRING, VECTOR, BOOLEAN, NO_DATA, MULTIPLE_DOUBLES, MULTIPLE_INTEGERS, MULTIPLE_VECTORS, INVALID_TYPE };

ostream &
operator<<( ostream & out, const need_e & need )
{
  if(      need == REQUIRED )     { out << "REQUIRED"; }
  else if( need == OPTIONAL )     { out << "OPTIONAL"; }
  else if( need == MULTIPLE )     { out << "MULTIPLE"; }
  else if( need == INVALID_NEED ) { out << "INVALID_NEED"; }
  else {
    out << "Error in need_e '<<' operator: value of 'need': " << (int)need << ", is invalid... \n";
  }
  return out;
}

ostream &
operator<<( ostream & out, const type_e & type )
{
  if     ( type == DOUBLE )  { out << "DOUBLE"; }
  else if( type == INTEGER ) { out << "INTEGER"; }
  else if( type == STRING )  { out << "STRING"; }
  else if( type == VECTOR )  { out << "VECTOR"; }
  else if( type == BOOLEAN ) { out << "BOOLEAN"; }
  else if( type == NO_DATA ) { out << "NO_DATA"; }
  else if( type == MULTIPLE_INTEGERS ) { out << "MULTIPLE_INTEGERS"; }
  else if( type == MULTIPLE_DOUBLES )  { out << "MULTIPLE_DOUBLES"; }
  else if( type == MULTIPLE_VECTORS )  { out << "MULTIPLE_VECTORS"; }
  else {                       out << "Error: type_e '<<' operator.  Value of " << (int)type << " is invalid... \n"; }
  return out;
}

need_e
getNeed( const string & needStr )
{
  if(      needStr == "REQUIRED" ) {
    return REQUIRED;
  }
  else if( needStr == "OPTIONAL" ) {
    return OPTIONAL;
  }
  else if( needStr == "MULTIPLE" ) {
    return MULTIPLE;
  }
  else {
    cout << "Error: ProblemSpecReader.cc: need_e (" << needStr << ") did not parse correctly... "
         << "should be 'REQUIRED', 'OPTIONAL', or 'MULTIPLE'.\n";
    return INVALID_NEED;
  }
}

type_e
getType( const string & typeStr )
{
  if(      typeStr == "DOUBLE" ) {
    return DOUBLE;
  }
  else if( typeStr == "INTEGER" ) {
    return INTEGER;
  }
  else if( typeStr == "STRING" ) {
    return STRING;
  }
  else if( typeStr == "VECTOR" ) {
    return VECTOR;
  }
  else if( typeStr == "BOOLEAN" ) {
    return BOOLEAN;
  }
  else if( typeStr == "NO_DATA" ) {
    return NO_DATA;
  }
  else if( typeStr == "MULTIPLE_DOUBLES" ) {
    return MULTIPLE_DOUBLES;
  }
  else if( typeStr == "MULTIPLE_INTEGERS" ) {
    return MULTIPLE_INTEGERS;
  }
  else if( typeStr == "MULTIPLE_VECTORS" ) {
    return MULTIPLE_VECTORS;
  }
  else {
    throw ProblemSetupException( "Error: ProblemSpecReader.cc: type '" + typeStr + "' did not parse correctly...\n" +
                                 "should be 'DOUBLE', 'INTEGER', 'STRING', 'VECTOR', 'BOOLEAN', 'NO_DATA', 'MULTIPLE_DOUBLES',\n" +
                                 "'MULTIPLE_INTEGERS', or 'MULTIPLE_VECTORS'.\n",
                                 __FILE__, __LINE__ );
  }
}

/////////////////////////////////////////////////////////////////////////////////
// Helper Structs:

struct NeedAppliesTo {
  string parentAttributeName_;   // Eg: The parent of this tag will have an attribute named "type" or "label".
  vector< string > validValues_; //     the value of "type" might be 'hard_sphere_gas'.  If that value
};                               //     is in the validValues_ array, then the 'need' of the tag applies.

struct ChildRequirement : public RefCounted {
  enum Req { ONE_OF, ALL_OR_NONE_OF };

  Req              typeOfRequirement; 
  bool             appliesToAttribute; // 'false' if requirement applies to a child tag.
  vector< string > childrenList;       // used for ONE_OF and ALL_OR_NONE_OF
};

ostream &
operator<<( ostream & out, const ChildRequirement::Req & req )
{
  if(      req == ChildRequirement::ONE_OF )         { out << "ONE_OF"; }
  else if( req == ChildRequirement::ALL_OR_NONE_OF ) { out << "ALL_OR_NONE_OF"; }
  else {
    out << "Error in req '<<' operator: value of 'req': " << (int)req << ", is invalid... \n";
  }
  return out;
}

ostream &
operator<<( ostream & out, const ChildRequirement & chreq )
{
  if(      chreq.typeOfRequirement == ChildRequirement::ALL_OR_NONE_OF ){ out << "ALL_OR_NONE_OF( "; }
  else if( chreq.typeOfRequirement == ChildRequirement::ONE_OF )        { out << "ONE_OF( "; }
  else {
    ostringstream error;
    error << "Error in ChildRequirement::Req '<<' operator: value of 'chreq': " << (int)chreq.typeOfRequirement << ", is invalid... \n";
    throw ProblemSetupException( error.str(), __FILE__, __LINE__ );
  }
  for( unsigned int pos = 0; pos < chreq.childrenList.size(); pos++ ) {
    out << chreq.childrenList[ pos ] << " ";
  }
  if( chreq.childrenList.size() > 0 ) {
    out << ")";
  }
  return out;
}

/////////////////////////////////////////////////////////////////////////////////

struct AttributeAndTagBase :  public RefCounted {

  AttributeAndTagBase( const string & name, TagP parent ) :
    parent_( parent ), name_( name ), need_(INVALID_NEED), occurrences_( 0 ) { }

  AttributeAndTagBase( const string & name, need_e need, type_e type, 
                       const vector<string> & validValues, /*const*/ TagP parent ) :
    parent_( parent ),
    name_( name ), need_( need ), type_( type ),
    validValues_( validValues ),
    occurrences_( 0 ) { }

  AttributeAndTagBase( const string & name, need_e need, type_e type, 
                       const string & validValues, /*const*/ TagP parent ) :
    parent_( parent ),
    name_( name ), need_( need ), type_( type ),
    occurrences_( 0 )
  {
    vector<char> separators;
    separators.push_back( ',' );
    separators.push_back( ' ' );
    validValues_ = split_string( validValues, separators );
    for( unsigned int pos = 0; pos < validValues_.size(); pos++ ) {
      collapse( validValues_[ pos ] );
    }
  }

  virtual ~AttributeAndTagBase() {}

  TagP           parent_; // was const... should be, but I need to be able to pass it into findAttribute()...
  string         name_;
  need_e         need_;
  type_e         type_;
  vector<string> validValues_;
  int            occurrences_;

  NeedAppliesTo  needAppliesTo_;

  // currentValue_ is used only when this is an attribute.  Unlike the other tags,
  // its value can (and will) change during validation.  Its value is only 'valid'
  // when validating the tag's (for which this attribute applies) children.  Once
  // the validation moves past this tag (the tag for which this attribute applies)
  // then this field is no longer valid.
  string         currentValue_;

  // currentChildrenTags_ is used to validate the number of children (in some cases).
  // Its value changes as each tag is checked.  (It is valid while a given tag is 
  // being validated, but overwritten the next time that type of tag is validated.)
  vector< string > currentChildrenTags_;

  ///////////////////////////////////

  string getCompleteName() const;

  // 'node' is the gold standard node being validated against.
  void   validateText(    const string & text,  xmlNode * node  ) const;
  bool   validateString(  const string & value ) const;
  bool   validateBoolean( const string & value ) const;
  void   validateDouble(  double value         ) const;
  bool   validateVector(  const string & text ) const; // text should be "[#, #, #]" to be valid.

  virtual void cleanUp( bool force = false ) = 0;

  virtual void print( bool /* recursively = false */, unsigned int depth = 0, bool isTag = false ) { 

    // Fill is used to pad the Tag names so they line up better...
    if( depth > 14 ) {
      // Make sure the truncation is enough so that the below 30-depth*2 doesn't underflow...
      dbg << "WARNING... print truncating indention depth to 14...\n";
      depth = 14;
    }
    string fill;
    for( unsigned int pos = name_.size(); pos < (30-(depth*2)); pos++ ) {
      fill += " ";
    }

    indent( dbg, depth ); 
    dbg << (isTag ? "<" : "- " ) 
         << name_
         << (isTag ? ">" : "" ) 
         << fill << " - " << need_ << " - " << type_ << " - VVs: "
         << (validValues_.size() == 0 ? "" : "'" );
    for( unsigned int pos = 0; pos < validValues_.size(); pos++ ) {
      dbg << validValues_[pos] << " ";
    }
    dbg << (validValues_.size() == 0 ? "" : "'" )
        << "(occur'd: " << occurrences_ << ") "
        << "(rc: " << getReferenceCount() << ") "
        << "(" << this << ") ";
  }

};

struct Attribute : public AttributeAndTagBase { 

  Attribute( const string & name, need_e need, type_e type, const string & validValues, /*const*/ TagP parent ) :
    AttributeAndTagBase( name, need, type, validValues, parent ) {
  }

  virtual void print( bool recursively, unsigned int depth, bool isTag = false ) {
    AttributeAndTagBase::print( recursively, depth, isTag );
    dbg << "\n";
  }

  virtual void cleanUp( bool force = false ) {
    parent_ = 0;
  }

};

struct Tag : public AttributeAndTagBase {

private:
  const xmlNode              * originalXmlNode_; // This var is used (only) for debugging
                                                 // (it contains the file/line # this tag comes from...)
public:
  vector< AttributeP >         attributes_;
  vector< TagP >               subTags_;

  vector< ChildRequirementP >  childReqs_;

  bool                         forwardDeclaration_;
  bool                         isCommonTag_;

  // validValues is a _single_ string (it will be parsed as follows) that contains valid values
  // for the value of the tag.  The specification of valid values depends on the type of Tag:
  //
  //  STRING: a comma separated lists of strings, or "*" (or NULL) which means anything
  //  INTEGER/DOUBLE: "*" = any value, "positive" = a positive value, "num, num" = min, max values
  //  BOOLEAN: validValues is not allowed... because it defaults to true/false.
  //  VECTOR: FIXME... does nothing yet...
  //
  Tag( const string & name, TagP parent, const xmlNode * node ) : 
    // This constructor is used only for creating a tag that is a forward declaration place holder tag.
    AttributeAndTagBase( name, parent ), 
    originalXmlNode_( node ), forwardDeclaration_( true ), isCommonTag_( false ) {}

  Tag( const string & name, need_e need, type_e type, const string & validValues, /*const*/ TagP parent, const xmlNode * node ) :
    AttributeAndTagBase( name, need, type, validValues, parent ),
    originalXmlNode_( node ), forwardDeclaration_( false ), isCommonTag_( false ) {}

  Tag( const TagP commonTag, /*const*/ TagP parent, need_e need, const xmlNode * node ) :
    AttributeAndTagBase( commonTag->name_, commonTag->need_, commonTag->type_, commonTag->validValues_, parent ),
    originalXmlNode_( node ), forwardDeclaration_( false ) {

    if( need == INVALID_NEED ) { 
      need_ = commonTag->need_;
    }
    else if( need != need_ ) {
      dbg << "Notice: need changed to " << need << "\n";
      need_ = need;
    }
    attributes_  = commonTag->attributes_;
    subTags_     = commonTag->subTags_;
    childReqs_   = commonTag->childReqs_;
    isCommonTag_ = true;
  }

  // CleanUp() is used to 'unlink' all the children from their parents so that the ReferenceCount will reach 0
  // and the items will be deleted.  'force' is only used by the very top level as it doesn't have a parent_.

  virtual void cleanUp( bool force = false ) {

    for( vector< AttributeP >::iterator iter = attributes_.begin(); iter != attributes_.end(); iter++ ) {
      (*iter)->cleanUp();
    }

    for( vector< TagP >::iterator iter = subTags_.begin(); iter != subTags_.end(); iter++ ) {
      if( (*iter)->parent_ != this ) {
        (*iter) = 0;
      }
      else {
        (*iter)->cleanUp();
      }
    }
    parent_ = 0;
  }

  ~Tag() {
    for( vector< AttributeP >::iterator iter = attributes_.begin(); iter != attributes_.end(); iter++ ) {
      *iter = 0;
    }
    for( vector< TagP >::iterator iter = subTags_.begin(); iter != subTags_.end(); iter++ ) {
      if( *iter ) {
        *iter = 0;
      }
    }
    for( vector< ChildRequirementP >::iterator iter = childReqs_.begin(); iter != childReqs_.end(); iter++ ) {
      *iter = 0;
    }
  }

  AttributeP findAttribute( const string & attrName );
  TagP       findChildTag( const string & tagName );
  void       validateAttribute( xmlAttr * attr );

  // User most likely should not use the 'depth' parameter.
  // 'ps' is the ProblemSpec to be validated (the representation of the loaded .ups file).
  void        validate( const ProblemSpec * ps, unsigned int depth = 0 );
  void        parseXmlTag( const xmlNode * xmlTag );

  // Updates an incomplete tag that was referencing a forwardly declared tag.  Used when
  // the forwardly declared tag is actually finalized and information exists to do the update.
  void        update( TagP tag );

  virtual void print( bool recursively = false, unsigned int depth = 0, bool isTag = true ) {

    AttributeAndTagBase::print( recursively, depth, isTag );

    dbg << "(parent: " << (parent_ ? parent_->name_ : "NULL") << " - " << parent_.get_rep() << ") " 
        << "(common: " << isCommonTag_ << ")\n";

    if( isCommonTag_ ) { return; }

    if( childReqs_.size() > 0 ) {
      for( unsigned int pos = 0; pos < childReqs_.size(); pos++ ) {
        indent( dbg, depth + 1 ); 
        dbg << ": " << *(childReqs_[ pos ].get_rep()) << "\n";
      }
    }

    for( unsigned int pos = 0; pos < attributes_.size(); pos++ ) {
      attributes_[ pos ]->print( recursively, depth+1 );
    }

    if( recursively ) {
      for( unsigned int pos = 0; pos < subTags_.size(); pos++ ) {
        subTags_[pos]->print( recursively, depth+1 );
      }
    }
  }
}; // struct Tag

///////////////////////////////////////////////////////////////////////
// Currently all ProblemSpecReader's share the validation data...
// (Pragmatically I use this to not parse the DW created files,
//  and only parse the original .ups...)
static TagP uintahSpec_g;
static TagP commonTags_g;

// This map is used to allow validation of Geom tags (in the .ups files) that
// are 'name'd (or 'label'd) so they can be referenced again.  This is used
// only for 'cylinder's and 'box's currently.
map<string, TagP> namedGeomPieces_g;

list< TagP >      needForwardDeclResolution;
map<string, bool> forwardDeclMap;

//
///////////////////////////////////////////////////////////////////////

string
AttributeAndTagBase::getCompleteName() const
{
  string      result = name_;
  const Tag * tag = parent_.get_rep();
    
  while( tag != NULL ) {
    result = tag->name_ + "->" + result;
    tag = tag->parent_.get_rep();
  }
  return result;
}

AttributeP
Tag::findAttribute( const string & attrName )
{
  for( unsigned int pos = 0; pos < attributes_.size(); pos++ ) {
    if( attributes_[ pos ]->name_ == attrName ) {
      return attributes_[ pos ];
    }
  }
  return NULL;
}

TagP
Tag::findChildTag( const string & tagName )
{
  for( unsigned int pos = 0; pos < subTags_.size(); pos++ ) {
    if( subTags_[ pos ]->name_ == tagName ) {
      return subTags_[ pos ];
    }
  }
  return NULL;
}

// Chops up 'validValues' (based on ','s) and verifies that 'value' is in the list.
// (If validValues is empty, then 'value' is considered valid by definition.)
bool
AttributeAndTagBase::validateString( const string & value ) const
{
  // If no 'valid values' are set, then all values are valid.
  if( validValues_.size() == 0 ) {
    return true;
  }

  vector<string>::const_iterator iter = find( validValues_.begin(), validValues_.end(), value );
  if( iter != validValues_.end() ) {
    return true;
  } 
  else {
    return false;
  }
}

bool
AttributeAndTagBase::validateBoolean( const string & value ) const
{
  if( value == "true" || value == "false" ) {
    return true;
  }
  return false;
}

// validValues may be:  "positive" | "*" | "num, num" which means min, max (see .h file)
// for more info on 'validValues'. An empty validValues means anything is valid.
// 
void
AttributeAndTagBase::validateDouble( double value ) const
{
  if( validValues_.size() == 0 ) {
    return;
  }

  if( validValues_.size() == 1 ) {
    if( validValues_[0] == "positive" ) {
      if( value < 0 ) {
        ostringstream error;
        error << setprecision(12);
        error << "<" << getCompleteName() << ">: Specified value '" << value << "' is not 'positive' (as required).";
        throw ProblemSetupException( error.str(), __FILE__, __LINE__ );
      }
    }
  }
  else if( validValues_.size() == 2 ) {
    double max, min;
    sscanf( validValues_[0].c_str(), "%lf", &min );
    sscanf( validValues_[1].c_str(), "%lf", &max );
    if( value < min || value > max ) {
      ostringstream error;
      error << setprecision(12);
      error << "<" << getCompleteName() << "> - " << "Specified value '" << value << "' is outside of valid range (" 
            << min << ", " << max << ")";
      throw ProblemSetupException( error.str(), __FILE__, __LINE__ );
    }
  }
  else {
    throw ProblemSetupException( getCompleteName() + " - Invalid 'validValues' string.", __FILE__, __LINE__ );
  }
}

bool
AttributeAndTagBase::validateVector( const string & text ) const
{
  // remove " " from text
  string cleanText = text;
  replace_substring(cleanText, " ","");
  
  int numCommas = count_substrs( cleanText, "," ); 
  if( numCommas != 2 ) {
    return false;
  }

  double val1, val2, val3;
  int    num = sscanf( cleanText.c_str(), "[%lf,%lf,%lf]", &val1, &val2, &val3 );
  if( num != 3 ) {
    return false;
  }
  return true;
}

// Returns false if 'specStr' does not include the 'need' and 'type'
// (etc).  In this case, the tag is a common tag and needs to be found
// in the list of common tags.
bool
getNeedAndTypeAndValidValues( const string & specStr, need_e & need, type_e & type, string & validValues )
{
  // First bust up the specStr string based on the substring
  // (specified with ' (a quote).  This should give us 1 or 2 pieces.
  // (1 piece if there is not a 'validValues' string, and 2 pieces if
  // there is.)
  //
  vector<char> separators;
  separators.push_back( '\'' );

  vector<string> specs = split_string( specStr, separators );

  if( specs.size() < 1 || specs.size() > 2 ) {
    throw ProblemSetupException( "Error in getNeedAndTypeAndValidValues()...", __FILE__, __LINE__ );
  }

  separators.clear();
  separators.push_back( ' ' );
  separators.push_back( '\t' );
  vector<string> needType = split_string( specs[0], separators );

  if( needType.size() == 1 ) {
    // Only the 'need' is provided... grab it, and drop out.
    need = getNeed( needType[ 0 ] );
    return false; // Must be a common tag...
  }

  if( needType.size() != 2 ) {
    throw ProblemSetupException( string( "Error: need/type specification '" ) + concatStrings( needType ) + 
                                 "' did not parse correctly...", __FILE__, __LINE__ );
  }

  need = getNeed( needType[ 0 ] );
  type = getType( needType[ 1 ] );

  if( specs.size() == 2 ) {
    validValues = specs[1];
    if( type == NO_DATA ) {
      throw ProblemSetupException( "Error: type of Tag specified as 'NO_DATA', yet has a list of validValues: '" +
                                   validValues + "'", __FILE__, __LINE__ );
    }
    else if( type == BOOLEAN ) {
      throw ProblemSetupException( "Error: type of Tag specified as 'BOOLEAN', yet has list of validValues: '" +
                                   validValues + "'", __FILE__, __LINE__ );
    }
  }
  return true; // Not a common tag...
}

void
getLabelAndNeedAndTypeAndValidValues( const string & specStr, string & label, 
                                      need_e & need, type_e & type, string & validValues )
{
  // First bust up the specStr string based on the substring
  // (specified with ' (a quote).  This should give us 1 or 2 pieces.
  // (1 piece if there is not a 'validValues' string, and 2 pieces if
  // there is.)
  //
  vector<char> separators;
  separators.push_back( '\'' ); // Split by "'"s (single quotes).

  vector<string> specs = split_string( specStr, separators );

  if( specs.size() < 1 || specs.size() > 2 ) {
    ostringstream errorMsg;
    errorMsg << "Error in getLabelAndNeedAndTypeAndValidValues... Spec string split into " << specs.size() << " pieces,\n"
             << "(using ' (single quote) as the delimiter) but should have been only 1 or 2.  Spec string: '" << specStr << "'";
    throw ProblemSetupException( errorMsg.str(), __FILE__, __LINE__ );
  }

  separators.clear();
  separators.push_back( ' ' );
  separators.push_back( '\t' );
  vector<string> labelNeedType = split_string( specs[0], separators );

  if( labelNeedType.size() != 3 ) {
    throw ProblemSetupException( "Error: label/need/type did not parse correctly...", __FILE__, __LINE__ );
  }

  label = labelNeedType[ 0 ];
  need  = getNeed( labelNeedType[ 1 ] );
  type  = getType( labelNeedType[ 2 ] );

  if( specs.size() == 2 ) {
    if( type == NO_DATA ) {
      throw ProblemSetupException( "Error: type of Tag specified as NO_DATA, yet has a validValues '" +
                                   concatStrings( specs ) +"' component...", __FILE__, __LINE__ );
    }
    validValues = specs[1];
  }
}

} // end namespace Uintah

void
Tag::update( TagP tag )
{
  forwardDeclaration_ = false;

  need_          = tag->need_;
  type_          = tag->type_; 

  validValues_   = tag->validValues_;

  needAppliesTo_ = tag->needAppliesTo_;

  subTags_       = tag->subTags_;
  attributes_    = tag->attributes_;
  childReqs_     = tag->childReqs_;

  // Re-parent the sub tags and attributes...
  for( unsigned int pos = 0; pos < subTags_.size(); pos++ ) {
    subTags_[ pos ]->parent_ = this;
  }
  for( unsigned int pos = 0; pos < attributes_.size(); pos++ ) {
    attributes_[ pos ]->parent_ = this;
  }
}

void
Tag::parseXmlTag( const xmlNode * xmlTag )
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpecReader::parseXmlTag");

  //  string name = to_char_ptr( xmlTag->name );
  string name = (const char *)( xmlTag->name );
  collapse( name );

  dbg << "Parse node: " << name << "\n";

  bool hasSpecString = true;

  if( xmlTag == NULL ) {
    throw ProblemSetupException( "Error... passed in xmlTag is null...", __FILE__, __LINE__ );
  }
  else {
    if( name != "CommonTags" ) {

      TagP tempTag = commonTags_g->findChildTag( name ); // If the tag is (accidentally) declared twice, then
                                                         // this var will hold the first declaration...

      if( ( xmlTag->properties != NULL ) && tempTag ) {

        // This section checks for multiply declared versions of the same 'Common' tag...

        string specStr = (const char *)( xmlTag->properties->children->content );

        if( tempTag->forwardDeclaration_ ) { // If the original (first) specification of this Tag is a forward
                                             // declaration, then if the new one is a forward declaration, there
                                             // is an error, otherwise we are good to continue...
          if( specStr == "FORWARD_DECLARATION" ) {
            throw ProblemSetupException( "Error: The CommonTag <" + name + "> is forwardly declared twice.\n" +
                                         "First time: " + getErrorInfo(tempTag->originalXmlNode_) + "\n"
                                         "Second time: " + getErrorInfo(xmlTag), __FILE__, __LINE__ );
          }
        }
        else {

          need_e nd   = INVALID_NEED; // Need          - These three variable names are just used in debugging
          type_e tp   = INVALID_TYPE; // Type          - and just in the following line, so am using
          string vvs;                 // Valid Values  - abbreviated names as not to confuse with other vars later.
          getNeedAndTypeAndValidValues( specStr, nd, tp, vvs );
          if( tp != INVALID_TYPE ) {
            throw ProblemSetupException( "Error: The CommonTag <" + name + "> appears to be created (specified) twice.\n" +
                                         "First time: " + getErrorInfo(tempTag->originalXmlNode_) + "\n"
                                         "Second time: " + getErrorInfo(xmlTag), __FILE__, __LINE__ );
          }
        }
      }

      if( xmlTag->properties == NULL ) {

        TagP tag = commonTags_g->findChildTag( name );
        if( tag ) {
          dbg << "Found common tag for " << name << "\n";
          hasSpecString = false;
        }
        else {
          throw ProblemSetupException( "Error (a)... <" + name + "> does not have required 'spec' attribute (eg: spec=\"REQUIRED NO_DATA\").\n" +
                                       "              or couldn't find in CommonTags.\n" + getErrorInfo(xmlTag), __FILE__, __LINE__ );
        }
      }
      else if( xmlTag->properties->children == NULL ) {
        throw ProblemSetupException( "Error (b)... <" + name + "> does not have required 'spec' attribute (eg: spec=\"REQUIRED NO_DATA\").",
                                     __FILE__, __LINE__ );
      }
      else if( string( "spec" ) != (const char *)( xmlTag->properties->name ) ) {
        throw ProblemSetupException( "Error (c)... <" + name + "> does not have required 'spec' attribute (eg: spec=\"REQUIRED NO_DATA\").  " +
                                     "Found attribute '" + (const char *)( xmlTag->properties->name ) + "' instead.", __FILE__, __LINE__ );
      }
    }
  }

  need_e need        = INVALID_NEED;
  type_e type        = INVALID_TYPE;
  bool   common      = true;
  bool   forwardDecl = false;
  string validValues;

  if( hasSpecString ) {
    string specStr = (const char *)( xmlTag->properties->children->content );

    if( specStr == "FORWARD_DECLARATION" ) {
      forwardDecl = true;
    }
    else {
      common = !getNeedAndTypeAndValidValues( specStr, need, type, validValues );

      map<string,bool>::iterator iter = forwardDeclMap.find( name );

      if( iter != forwardDeclMap.end() ) {
        // The current tag being worked on is referencing a forwardly
        // declared tag.  All forwardly declared tags must be
        // 'common', so mark it as such so that further below we will
        // look for it in the list of common tags.
        common = true;
      }
      if( need == INVALID_NEED ) {
        throw ProblemSetupException( "The value of 'need' was invalid for: " + name, __FILE__, __LINE__ );
      }
    }
  }

  TagP newTag;
  bool updateForwardDecls = false;
  TagP commonTag = NULL;

  if( forwardDecl ) {
    // Add tag to the list of forwardly declared tags that need to be resolved when the real definition is found.
    forwardDeclMap[ name ] = true;
    newTag = new Tag( name, this, xmlTag );
  }
  else if( common ) {
    // Find this tag in the list of common tags... 
    commonTag = commonTags_g->findChildTag( name );
    if( !commonTag ) {
      throw ProblemSetupException( "Error, commonTag <" + name + "> not found... was looking for a common tag " +
                                   "because spec string only had one entry.", __FILE__, __LINE__ );
    }

    bool storeTag = false;

    if( commonTag->forwardDeclaration_ ) {
      if( type != INVALID_TYPE ) {
        // This is the real definition of a previously only forwardly declared tag.
        updateForwardDecls = true;
        commonTag->type_ = type;
        commonTag->need_ = need;
      }
      else {
        // add the new tag to a list of tags to be updated when we get the info we need
        storeTag = true;
      }
    }
    newTag = new Tag( commonTag, this, need, xmlTag );

    if( storeTag ) {
      // Save newTag in the list of tags that must be fixed up later (when the forwardly declared tag has been defined).
      needForwardDeclResolution.push_back( newTag );
    }
  }
  else {
    newTag = new Tag( name, need, type, validValues, this, xmlTag );
  }

  if( !updateForwardDecls ) {  // If we are updating a forward declaration, then our parent already knows about us. 
    subTags_.push_back( newTag );
  }

  // Handle attributes... (if applicable)
  if( hasSpecString && xmlTag->properties->next != NULL ) {
    for( xmlAttr * child = xmlTag->properties->next; child != 0; child = child->next) {
      if( child->type == XML_ATTRIBUTE_NODE ) {

        need_e need;
        type_e type;
        string label, validValues;

        const string attrName = (const char *)( child->name );

        if( attrName.find( "attribute") == 0 ) { // attribute string begins with "attribute"
          string specStr = (const char *)( child->children->content );
          getLabelAndNeedAndTypeAndValidValues( specStr, label, need, type, validValues );

          if( need == INVALID_NEED ) {
            throw ProblemSetupException("The value of 'need' was invalid for: " + name, __FILE__, __LINE__ );
          }
          newTag->attributes_.push_back( new Attribute( label, need, type, validValues, newTag ) );
        }
        else if( attrName.find( "children") == 0 ) {  // attribute string begins with "children"

          string attrStr = (const char *)( child->children->content );
          vector<string> strings;
          vector<char> separators;

          separators.push_back( ',' );
          separators.push_back( '(' );
          separators.push_back( ')' );
          separators.push_back( ' ' );

          strings = split_string( attrStr, separators );

          ChildRequirement * chreq = new ChildRequirement();

          if( strings[0] == "ONE_OF" ) {
            chreq->typeOfRequirement = ChildRequirement::ONE_OF;
          }
          else if( strings[0] == "ALL_OR_NONE_OF" ) {
            chreq->typeOfRequirement = ChildRequirement::ALL_OR_NONE_OF;
          }
          else {
            throw ProblemSetupException( string( "ERROR in parsing '" ) + attrStr + "'.  'children' tag does not support: " + 
                                         strings[0] + "...", __FILE__, __LINE__ );
          }

          if( strings.size() < 3 ) {
            stringstream error;
            error << "ERROR in parsing " << chreq->typeOfRequirement << "().  Not enough fields."
                  <<  getErrorInfo( xmlTag );
            throw ProblemSetupException( error.str(), __FILE__, __LINE__ );
          }

          if( strings[1] == "ATTRIBUTE" ) {
            chreq->appliesToAttribute = true;
          }
          else if( strings[1] == "CHILD" ) {
            chreq->appliesToAttribute = false;
          }
          else {
            stringstream error;
            error << "ERROR in parsing " << chreq->typeOfRequirement << "().  First field must be either ATTRIBUTE or CHILD, but found: '"
                  << strings[1] << "', " 
                  << getErrorInfo( xmlTag );
            throw ProblemSetupException( error.str(), __FILE__, __LINE__ );
          }

          for( unsigned int pos = 2; pos < strings.size(); pos++ ) {
            chreq->childrenList.push_back( strings[ pos ] );
          }
          newTag->childReqs_.push_back( chreq );
        }
        else if( attrName.find( "need_applies_to") == 0 ) {  // attribute string begins with "children"

          string attrStr = (const char *)( child->children->content );
          vector<string> strings;
          vector<char> separators;

          separators.push_back( ',' );
          separators.push_back( ' ' );

          strings = split_string( attrStr, separators );

          if( strings.size() <= 1 ) {
            throw ProblemSetupException( string( "ERROR in parsing '" ) + attrStr + "'.  Not enough values for 'need_applies_to' field.\n" +
                                         "Please fix 'ups_spec.xml'.\n", __FILE__, __LINE__ );
          }
          if( !findAttribute( strings[0] ) ) {
            throw ProblemSetupException( string( "ERROR in parsing '" ) + attrStr + "'.  Parent does not have attribute '" + strings[0] + "'",
                                         __FILE__, __LINE__ );
          }

          newTag->needAppliesTo_.parentAttributeName_ = strings[0];

          for( unsigned int pos = 1; pos < strings.size(); pos++ ) {

            string value = strings[ pos ];

            AttributeP attribute = findAttribute( newTag->needAppliesTo_.parentAttributeName_ );
            if( !attribute ) {
              throw ProblemSetupException( string( "Parent attribute '" ) + newTag->needAppliesTo_.parentAttributeName_ + "' specified for '"  +
                                           getCompleteName() + "' does not exist!\n" + "The 'need_applies_to' field " +
                                           "in the 'ups_spec.xml' is broken.  Please fix.",
                                           __FILE__, __LINE__ );
            }
            if( attribute->need_ != REQUIRED ) {
              ostringstream errorMsg;
              errorMsg << "For 'need_applies_to' tag '" << newTag->getCompleteName() << "', parent's attribute '" 
                       << newTag->needAppliesTo_.parentAttributeName_ << "'\nspecified for '"  
                       << getCompleteName() << "' must be REQUIRED, but it is marked as '" 
                       << attribute->need_ << "'.\nPlease fix 'ups_spec.xml'." ;
              throw ProblemSetupException( errorMsg.str(), __FILE__, __LINE__ );
            }

            if( !attribute->validateString( value ) ) {
              throw ProblemSetupException( value + " is not a valid value for attribute " + attribute->getCompleteName(),
                                           __FILE__, __LINE__ );
            }

            newTag->needAppliesTo_.validValues_.push_back( value );
          }

          // DEBUG print:
          dbg << "here: " << newTag->needAppliesTo_.validValues_.size() << "\n";

          dbg << "need_applies_to: " << newTag->needAppliesTo_.parentAttributeName_ << " for " 
               << concatStrings( newTag->needAppliesTo_.validValues_ ) << "\n";
        }
        else {
          throw ProblemSetupException( "Invalid attribute (" + attrName + ").", __FILE__, __LINE__ );
        }
      }
    }
  }

  // Handle any children of the node...
  for( xmlNode * child = xmlTag->children; child != 0; child = child->next) {
    if( child->type == XML_ELEMENT_NODE ) {

      string node = (const char *)( child->name );
      collapse( node );

      newTag->parseXmlTag( child );
    }
  }

  /////////////////////////////////////////////////////////////////////
  // Validate that any child requirements that have been specified
  // point at valid child tags/attributes.

  for( unsigned int pos = 0; pos < newTag->childReqs_.size(); pos++ ) {

    ChildRequirementP chreq = newTag->childReqs_[ pos ];
    if( chreq->typeOfRequirement == ChildRequirement::ONE_OF ||
        chreq->typeOfRequirement == ChildRequirement::ALL_OR_NONE_OF ) {

      for( unsigned int pos = 0; pos < chreq->childrenList.size(); pos++ ) {
        string childName = chreq->childrenList[ pos ];
        string theType;
        bool   valid = true;
        if( chreq->appliesToAttribute ) {
          AttributeP attributeTag = newTag->findAttribute( childName );
          if( !attributeTag ) {
            valid = false;
            theType = "ATTRIBUTE";
          }
        }
        else {
          TagP childTag = newTag->findChildTag( childName );
          if( !childTag ) {
            valid = false;
            theType = "CHILD";
          }
        }
        if( !valid ) {
          stringstream error;
          error << newTag->getCompleteName() << ": " << chreq->typeOfRequirement << " has nonexistent " 
                << theType << ": '" << childName << "'\n" 
                << getErrorInfo( xmlTag );
          throw ProblemSetupException( error.str(), __FILE__, __LINE__ );          
        }
      }
    }
    else {
      throw ProblemSetupException( "Unknown ChildRequirement type...", __FILE__, __LINE__ );
    }
  }

  if( updateForwardDecls ) {
    // This is the real definition of the tag, update the place holder (common) tag with the real data...
    commonTag->update( newTag );

    newTag = 0; // Give back the memory (or at least decrease the ref count).

    // Remove the tag's name from the map of tag names that records
    // forwardly declared tags that need to be resolved.  (Warning,
    // because the map only uses the 'local' name of a tag, if in the
    // future someone tried to get fancy with the forward declaration
    // infrastruction, they probably could easily break the code.)

    map<string,bool>::iterator fdmIter = forwardDeclMap.find( name );
    forwardDeclMap.erase( fdmIter );

    // Now that we have the full definition of the tag (that was
    // previously only forwardly declared), we can update any tags that 
    // referred to it...

    list<TagP>::iterator iter = needForwardDeclResolution.begin();
    while( iter != needForwardDeclResolution.end() ) {
      TagP fdtag = *iter;
      if( fdtag->name_ == name ) {
        // Update 'fdtag' which previously had just the (incomplete) forward dcl pointer...
        
        fdtag->attributes_  = commonTag->attributes_;

        fdtag->subTags_     = commonTag->subTags_;

        // Don't update 'need_' as it was already set with a potentially different value.
        fdtag->type_        = commonTag->type_;
        fdtag->validValues_ = commonTag->validValues_;

        fdtag->childReqs_   = commonTag->childReqs_;

        list<TagP>::iterator temp = iter;
        iter++;
        needForwardDeclResolution.erase( temp );
      }
      else {
        iter++;
      }
    }
  } // end if( updateForwardDecls )

} // end parseXmlTag()

void
ProblemSpecReader::parseValidationFile()
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpecReader::parseValidationFile");

  dbg << "\n";
  dbg << "-------------------------------------------------------------------\n";
  dbg << "- Parsing ups_spec.xml\n";
  dbg << "- \n";

  xmlDocPtr doc; /* the resulting document tree */
  
  string * valFile = scinew string( string( sci_getenv("SCIRUN_SRCDIR") ) + "/StandAlone/inputs/UPS_SPEC/ups_spec.xml" );

  d_upsFilename.push_back( valFile );

  doc = xmlReadFile( valFile->c_str(), 0, XML_PARSE_PEDANTIC );

  if (doc == 0) {
    throw ProblemSetupException( "Error opening .ups validation specification file: " + *valFile, __FILE__, __LINE__ );
  }

  xmlNode * root = xmlDocGetRootElement( doc );

  // FYI, We are hijacking the '_private' to be used as application data... specifically the name of the file
  // that this node came from.
  root->_private = (void*)valFile;
  resolveIncludes( root->children, root );

  uintahSpec_g = new Tag( "Uintah_specification", REQUIRED, NO_DATA, "", NULL, root );
  commonTags_g = new Tag( "CommonTags", REQUIRED, NO_DATA, "", NULL, root );

  //  string tagName = to_char_ptr( root->name );
  string tagName = (const char *)( root->name );

  if( tagName != "Uintah_specification" ) {
    throw ProblemSetupException( *valFile + " does not appear to be valid... First tag should be\n" +
                                 + "<Uintah_specification>, but found: " + tagName,
                                 __FILE__, __LINE__ );
  }

  // Find <CommonTags> (if it exists)
  bool commonTagsFound = false;
  for( xmlNode * child = root->children; child != 0; child = child->next) {

    if( child->type == XML_ELEMENT_NODE ) {
      string tagName = (const char *)( child->name );
    
      if( tagName == "CommonTags" ) {
        commonTagsFound = true;
        // Find first (real) child of the <CommonTags> block...
        xmlNode * gc = child->children; // Grand Child
        while( gc != NULL ) {
          if( gc->type == XML_ELEMENT_NODE ) {
            commonTags_g->parseXmlTag( gc );
          }
          gc = gc->next;
        }
      }
    }
  }
  if( !commonTagsFound ) {
    dbg << "\n\nWARNING: <CommonTags> block was not found...\n\n";
  }

  // Parse <Uintah_specification>
  for( xmlNode * child = root->children; child != 0; child = child->next) {
    if( child->type == XML_ELEMENT_NODE ) {
      string tagName = (const char *)( child->name );
      if( tagName != "CommonTags" ) { // We've already handled the Common Tags...
        uintahSpec_g->parseXmlTag( child );
      }
    }
  }
  // Freeing the xml doc here is causing crashes later.  I'm not sure why this is.  
  // Perhaps freeing this doc also frees the child elements which are accessed later
  //
  // qwerty, I think this is the reason for the memory increase... we DO need to call
  // xmlFreeDoc.  I believe that the reorganization of these routines fixes the
  // strange error that was happening before that caused Justin to comment this
  // out in the first place.
  xmlFreeDoc(doc);

  if( forwardDeclMap.size() > 0 ) {
    string name = forwardDeclMap.begin()->first;

    throw ProblemSetupException( string( "Forward declaration of '" ) + name +
                                 "' was never resolved... Please fix ups_spec.xml.", __FILE__, __LINE__ );
  }

  dbg << "Done parsing ups_spec.xml\n";

} // end parseValidationFile()

void
AttributeAndTagBase::validateText( const string & text, xmlNode * node ) const
{
  string classType = "Attribute";
  const string completeName = getCompleteName();

  const Tag * testIfTag = dynamic_cast<const Tag*>( this );
  if( testIfTag ) {
    classType = "Tag";
  }

  // Verify that 'the text' of the node exists or doesn't exist as required... 
  //    <myTag>the text</myTag>
  //
  if( type_ == NO_DATA ) {
    if( text != "" ) {
      throw ProblemSetupException( classType + " <" + completeName + "> should not have data (but has: '" + text +
                                   "').  Please fix XML in .ups file or correct validation Tag list.\n" +
                                   getErrorInfo( node ),
                                   __FILE__, __LINE__ );
    }
    return; // We're done... no text to validate.
  }
  else { // type != NO_DATA
    if( text == "" ) {
      stringstream error_stream;
      error_stream << classType << " <" << completeName << "> should have a value (of type: " 
                   << type_ << ") but is empty. " << "Please fix XML in .ups file or\n"
                   << "correct validation Tag list.\n"
                   << getErrorInfo( node );
      throw ProblemSetupException( error_stream.str(), __FILE__, __LINE__ );
    }
  }
  
  switch( type_ ) {
  case DOUBLE:
    {
      // WARNING: this sscanf isn't a sufficient test to validate that a double (and only
      //          a double exists in the text... 
      double value;
      int    num = sscanf( text.c_str(), "%lf", &value );
      
      if( num != 1 ) {
        throw ProblemSetupException( classType + " <" + completeName + "> should have a double value (but has: '" + text +
                                     "').  Please fix XML in .ups file or correct validation Tag list.\n" +
                                     getErrorInfo( node ),
                                     __FILE__, __LINE__ );
      } 
      else {
        validateDouble( value );
      }
    }
    break;
  case INTEGER:
    {
      int value;
      int num = sscanf( text.c_str(), "%d", &value ); // WARNING: this is probably not a sufficient check for an integer...
      if( num != 1 ) {
        throw ProblemSetupException( classType + " <" + completeName + "> should have an integer value (but has: '" + text +
                                     "').  Please fix XML in .ups file or correct validation Tag list.\n" +
                                     getErrorInfo( node ),
                                     __FILE__, __LINE__ );
      }
      else {
        validateDouble( (double)value );
      }
    }
    break;
  case STRING:
    if( !validateString( text ) ) {
      throw ProblemSetupException( "Invalid string value for " + classType + ": " + completeName + ". '" + 
                                   text + "' not found in this list:\n" + concatStrings( validValues_ ) + "\n" +
                                   getErrorInfo( node ),
                                   __FILE__, __LINE__ );
    }
    break;
  case BOOLEAN:
    if( !validateBoolean( text ) ) {
      throw ProblemSetupException( "Invalid boolean string value for " + classType + " <" + completeName +
                                   ">.  Value must be either 'true', or 'false', but '" + text + "' was found...\n" +
                                   getErrorInfo( node ),
                                   __FILE__, __LINE__ );
    }
    break;
  case VECTOR:
    {
      if( !validateVector( text ) ) {
        throw ProblemSetupException( classType + " ('" + completeName + "') should have a Vector value (but has: '" +
                                     text + "').  Please fix XML in .ups file or correct validation Tag list.\n" +
                                     getErrorInfo( node ),
                                     __FILE__, __LINE__ );
      }
    }
    break;
  case MULTIPLE_INTEGERS:
    {
      int loc = text.find( "." );
      if( loc != -1 ) {
        throw ProblemSetupException( classType + " ('" + completeName + "') should have a multiple integer values (but has: '" +
                                     text + "').  Please fix XML in .ups file or correct validation Tag list.\n" +
                                     getErrorInfo( node ),
                                     __FILE__, __LINE__ );
      }
      char tokens[ text.length() + 1];
      strcpy ( tokens, text.c_str() );

      char * token = strtok( tokens, "[,]" );

      while( token != NULL ) {
        int result;
        int num = sscanf( token, "%d", &result );

        if( num != 1 ) {
          throw ProblemSetupException( classType + " ('" + completeName + "') should have a multiple double values (but has: '" +
                                       text + "').  Please fix XML in .ups file or correct validation Tag list.\n" +
                                       getErrorInfo( node ),
                                       __FILE__, __LINE__ );
        } 
        token = strtok( NULL, "[,]" );
      }
    }
    break;
  case MULTIPLE_DOUBLES:
    {
      char tokens[ text.length() + 1];
      strcpy ( tokens, text.c_str() );

      char * token = strtok( tokens, "[,]" );

      while( token != NULL ) {
        double result;
        int    num = sscanf( token, "%lf", &result );

        if( num != 1 ) {
          throw ProblemSetupException( classType + " ('" + completeName + "') should have a multiple double values (but has: '" +
                                       text + "').  Please fix XML in .ups file or correct validation Tag list.\n" +
                                       getErrorInfo( node ),
                                       __FILE__, __LINE__ );
        } 
        token = strtok( NULL, "[,]" );
      }
    }
    break;
  case MULTIPLE_VECTORS:
    {
      string tempText = text;
      replace_substring( tempText, " ", "" );
      replace_substring( tempText, "\t", "" );
      collapse( tempText );

      // Vector of Vectors starts with [[ and ends with ]]... verify this...
      if( tempText.substr(0,2) != "[[" || tempText.substr( tempText.length() - 2, 2 ) != "]]" ) {
        throw ProblemSetupException( "This does not look like a Vector of Vectors.  Expected to find [[ and ]] but have this: '" + tempText + "'",
                                     __FILE__, __LINE__ );
      }

      unsigned int pos = 1; // start at first vectors '['

      while( pos < tempText.length()-2 ) {

        string vectorStr = tempText.substr( pos, tempText.find( "]", pos ) - pos + 1 );
        if( !validateVector( vectorStr ) ) {
          throw ProblemSetupException( classType + " ('" + completeName + "') should have a MULTIPLE_VECTOR value (but has: '" +
                                       text + "').  Please fix XML in .ups file or correct validation Tag list.\n" +
                                       getErrorInfo( node ),
                                       __FILE__, __LINE__ );
        }
        pos = tempText.find( "[", pos+1 );
      }
    }
    break;
  case NO_DATA:
    // Already handled above...
  case INVALID_TYPE:
    break;
  }

} // end validateText()

void
Tag::validateAttribute( xmlAttr * attr )
{
  if( attr == NULL ) {
    throw ProblemSetupException( "Error... attr is NULL", __FILE__, __LINE__ );
  }

  const string attrName = (const char *)( attr->name );

  AttributeP   attribute = findAttribute( attrName );

  if( !attribute ) {
    string errorInfo = getErrorInfo( attr->parent );
    throw ProblemSetupException( "Error, attribute ('" + attrName + "') not found for " + getCompleteName() + "\n" +
                                 errorInfo, __FILE__, __LINE__ );
  }

  attribute->occurrences_++;

  string attrContent = (const char *)( attr->children->content );
  collapse( attrContent );
  attribute->validateText( attrContent, (xmlNode*)attr );

  // dbg << "currentValue_ of " << name_ << " is now " << attrContent << "\n";

  attribute->currentValue_ = attrContent;

  if( name_ == "cylinder" || name_ == "box" ) {
    if( attributes_.size() == 1 && attributes_[0]->name_ == "label" ) {
      namedGeomPieces_g[attrContent] = this;
    }
  }
} // end validateAttribute()

void
Tag::validate( const ProblemSpec * ps, unsigned int depth /* = 0 */ )
{
  string name = ps->getNodeName();

  indent( inc_dbg, depth );
  inc_dbg << name << "\t\t\t" << getErrorInfo( ps->getNode() ) << "\n";

  MALLOC_TRACE_TAG_SCOPE("ProblemSpecReader::validate");
  if( !uintahSpec_g ) {
    throw ProblemSetupException( "Strange, UintahSpec_g does not exist...", __FILE__, __LINE__ );
  }

  if( dbg.active() ) {
    dbg << "ProblemSpec::validate - ";
    indent( dbg, depth ); dbg << name << "\n";
  }

  /////////////////////////////////////////////////////////////////////
  // Zero out the number of occurrences of all the sub tags and
  // attributes so occurrences in previous tags will not be counted
  // against the current tag.  (Clean up all state tracking variables.)
  //
  currentChildrenTags_.clear();

  for( unsigned int pos = 0; pos < subTags_.size(); pos++ ) {
    subTags_[ pos ]->occurrences_ = 0;
  }
  for( unsigned int pos = 0; pos < attributes_.size(); pos++ ) {
    attributes_[ pos ]->occurrences_ = 0;
    // Reset currentValue_
    attributes_[ pos ]->currentValue_ = "";
  }
  /////////////////////////////////////////////////////////////////////

  // Validate elements (we also call them attributes).  We need to do it
  // here (before the child tags) so that possible need_applies_to values
  // can be set.

  xmlAttr* attr = ps->getNode()->properties;

  if( attributes_.size() == 0 && attr ) {
    throw ProblemSetupException( "Tag " + getCompleteName() + " has an attribute ('" + (const char *)( attr->name ) + 
                                 "'), but spec says there are none...\n" + getErrorInfo( attr->parent ), __FILE__, __LINE__ );
  }

  for (; attr != 0; attr = attr->next) {
    if (attr->type == XML_ATTRIBUTE_NODE) {
      validateAttribute( attr );
    }
    // else skip comments, blank lines, etc...
  }

  /////////////////////////////////////////////////////////////////////

  // Run through all the nodes of the ProblemSpec (from the .ups file) to validate them:
  //
  //     FYI, child->children would only be null for a tag like this <myTag></myTag>
  //     If it was: <myTag>  </myTag> then children is not null... it just filled with blanks.

  int       numTextNodes = 0;
  string    text = "";
  xmlNode * textNode = ps->getNode(); // default to 'ps' in case a text node is not found...

  for( xmlNode *child = ps->getNode()->children; child != 0; child = child->next) {

    if (child->type == XML_TEXT_NODE) {

      string tempText = (const char *)( child->content );
      collapse( tempText );

      if( tempText != "" ) {
        if( numTextNodes >= 1 ) {
          throw ProblemSetupException( string( "Node has multiple text (non-tag) nodes in it, but should have only one!\n" ) +
                                       "       The 2nd text node contains: '" + tempText + "'\n" +
                                       getErrorInfo( child ), __FILE__, __LINE__ );
        }
        numTextNodes++;
        textNode = child;
        text = tempText;
      }
    }
    else if( child->type == XML_COMMENT_NODE ) {
      continue;
    }
    else if( child->type != XML_ELEMENT_NODE ) {
      throw ProblemSetupException( string( "Node has an unknown type of child node... child node's name is '" ) +
                                   (const char *)( child->name ) + "'", __FILE__, __LINE__ );
    }
    else {
      // Handle sub tag
      const char * childName = (const char *)( child->name );
      TagP         childTag  = findChildTag( childName );

      if( !childTag ) {
        throw ProblemSetupException( string( "Tag '" ) + childName + "' not valid (for <" + getCompleteName() + 
                                     ">).  Please fix XML in .ups file or correct validation Tag list.\n" +
                                     getErrorInfo( child ), __FILE__, __LINE__ );
      }

      currentChildrenTags_.push_back( childTag->name_ );

      childTag->occurrences_++;

      if( childTag->occurrences_ > 1 && childTag->need_ != MULTIPLE ) {
        throw ProblemSetupException( string( "Tag <" ) + childTag->getCompleteName() +
                                     "> occurred too many times.  Please fix XML in .ups file or correct validation Tag list.\n" +
                                     getErrorInfo( child ),
                                     __FILE__, __LINE__ );
      }

      // Verify that OPTIONAL parameters that have a 'need_applies_to' field are only used
      // with the appropriate parent tag types...
      dbg << "checking for optional tag, and if need_applies_to: " << childTag->name_ << ", '" << childTag->needAppliesTo_.parentAttributeName_ << "'\n";

      if( childTag->need_ == OPTIONAL && childTag->needAppliesTo_.parentAttributeName_ != "" ) {
        
        AttributeP attribute = childTag->parent_->findAttribute( childTag->needAppliesTo_.parentAttributeName_ );

        if( !attribute ) {
          throw ProblemSetupException( string( "Parent attribute '" + childTag->needAppliesTo_.parentAttributeName_ + "' specified for '"  +
                                               childTag->parent_->getCompleteName() + "' does not exist!\n" + "The 'need_applies_to' field " +
                                               "in the 'ups_spec.xml' is broken.  Please fix." ),
                                       __FILE__, __LINE__ );
        }
        
        if( attribute->currentValue_ == "" ) {
          // If this tag has a "needAppliesTo_", then the parent's attribute must have a current value.
          throw ProblemSetupException("this is an error\n", __FILE__, __LINE__); // FIXME fix error message...
        }

        dbg << "NEED_APPLIES_TO '" << childTag->parent_->getCompleteName() << " '" << childTag->needAppliesTo_.parentAttributeName_ 
            << "' attribute, when the attribute's value is: '" << concatStrings( childTag->needAppliesTo_.validValues_ ) << "'\n";
        dbg << "  We are currently looking at the " + childTag->getCompleteName() + " tag.\n";
        
        vector<string>::const_iterator iter = find( childTag->needAppliesTo_.validValues_.begin(), childTag->needAppliesTo_.validValues_.end(),
                                                    attribute->currentValue_ );
        if( iter == childTag->needAppliesTo_.validValues_.end() ) {
          throw ProblemSetupException( string( "The OPTIONAL tag '" ) + childTag->getCompleteName() + "' is not a valid child for\n'"  +
                                               childTag->parent_->getCompleteName() + " (" + attribute->name_ + ": " +
                                               attribute->currentValue_ + ")'.  See the 'need_applies_to' field " +
                                               "in the 'ups_spec.xml'.\n" +
                                               getErrorInfo( child ),
                                       __FILE__, __LINE__ );
        }
      }

      ProblemSpec gcPs( child );
      childTag->validate( &gcPs, depth+1 );
    }
    
  } // end for child in d_node->children
  
  validateText( text, textNode );

  // Verify that all REQUIRED attributes were found:
  for( unsigned int pos = 0; pos < attributes_.size(); pos++ ) {
    AttributeP attr = attributes_[ pos ];
    if( attr->need_ == REQUIRED && attr->occurrences_ == 0 ) {
      throw ProblemSetupException( string( "Required attribute '" ) + attr->getCompleteName() +
                                   "' missing.  Please fix XML in .ups file or correct validation Attribute list.\n" +
                                   getErrorInfo( ps->getNode() ),
                                   __FILE__, __LINE__ );
    }
  }

  /////////////////////////////////////////////////////////////////////
  // Validate the child requirements

  dbg << "validate child requirements for " << name << ": " << childReqs_.size() << "\n";

  for( unsigned int pos = 0; pos < childReqs_.size(); pos++ ) {
    string theType = (childReqs_[ pos ]->typeOfRequirement) ? "ATTRIBUTE" : "CHILD";
    switch( childReqs_[ pos ]->typeOfRequirement ) {
    case ChildRequirement::ONE_OF :  // Verify that the ONE_OF is valid
      {
        bool   foundIt = false;
        for( unsigned int childNamePos = 0; childNamePos < childReqs_[ pos ]->childrenList.size(); childNamePos++ ) {
          string & childName = childReqs_[ pos ]->childrenList[ childNamePos ];

          if( childReqs_[ pos ]->appliesToAttribute ) {
            string dummy;
            bool foundAttribute = ps->getAttribute( childName, dummy );
            if( foundAttribute ) {
              if( foundIt ) {
                throw ProblemSetupException( "Error with " + getCompleteName() + ": ONE_OF child.  More than one child found!\n" +
                                             "Should only be one of: " +  concatStrings( childReqs_[pos]->childrenList ) + ".\n" + 
                                             getErrorInfo( ps->getNode() ),
                                             __FILE__, __LINE__ );
              }
              foundIt = true;
            }
          }
          else {
            ProblemSpecP childPs = ps->findBlock( childName );
            if( childPs ) {
              if( foundIt ) {
                throw ProblemSetupException( "Error with " + getCompleteName() + ": ONE_OF child.  More than one child found!\n" +
                                             "Should only be one of: " +  concatStrings( childReqs_[pos]->childrenList ) + ".\n" + 
                                             getErrorInfo( ps->getNode() ),
                                             __FILE__, __LINE__ );
              }
              foundIt = true;
            }
          }
        } //end for
        if( !foundIt ) {
          throw ProblemSetupException( "Error with " + getCompleteName() + ":\nONE_OF( " + theType + " " + 
                                       concatStrings( childReqs_[pos]->childrenList ) + " )" + 
                                       "' does not have a required " + theType + " tag.\n" +
                                       getErrorInfo( textNode ),
                                       __FILE__, __LINE__ );
        }
      }
      break;
    case ChildRequirement::ALL_OR_NONE_OF :
      {
        vector<bool> found( childReqs_[ pos ]->childrenList.size() );
        for( unsigned int childNamePos = 0; childNamePos < childReqs_[ pos ]->childrenList.size(); childNamePos++ ) {
          string & childName = childReqs_[ pos ]->childrenList[ childNamePos ];
          if( childReqs_[ pos ]->appliesToAttribute ) {
            string dummy;
            found[ childNamePos ] = ps->getAttribute( childName, dummy );
          }
          else {
            ProblemSpecP childPs = ps->findBlock( childName );
            found[ childNamePos ] = (childPs.get_rep() != NULL);
          }
        }
        bool valuesMustAllBe = found[ 0 ];
        for( unsigned int childNamePos = 1; childNamePos < childReqs_[ pos ]->childrenList.size(); childNamePos++ ) {
          if( found[ childNamePos ] != valuesMustAllBe ) {
            throw ProblemSetupException( "Error with " + getCompleteName() + ": \n" +
                                         "Incomplete option specification.  " +
                                         "You must specify all of the following options: '" +
                                         concatStrings( childReqs_[pos]->childrenList ) + "' or none of them.\n" +
                                         getErrorInfo( textNode ),
                                         __FILE__, __LINE__ );
          }
        }
      }
      break;
    default:
      throw ProblemSetupException( "Error with " + getCompleteName() + ": Unknown type of child requirement..." +
                                   getErrorInfo( textNode ),
                                   __FILE__, __LINE__ );
    }
  }

  /////////////////////////////////////////////////////////////////////
  // Verify that all REQUIRED tags were found:
  for( unsigned int pos = 0; pos < subTags_.size(); pos++ ) {
    TagP tag = subTags_[ pos ];
    if( tag->need_ == REQUIRED && tag->occurrences_ == 0 ) {

      xmlAttr* attr = ps->getNode()->properties;
      if( attr != 0 ) {
        const char * attrContent = (const char *)( attr->children->content );
        if( namedGeomPieces_g[ attrContent ] ) {
          // If this is a named geometry piece, then it won't (shouldn't) have child tags.
          continue;
        }
      }

      if( tag->needAppliesTo_.parentAttributeName_ != "" ) {
        // Tag only applies to specific versions of the parent...
        AttributeP attribute = tag->parent_->findAttribute( tag->needAppliesTo_.parentAttributeName_ );

        dbg << "Need_applies_to '" << tag->parent_->getCompleteName() << " '" << tag->needAppliesTo_.parentAttributeName_ 
            << "' attribute, when the attribute's value is: '" << concatStrings( tag->needAppliesTo_.validValues_ ) << "'\n";
        dbg << "  We are currently looking at the " + tag->getCompleteName() + " tag.\n";

        if( !attribute ) {
          throw ProblemSetupException( string( "Parent attribute '" + tag->needAppliesTo_.parentAttributeName_ + "' specified for '"  +
                                               tag->parent_->getCompleteName() + "' does not exist!\n" + "The 'need_applies_to' field " +
                                               "in the 'ups_spec.xml' is broken.  Please fix." ),
                                       __FILE__, __LINE__ );
        }

        dbg << "  Note, the value of the attribute at this point is '" << attribute->currentValue_ << "'.\n";

        if( attribute->currentValue_ == "" ) {
          // If this tag has a "needAppliesTo_", then the parent's attribute must have a current value.
          throw ProblemSetupException("this is an error\n", __FILE__, __LINE__); // FIXME fix error message...
        }
        else {
          vector<string>::const_iterator iter = find( tag->needAppliesTo_.validValues_.begin(), tag->needAppliesTo_.validValues_.end(),
                                                      attribute->currentValue_ );

          // template tag is 'lattice_refinement_ratio'.  it is required for Hierarchical (regridder)
          // we are working on a currentValue_ of "BNR".  which is not found.

          if( iter == tag->needAppliesTo_.validValues_.end() ) {
            continue;  // Tag is required, but not for this 'version' of the parent tag, so we just skip it.
          }
        }
      }

      throw ProblemSetupException( string( "Required tag '" ) + tag->getCompleteName() +
                                   "' missing.  Please fix .ups file (or update ups_spec.xml).\n" +
                                   getErrorInfo( textNode ),
                                   __FILE__, __LINE__ );
    }
  }
} // end validate()

void
ProblemSpecReader::validateProblemSpec( ProblemSpecP & prob_spec )
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpecReader::validateProblemSpec");
  // qwerty: fixme: this comment is no longer correct... i think... Currently, the readInputFile() (and thus this validation) is called
  // multiple times (once for the initial .ups, but then again for 
  // saving data archives and such...)  this (temporary?) hack is used
  // to only validate the initial .ups file.
  //

  if( !uintahSpec_g ) {

    parseValidationFile();

    if( dbg.active() ) {
      dbg << "-----------------------------------------------------------------\n";
      dbg << "-- uintah spec: \n";
      dbg << "\n";
      uintahSpec_g->print( true );
      dbg << "---done printing --------------------------------------------------------------\n";
      dbg << "-- commont Tags: \n";
      commonTags_g->print( true );
      dbg << "---done printing --------------------------------------------------------------\n";
    }
  }

  try {
    dbg << "Now to validate the input file...\n";
    uintahSpec_g->validate( prob_spec.get_rep() );
  }
  catch( ProblemSetupException & pse ) {
    if( Parallel::getMPIRank() == 0 ) {
      cout << "\n";
      cout << "!! WARNING: Your .ups file did not parse successfully...\n";
      cout << "!!          Fix your .ups file or update the ups_spec.xml\n";
      cout << "!!          specification.  Reason for failure is:\n";
      cout << "\n";
      throw;
    }
  }

  namedGeomPieces_g.clear();

  commonTags_g->cleanUp( true );
  commonTags_g = 0; // Give back the memory.

  uintahSpec_g->cleanUp( true );
  uintahSpec_g = 0; // Give back the memory.
}


//////////////////////////////////////////////////////////////////////////////////////////////

ProblemSpecReader::ProblemSpecReader()
{
}

ProblemSpecReader::~ProblemSpecReader()
{
  d_xmlData = 0;

  for( unsigned int pos = 0; pos < d_upsFilename.size(); pos++ ) {
    delete d_upsFilename[ pos ];
  }
}

string
getPath( const string & filename )
{
  return filename.substr( 0, filename.rfind( "/" ) );
}

string
validateFilename( const string & filename, const xmlNode * parent )
{
  string fullFilename;
  string errorMsg;
  bool   filenameIsBad = false;

  if( filename[0] != '/') { // If not absolute path, make it one...
          
    if( parent ) {
      fullFilename = getPath( *((string*)(parent->_private)) ) + "/" + filename;
      inc_dbg << "1) filename: " << fullFilename << "\n";
    }

    if( !parent || !validFile( fullFilename ) ) { // Check to see if the file is relative to where the program was run... 

      char buffer[2000];
      char * str = getcwd( buffer, 2000 );
      if( str == NULL ) {
        proc0cout << "WARNING: Directory not returned by getcwd()...\n";
      }
      else {
        fullFilename = string(buffer) + "/" + filename;
      }
      if( !validFile( fullFilename ) ) {
        filenameIsBad = true;
        errorMsg = "Couldn't find include file: '" + fullFilename + 
                   "' or '" + fullFilename + "'\n";
      }
    }
  }
  else {
    // Verify that the absolute path is valid:
    fullFilename = filename;

    if( !validFile( fullFilename ) ) {
      filenameIsBad = true;
    }
  }

  if( filenameIsBad ) {
    if ( !getInfo( fullFilename ) ) { // Stat'ing the file failed... so let's try testing the filesystem...
      stringstream error_stream;
      
      string directory = fullFilename.substr(0, fullFilename.rfind( "/" ) );
          
      if( !testFilesystem( directory, error_stream, Parallel::getMPIRank() ) ) {
        cout << error_stream.str();
        cout.flush();
      }
    }
    string errorInfo;
    if( parent ) {
      errorInfo = getErrorInfo( parent );
    }
    throw ProblemSetupException( errorMsg + errorInfo, __FILE__, __LINE__ );
  }
  else {
    return fullFilename;
  }

} // end validateFilename()

void
printDoc( xmlNode * node, int depth )
{
  if( depth == 0 ) {
    cout << "PRINTING DOC\n";
  }

  if( node == NULL ) {
    return;
  }

  if( node->type == XML_ELEMENT_NODE ) {
    string name = (char*)node->name;

    indent( cout, depth );
    cout << name << "\n";
  }

  xmlNode * child = node->children;

  while( child != 0 ) {

    if( child->type == XML_ELEMENT_NODE ) {
      printDoc( child, depth+1 );
    }
    child = child->next;
  }
  if( depth == 0 ) {
    cout << "DONE PRINTING DOC\n";
  }

} // end printDoc()


ProblemSpecP
ProblemSpecReader::readInputFile( const string & filename, bool validate /* = false */ )
{
  MALLOC_TRACE_TAG_SCOPE( "ProblemSpecReader::readInputFile" );
  if( d_xmlData != 0 ) {
    return d_xmlData;
  }

  static bool initialized = false;
  if (!initialized) {
    LIBXML_TEST_VERSION;
    initialized = true;
  }

  string full_filename = validateFilename( filename, NULL );

  xmlDocPtr doc = xmlReadFile( full_filename.c_str(), 0, XML_PARSE_PEDANTIC );

  if( doc == NULL ) {
    throw ProblemSetupException("Error parsing '" + filename + "'.  Please fix the XML.", __FILE__, __LINE__);
  }
    
  // you must free doc when you are done.
  // Add the parser contents to the ProblemSpecP

  ProblemSpecP prob_spec = scinew ProblemSpec( xmlDocGetRootElement(doc), true );

  string * strPtr = new string( full_filename );

  d_upsFilename.push_back( strPtr );

  xmlNode * tempNode = prob_spec->getNode();

  tempNode->_private = (void*)strPtr;

  resolveIncludes( prob_spec->getNode()->children, prob_spec->getNode() );

  // Debugging prints:
  //   cout << "------------------------------------------------------------------\n";
  //   printDoc( prob_spec->getNode(), 0 );
  if( validate ) {
    validateProblemSpec( prob_spec );
  }

  d_xmlData = prob_spec;

  return prob_spec;

} // end readInputFile()

string *
ProblemSpecReader::findFileNamePtr( const string & filename )
{
  for( unsigned int pos = 0; pos < d_upsFilename.size(); pos++ ) {
    if( *d_upsFilename[ pos ] == filename ) {
      return d_upsFilename[ pos ];
    }
  }
  return NULL;
}

void
ProblemSpecReader::resolveIncludes( xmlNode * child, xmlNode * parent, int depth /* = 0 */ )
{
  MALLOC_TRACE_TAG_SCOPE("ProblemSpecReader::resolveIncludes");

  while( child != NULL ) {
    if( child->type == XML_ELEMENT_NODE ) {
      string name1 = (const char *)(child->name);
      if( name1 == "include" ) {
        ProblemSpec temp( child );
        map<string, string> attributes;
        temp.getAttributes( attributes );
        xmlNode * prevChild = child->prev;
        
        string filename;
        string section = attributes[ "section" ];
        if( child->_private ) {
          filename = validateFilename( attributes["href"], child );
        }
        else {
          filename = validateFilename( attributes["href"], parent );
        }
        
        xmlDocPtr  doc     = xmlReadFile( filename.c_str(), 0, XML_PARSE_PEDANTIC );
        xmlNode  * include = xmlDocGetRootElement(doc);
        string * strPtr = new string( filename );
        d_upsFilename.push_back( strPtr );

        // nodes to be substituted must be enclosed in a 
        // "Uintah_Include" node

        string name = (const char *)( include->name );
        if( name == "Uintah_Include" || name == "Uintah_specification" ) {
          xmlNode * incChild = include->children;

          if( section != "" ) {  // Find the right section of the included file (to include).

            vector<string> sections;
            vector<char>   separators;
            separators.push_back( '/' );
            sections = split_string( section, separators );
            for( unsigned int pos = 0; pos < sections.size(); pos++ ) {
              bool done = false;
              while( !done ) {
                if( incChild == NULL ) {
                  throw ProblemSetupException("Error parsing included file '" + filename + "' section '" + section + "'", __FILE__, __LINE__);
                }

                string childName = (const char *)(incChild->name);
                
                if( ( incChild->type == XML_ELEMENT_NODE ) && ( childName == sections[ pos ] ) ) {
                  if( pos != sections.size()-1 ) { // Not the last section, so descend.
                    incChild = incChild->children;
                  }
                  done = true;
                }
                else {
                  incChild = incChild->next;
                }
              }
            }
          }

          while( incChild != 0 ) {
            // Make include be created from same document that created params...
            //ProblemSpecP newnode = tempParentPS.importNode( incChild, true );

            xmlNode * newnode = xmlDocCopyNode( incChild, parent->doc, true );
            if( prevChild == NULL ) {
              prevChild = newnode;
            }

            // Record the newnode's real file info...
            newnode->_private = strPtr;
            xmlAddPrevSibling( child, newnode );
            incChild = incChild->next;
          }

          // Remove the <include>
          xmlUnlinkNode( child );
          xmlFreeNode(   child );

          // Once all the 'included' tags have been added to the parent, and the <include>
          // has been removed, we need to start parsing at the first included tag.
          child = prevChild;
        }
        else {
          throw ProblemSetupException("No href attributes in include tag", __FILE__, __LINE__);
        }
        xmlFreeDoc( doc );
      }
      else { // !"include"
        indent( dbg, depth );
        dbg << " * " << name1 << "\n";
        if( child->_private == NULL ) {
          child->_private = parent->_private;
        }
      }

      if( child != NULL ) {
        // Child can be NULL if an <include> is the last 'child'... as the <include> is
        // removed from the tree above.
        xmlNode * grandchild = child->children;

        if( grandchild != NULL ) {
          resolveIncludes( grandchild, child, depth+1 );
        }
      }
    } // end if( child->getNodeType() == XML_ELEMENT_NODE ) {

    child = child->next;

  } // end while( child != NULL )

} // end resolveIncludes()

