/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

#include <Packages/Uintah/CCA/Components/ProblemSpecification/ProblemSpecReader.h>

#include <Packages/Uintah/Core/Parallel/Parallel.h> // Only used for MPI cerr
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h> // process determination
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>

#include <Core/Containers/StringUtil.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/FileUtils.h>

#include <iostream>
#include <iomanip>
#include <sstream>

#include <stdio.h>

#include <libxml/tree.h>
#include <libxml/parser.h>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

//////////////////////////////////////////////////////////////////////////////////////////////

static DebugStream dbg( "ProblemSpecReader" );

//////////////////////////////////////////////////////////////////////////////////////////////
// Utility Functions:

// Prints out 2 spaces for each level of indentation.
static
void
indent( ostream & out, unsigned int level )
{
  for( unsigned int pos = 0; pos < level; pos++ ) {
    out << "  ";
  }
}

// Coverts a vector of strings into a single " " separated string...
static
string
toString( const vector<string> & vec )
{
  string result;
  for( unsigned int pos = 0; pos < vec.size(); pos++ ) {
    result += vec[ pos ] + "\n";
  }
  return result;
}

//////////////////////////////////////////////////////////////////////////////////////////////
// The following section holds structures used in validating the problem spec file.

/////////////////////////////////////////////////////////////////
//
// need_e, type_e, Element, and Tag are all used for validation of XML
//

// MULTIPLE = 0 or more occurrences
enum need_e { OPTIONAL, REQUIRED, MULTIPLE, INVALID_NEED };
// VECTORs are specified as [0.0,0.0,0.0]
enum type_e { DOUBLE, INTEGER, STRING, VECTOR, BOOLEAN, NO_DATA, INVALID_TYPE };

ostream &
operator<<( ostream & out, const need_e & need )
{
  if(      need == REQUIRED )     { out << "REQUIRED"; }
  else if( need == OPTIONAL )     { out << "OPTIONAL"; }
  else if( need == MULTIPLE )     { out << "MULTIPLE"; }
  else if( need == INVALID_NEED ) { out << "INVALID_NEED"; }
  else {                        out << "Error: need (" << (int)need << ") did not parse correctly... \n"; }
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
  else {                       out << "Error: type (" << (int)type << ") did not parse correctly... \n"; }
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
    cout << "Error: need (" << needStr << ") did not parse correctly... "
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
  else {
    cout << "Error: type (" << typeStr << ") did not parse correctly... "
         << "should be 'REQUIRED', 'OPTIONAL', or 'MULTIPLE'.\n";
    return INVALID_TYPE;
  }
}

struct ProblemSpecReader::AttributeAndTagBase {

  AttributeAndTagBase( const string & name, need_e need, type_e type, 
                       const vector<string> & validValues, const Tag * parent ) :
    parent_( parent ),
    name_( name ), need_( need ), type_( type ),
    validValues_( validValues ),
    occurrences_( 0 )
  {
  }

  AttributeAndTagBase( const string & name, need_e need, type_e type, 
                       const string & validValues, const Tag * parent ) :
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

  const Tag *    parent_;
  string         name_;
  need_e         need_;
  type_e         type_;
  vector<string> validValues_;
  int            occurrences_;

  ///////////////////////////////////

  string getCompleteName();

  virtual void print( bool /* recursively = false */, unsigned int level = 0, bool isTag = false ) { 

    // Fill is used to pad the Tag names so they line up better...
    if( level > 14 ) {
      // Make sure the truncation is enough so that the below 30-level*2 doesn't underflow...
      dbg << "WARNING... print truncating indention level to 14...\n";
      level = 14;
    }
    string fill;
    for( unsigned int pos = name_.size(); pos < (30-(level*2)); pos++ ) {
      fill += " ";
    }

    indent( dbg, level ); 
    dbg << (isTag ? "<" : "- " ) 
         << name_
         << (isTag ? ">" : "" ) 
         << fill << " - " << need_ << " - " << type_ << " - VVs: "
         << (validValues_.size() == 0 ? "" : "'" );
    for( unsigned int pos = 0; pos < validValues_.size(); pos++ ) {
      dbg << validValues_[pos] << " ";
    }
    dbg << (validValues_.size() == 0 ? "" : "'" )
         << "(occur'd: " << occurrences_ << ") ";
  }

};

struct ProblemSpecReader::Attribute : public ProblemSpecReader::AttributeAndTagBase { 

  Attribute( const string & name, need_e need, type_e type, const string & validValues, const Tag * parent ) :
    AttributeAndTagBase( name, need, type, validValues, parent ) {}

  virtual void print( bool recursively, unsigned int level, bool isTag = false ) {
    AttributeAndTagBase::print( recursively, level, isTag );
    dbg << "\n";
  }
};

struct ProblemSpecReader::Tag : public ProblemSpecReader::AttributeAndTagBase {

  vector<Attribute*> attributes_;
  vector<Tag*>       subTags_;

  // validValues is a _single_ string (it will be parsed as follows) that contains valid values
  // for the value of the tag.  The specification of valid values depends on the type of Tag:
  //
  //  STRING: a comma separated lists of strings, or "*" (or NULL) which means anything
  //  INTEGER/DOUBLE: "*" = any value, "positive" = a positive value, "num, num" = min, max values
  //  BOOLEAN: validValues is not allowed... because it defaults to true/false.
  //  VECTOR: FIXME... does nothing yet...
  //
  Tag( const string & name, need_e need, type_e type, const string & validValues, const Tag * parent ) :
    AttributeAndTagBase( name, need, type, validValues, parent ) {}

  Tag( const Tag * tag, const Tag * parent, need_e need ) :
       AttributeAndTagBase( tag->name_, tag->need_, tag->type_, tag->validValues_, parent ) {

    if( need == INVALID_NEED ) { 
      need_ = tag->need_;
    }
    else if( need != need_ ) {
      dbg << "Notice: need changed to " << need << "\n";
      need_ = need;
    }
    subTags_    = tag->subTags_;
    attributes_ = tag->attributes_;
  }

  virtual void print( bool recursively = false, unsigned int level = 0, bool isTag = true ) {

    AttributeAndTagBase::print( recursively, level, isTag );

    dbg << "(parent: " << (parent_ ? parent_->name_ : "NULL") << ")\n";

    for( unsigned int pos = 0; pos < attributes_.size(); pos++ ) {
      attributes_[ pos ]->print( recursively, level+1 );
    }

    if( recursively ) {
      for( unsigned int pos = 0; pos < subTags_.size(); pos++ ) {
        subTags_[pos]->print( recursively, level+1 );
      }
    }
  }
};

string
ProblemSpecReader::AttributeAndTagBase::getCompleteName() 
{
  string      result = name_;
  const Tag * tag = parent_;
    
  while( tag != NULL ) {
    result = tag->name_ + "->" + result;
    tag = tag->parent_;
  }
  return result;
}

ProblemSpecReader::Attribute *
ProblemSpecReader::findAttribute( Tag * root, const string & attrName )
{
  vector<Attribute*> & attribs = root->attributes_;

  for( unsigned int pos = 0; pos < attribs.size(); pos++ ) {
    if( attribs[ pos ]->name_ == attrName ) {
      return attribs[ pos ];
    }
  }
  return NULL;
}

ProblemSpecReader::Tag *
ProblemSpecReader::findSubTag( Tag * root, const string & tagName )
{
  vector<Tag*> & tags = root->subTags_;

  for( unsigned int pos = 0; pos < tags.size(); pos++ ) {
    if( tags[ pos ]->name_ == tagName ) {
      return tags[ pos ];
    }
  }
  return NULL;
}

// Chops up 'validValues' (based on ','s) and verifies that 'value' is in the list.
// (If validValues is empty, then 'value' is considered valid by definition.)
bool
ProblemSpecReader::validateString( const string & value, const vector<string> & validValues )
{
  if( validValues.size() == 0 ) {
    return true;
  }

  vector<string>::const_iterator iter = find( validValues.begin(), validValues.end(), value );
  if( iter != validValues.end() ) {
    return true;
  } 
  else {
    return false;
  }
}

bool
ProblemSpecReader::validateBoolean( const string & value )
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
ProblemSpecReader::validateDouble( double value, const vector<string> & validValues )
{
  if( validValues.size() == 0 ) {
    return;
  }

  if( validValues.size() == 1 ) {
    if( validValues[0] == "positive" ) {
      if( value < 0 ) {
        ostringstream error;
        error << setprecision(12);
        error << "Specified value '" << value << "' is not 'positive' (as required).";
        throw ProblemSetupException( error.str(), __FILE__, __LINE__ );
      }
    }
  }
  else if( validValues.size() == 2 ) {
    double max, min;
    sscanf( validValues[0].c_str(), "%lf", &min );
    sscanf( validValues[1].c_str(), "%lf", &max );
    if( value < min || value > max ) {
      ostringstream error;
      error << setprecision(12);
      error << "Specified value '" << value << "' is outside of valid range (" << min << ", " << max << ")";
      throw ProblemSetupException( error.str(), __FILE__, __LINE__ );
    }
  }
  else {
    // FIXME... more descriptive and catch and add correct file/line
    throw ProblemSetupException( "Invalid 'validValues' string.", NULL, 0 );
  }
}

//// For now, validValues doesn't mean anything... so it must be "".
//string
//validateVector( string value, const string & validValues )
//{
//}

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
    return false;
  }

  if( needType.size() != 2 ) {
    // FIXME: Better error handling
    throw ProblemSetupException( "Error: need/type did not parse correctly...", __FILE__, __LINE__ );
  }

  need = getNeed( needType[ 0 ] );
  type = getType( needType[ 1 ] );

  if( specs.size() == 2 ) {
    validValues = specs[1];
    if( type == NO_DATA ) {
      // FIXME: handle error
      throw ProblemSetupException( "Error: type of Tag specified as 'NO_DATA', yet has a list of validValues: '" +
                                   validValues + "'", __FILE__, __LINE__ );
    }
    else if( type == BOOLEAN ) {
      // FIXME: handle error
      throw ProblemSetupException( "Error: type of Tag specified as 'BOOLEAN', yet has list of validValues: '" +
                                   validValues + "'", __FILE__, __LINE__ );
    }
  }
  return true;
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
  separators.push_back( '\'' );

  vector<string> specs = split_string( specStr, separators );

  if( specs.size() < 1 || specs.size() > 2 ) {
    throw ProblemSetupException( "Error in getLabelAndNeedAndTypeAndValidValues...", __FILE__, __LINE__ );
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
      throw ProblemSetupException( "Error: type of Tag specified as NO_DATA, yet has a validValues component...", __FILE__, __LINE__ );
    }
    validValues = specs[1];
  }
}

void
ProblemSpecReader::parseTag( Tag * parent, const xmlNode * xmlTag )
{
  string name = to_char_ptr( xmlTag->name );
  collapse( name );

  dbg << "Parse node: " << name << "\n";

  bool hasSpecString = true;

  if( xmlTag == NULL ) {
    throw ProblemSetupException( "Error... passed in xmlTag is null...", __FILE__, __LINE__ );
  }
  else {
    if( name != "CommonTags" ) {
      if( xmlTag->properties == NULL ) {
        Tag * tag = findSubTag( commonTags_, name );
        if( tag ) {
          dbg << "Found common tag for " << name << "\n";
          hasSpecString = false;
        }
        else {
          throw ProblemSetupException( "Error (a)... <" + name + "> does not have required 'spec' attribute (eg: spec=\"REQUIRED NO_DATA\")." +
                                       "              or couldn't find in CommonTags.", __FILE__, __LINE__ );
        }
      }
      else if( xmlTag->properties->children == NULL ) {
        // FIXME... not sure what this case is...
        throw ProblemSetupException( "Error (b)... <" + name + "> does not have required 'spec' attribute (eg: spec=\"REQUIRED NO_DATA\").",
                                     __FILE__, __LINE__ );
      }
      else if( string( "spec" ) != to_char_ptr( xmlTag->properties->name ) ) {
        // FIXME... better error msg/handling
        throw ProblemSetupException( "Error (c)... <" + name + "> does not have required 'spec' attribute (eg: spec=\"REQUIRED NO_DATA\").  " +
                                     "Found attribute '" + to_char_ptr( xmlTag->properties->name ) + "' instead.", __FILE__, __LINE__ );
      }
    }
  }

  need_e need = INVALID_NEED;
  type_e type;
  string validValues;
  bool   common;

  if( hasSpecString ) {
    string specStr = to_char_ptr( xmlTag->properties->children->content );
    common = !getNeedAndTypeAndValidValues( specStr, need, type, validValues );
  }
  else {
    common = true;
  }

  Tag * newTag;

  if( common ) {
    // Find this tag in the list of common tags... 
    Tag * commonTag = findSubTag( commonTags_, name );
    if( !commonTag ) {
      throw ProblemSetupException( "Error, commonTag <" + name + "> not found... was looking for a common tag " +
                                   "because spec string only had one entry.", __FILE__, __LINE__ );
    }
    newTag = new Tag( commonTag, parent, need );
  }
  else {
    newTag = new Tag( name, need, type, validValues, parent );
  }

  parent->subTags_.push_back( newTag );

  // Handle attributes... (if applicable)
  if( hasSpecString && xmlTag->properties->next != NULL ) {
    for( xmlAttr * child = xmlTag->properties->next; child != 0; child = child->next) {
      if( child->type == XML_ATTRIBUTE_NODE ) {

        need_e need;
        type_e type;
        string label, validValues;

        const string attrName = to_char_ptr( child->name );

        if( attrName.find( "attribute") == 0 ) { // attribute string begins with "attribute"
          string specStr = to_char_ptr( child->children->content );
          getLabelAndNeedAndTypeAndValidValues( specStr, label, need, type, validValues );

          newTag->attributes_.push_back( new Attribute( label, need, type, validValues, newTag ) );
        }
        else if( attrName.find( "children") == 0 ) {  // attribute string begins with "children"
          // FIXME:
          dbg << "WARNING: Code doesn't handle 'children' directives yet...\n";
        }
      }
    }
  }

  // Handle any children of the node...
  for( xmlNode * child = xmlTag->children; child != 0; child = child->next) {
    if( child->type == XML_ELEMENT_NODE ) {

      string node = to_char_ptr( child->name );
      collapse( node );

      parseTag( newTag, child );
    }
  }
}

void
ProblemSpecReader::parseValidationFile()
{
  dbg << "parsing ups_spec.xml\n";

  xmlDocPtr doc; /* the resulting document tree */
  
  const string valFile = "inputs/ups_spec.xml";

  doc = xmlReadFile( valFile.c_str(), 0, XML_PARSE_PEDANTIC );
  
  if (doc == 0) {
    cout << "\nWARNING: can't find '" << valFile << "'... .ups validation will not take place.\n\n";
    return;
    //throw ProblemSetupException( "Error opening " + valFile, __FILE__, __LINE__ );
  }

  xmlNode * root = xmlDocGetRootElement( doc );

  uintahSpec_ = new Tag( "Uintah_specification", REQUIRED, NO_DATA, "", NULL );
  commonTags_ = new Tag( "CommonTags", REQUIRED, NO_DATA, "", NULL );

  string tagName = to_char_ptr( root->name );

  if( tagName != "Uintah_specification" ) {
    throw ProblemSetupException( valFile + " does not appear to be valid... First tag should be\n" +
                                 + "<Uintah_specification>, but found: " + tagName,
                                 __FILE__, __LINE__ );
  }

  // Find <CommonTags> (if it exists)
  bool commonTagsFound = false;
  for( xmlNode * child = root->children; child != 0; child = child->next) {
    if( child->type == XML_ELEMENT_NODE ) {
      string tagName = to_char_ptr( child->name );
    
      if( tagName == "CommonTags" ) {
        commonTagsFound = true;
        // Find first (real) child of the <CommonTags> block...
        xmlNode * gc = child->children; // Grand Child
        while( gc != NULL ) {
          if( gc->type == XML_ELEMENT_NODE ) {
            parseTag( commonTags_, gc );
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
      string tagName = to_char_ptr( child->name );
      if( tagName != "CommonTags" ) { // We've already handled the Common Tags...
        parseTag( uintahSpec_, child );
      }
    }
  }
  dbg << "done parsing ups_spec.xml\n";
}

void
ProblemSpecReader::validateText( AttributeAndTagBase * root, const string & text )
{
  string classType = "Attribute";
  string completeName = root->getCompleteName();

  Tag * testIfTag = dynamic_cast<Tag*>( root );
  if( testIfTag ) {
    classType == "Tag";
  }

  // Verify that 'the text' of the node exists or doesn't exist as required... 
  //    <myTag>the text</myTag>
  //
  if( root->type_ == NO_DATA ) {
    if( text != "" ) {
      throw ProblemSetupException( classType + " <" + completeName + "> should not have data (but has: '" + text +
                                   "').  Please fix XML in .ups file or correct validation Tag list.",
                                   __FILE__, __LINE__ );
    }
  }
  else { // type != NO_DATA
    if( text == "" ) {
      stringstream error_stream;
      error_stream << classType << " <" << completeName << "> should have a value (of type: " 
                   << root->type_ << ") but is empty. " << "Please fix XML in .ups file or\n"
                   << "correct validation Tag list.";
      throw ProblemSetupException( error_stream.str(), __FILE__, __LINE__ );
    }
  }
  
  switch( root->type_ ) {
  case DOUBLE:
    {
      // WARNING: this sscanf isn't a sufficient test to validate that a double (and only
      //          a double exists in the text... 
      double value;
      int    num = sscanf( text.c_str(), "%lf", &value );
      
      if( num != 1 ) {
        throw ProblemSetupException( classType + " <" + completeName + "> should have a double value (but has: '" + text +
                                     "').  Please fix XML in .ups file or correct validation Tag list.",
                                     __FILE__, __LINE__ );
      } 
      else {
        validateDouble( value, root->validValues_ );
      }
    }
    break;
  case INTEGER:
    {
      int value;
      // WARNING: this is probably not a sufficient check for an integer...
      int     num = sscanf( text.c_str(), "%d", &value );
      //cout << "VALUE is " << value << "\n";
      if( num != 1 ) {
        throw ProblemSetupException( classType + " <" + completeName + "> should have an integer value (but has: '" + text +
                                     "').  Please fix XML in .ups file or correct validation Tag list.",
                                     __FILE__, __LINE__ );
      }
      else {
        validateDouble( (double)value, root->validValues_ );
      }
    }
    break;
  case STRING:
    if( !validateString( text, root->validValues_ ) ) {
      throw ProblemSetupException( "Invalid string value for " + classType + ": " + completeName + ". '" + 
                                   text + "' not found in this list:\n" + toString( root->validValues_ ),
                                   __FILE__, __LINE__ );
    }
    break;
  case BOOLEAN:
    if( !validateBoolean( text ) ) {
      throw ProblemSetupException( "Invalid boolean string value for " + classType + " <" + completeName +
                                   ">.  Value must be either 'true', or 'false', but '" + text + "' was found...",
                                   __FILE__, __LINE__ );
    }
    break;
  case VECTOR:
    {
      double val1, val2, val3;
      int    num = sscanf( text.c_str(), "[%lf,%lf,%lf]", &val1, &val2, &val3 );
      if( num != 3 ) {
        throw ProblemSetupException( classType + " ('" + completeName + "') should have a Vector value (but has: '" +
                                     text + "').  Please fix XML in .ups file or correct validation Tag list.",
                                     __FILE__, __LINE__ );
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
ProblemSpecReader::validateAttribute( Tag * root, xmlAttr * attr )
{
  if( attr == NULL ) {
    // This should never actually happen...
    return;
  }

  const string attrName = to_char_ptr( attr->name );

  Attribute * attribute = findAttribute( root, attrName );

  if( !attribute ) {
    // FIXME... better error message
    throw ProblemSetupException( "Error, attribute ('" + attrName + "') not found...", __FILE__, __LINE__ );
  }

  attribute->occurrences_++;

  const char * attrContent = to_char_ptr(attr->children->content);
  validateText( attribute, attrContent );
}

void
ProblemSpecReader::validate( Tag * root, const ProblemSpec * ps, unsigned int level /* = 0 */ )
{
  if( !uintahSpec_ ) {
    return;
  }

  if( dbg.active() ) {
    dbg << "ProblemSpec::validate - ";
    indent( dbg, level ); dbg << ps->getNode()->name << "\n";
  }

  /////////////////////////////////////////////////////////////////////
  // Zero out the number of occurrences of all the sub tags and
  // attributes so occurrences in previous tags will not be counted
  // against the current tag.
  //
  for( unsigned int pos = 0; pos < root->subTags_.size(); pos++ ) {
    root->subTags_[ pos ]->occurrences_ = 0;
  }
  for( unsigned int pos = 0; pos < root->attributes_.size(); pos++ ) {
    root->attributes_[ pos ]->occurrences_ = 0;
  }
  /////////////////////////////////////////////////////////////////////

  // Run through all the nodes of the ProblemSpec (from the .ups file) to validate them:
  //
  //     FYI, child->children would only be null for a tag like this <myTag></myTag>
  //     If it was: <myTag>  </myTag> then children is not null... it just filled with blanks.

  int     numTextNodes = 0;
  string  text = "";

  for( xmlNode *child = ps->getNode()->children; child != 0; child = child->next) {

    if (child->type == XML_TEXT_NODE) {

      text = to_char_ptr( child->content );
      collapse( text );

      if( text != "" ) {
        if( numTextNodes == 1 ) {
          throw ProblemSetupException( string( "Node has multiple text (non-tag) nodes in it, but should have only one!\n" ) +
                                       "       The 2nd text node contains: '" + text + "'", __FILE__, __LINE__ );
        }
        numTextNodes++;
      }
    }
    else if( child->type == XML_COMMENT_NODE ) {
      continue;
    }
    else if( child->type != XML_ELEMENT_NODE ) {
      throw ProblemSetupException( string( "Node has an unknown type of child node... child node's name is '" ) +
                                   to_char_ptr( child->name ) + "'", __FILE__, __LINE__ );
    }
    else {
      const char * tagName = to_char_ptr( child->name );
      Tag *        tag = findSubTag( root, tagName );

      if( !tag ) { 
        throw ProblemSetupException( string( "Tag '" ) + tagName + "' not valid (for <" + root->getCompleteName() + 
                                     ">).  Please fix XML in .ups file or correct validation Tag list.", __FILE__, __LINE__ );
      }

      tag->occurrences_++;

      if( tag->occurrences_ > 1 && tag->need_ != MULTIPLE ) {
        throw ProblemSetupException( string( "Tag <" ) + tag->getCompleteName() +
                                     "> occurred too many times.  Please fix XML in .ups file or correct validation Tag list.",
                                     __FILE__, __LINE__ );
      }

      // Handle sub tag
      ProblemSpec gcPs( child );
      Tag * childTag = findSubTag( root, to_char_ptr( child->name ) );

      if( childTag == NULL ) {
        throw ProblemSetupException( "Error, childtag is null...", __FILE__, __LINE__ );
      }
      validate( childTag, &gcPs, level+1 );
    }
    
    validateText( root, text );

  } // end for child in d_node->children
  
  // Validate elements
  xmlAttr* attr = ps->getNode()->properties;

  if( root->attributes_.size() == 0 && attr ) {
    // FIXME better error message.
    throw ProblemSetupException( "Tag " + root->getCompleteName() + " has an element ('" + to_char_ptr( attr->name ) + 
                                 "'), but spec says there are none...", __FILE__, __LINE__ );
  }

  for (; attr != 0; attr = attr->next) {
    if (attr->type == XML_ATTRIBUTE_NODE) {
      validateAttribute( root, attr );
    }
    // else skip comments, blank lines, etc...
  }

  // Verify that all REQUIRED attributes were found:
  for( unsigned int pos = 0; pos < root->attributes_.size(); pos++ ) {
    Attribute * attr = root->attributes_[ pos ];
    if( attr->need_ == REQUIRED && attr->occurrences_ == 0 ) {
      throw ProblemSetupException( string( "Required attribute '" ) + attr->getCompleteName() +
                                   "' missing.  Please fix XML in .ups file or correct validation Attribute list.",
                                   __FILE__, __LINE__ );
    }
  }

  // Verify that all REQUIRED tags were found:
  for( unsigned int pos = 0; pos < root->subTags_.size(); pos++ ) {
    Tag * tag = root->subTags_[ pos ];
    if( tag->need_ == REQUIRED && tag->occurrences_ == 0 ) {
      throw ProblemSetupException( string( "Required tag '" ) + tag->getCompleteName() +
                                   "' missing.  Please fix XML in .ups file or correct validation Tag list.",
                                   __FILE__, __LINE__ );
    }
  }
} // end validate()

void
ProblemSpecReader::validateProblemSpec( ProblemSpecP & prob_spec )
{
  // Currently, the readInputFile() (and thus this validation) is called
  // multiple times (once for the initial .ups, but then again for 
  // saving data archives and such...)  this (temporary?) hack is used
  // to only validate the initial .ups file.
  //

  if( !uintahSpec_ ) {
    parseValidationFile();

    if( dbg.active() ) {
      dbg << "-----------------------------------------------------------------\n";
      dbg << "-- uintah spec: \n";
      dbg << "\n";
      uintahSpec_->print( true );
      dbg << "-----------------------------------------------------------------\n";
    }

    try {
      validate( uintahSpec_, prob_spec.get_rep() );
    }
    catch( ProblemSetupException & pse ) {
      cout << "\n";
      cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
      cout << "!! WARNING: Your .ups file did not parse successfully...\n";
      cout << "!!          Soon this will be a fatal error... please\n";
      cout << "!!          fix your .ups file or update the ups_spec.xml\n";
      cout << "!!          specification.  Reason for failure is:\n";
      cout << "\n";
      cout << pse.message() << "\n";
      cout << "\n";
      cout << "!!\n";
      cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n";
    }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////

ProblemSpecReader::Tag * ProblemSpecReader::commonTags_ = NULL;
ProblemSpecReader::Tag * ProblemSpecReader::uintahSpec_ = NULL;

ProblemSpecReader::ProblemSpecReader( const string & upsFilename ) :
  d_upsFilename( upsFilename )
{
  if( dbg.active() ) {
    cout << "FYI: ProblemSpecReader debug stream is active!\n";
  }
}

ProblemSpecReader::~ProblemSpecReader()
{
}

ProblemSpecP
ProblemSpecReader::readInputFile()
{
  if (d_xmlData != 0)
    return d_xmlData;

  ProblemSpecP prob_spec;
  static bool initialized = false;

  if (!initialized) {
    LIBXML_TEST_VERSION;
    initialized = true;
  }
  
  xmlDocPtr doc; /* the resulting document tree */
  
  doc = xmlReadFile(d_upsFilename.c_str(), 0, XML_PARSE_PEDANTIC);
  
  /* check if parsing suceeded */
  if (doc == 0) {
    if ( !getInfo( d_upsFilename ) ) { // Stat'ing the file failed... so let's try testing the filesystem...
      // Find the directory this file is in...
      string directory = d_upsFilename;
      unsigned int index;
      for( index = directory.length()-1; index >= 0; --index ) {
        //strip off characters after last /
        if (directory[index] == '/')
          break;
      }
      directory = directory.substr(0,index+1);

      stringstream error_stream;
      
      if( !testFilesystem( directory, error_stream, Parallel::getMPIRank() ) ) {
        cout << error_stream.str();
        cout.flush();
      }
    }
    throw ProblemSetupException("Error reading file: "+d_upsFilename, __FILE__, __LINE__);
  }
    
  // you must free doc when you are done.
  // Add the parser contents to the ProblemSpecP
  prob_spec = scinew ProblemSpec(xmlDocGetRootElement(doc), true);

  resolveIncludes(prob_spec);

  validateProblemSpec( prob_spec );

  d_xmlData = prob_spec;
  return prob_spec;
}

void
ProblemSpecReader::resolveIncludes(ProblemSpecP params)
{
  // find the directory the current file was in, and if the includes are 
  // not an absolute path, have them for relative to that directory
  string directory = d_upsFilename;

  int index;
  for( index = (int)directory.length()-1; index >= 0; --index ) {
    //strip off characters after last /
    if (directory[index] == '/')
      break;
  }
  directory = directory.substr(0,index+1);

  ProblemSpecP child = params->getFirstChild();
  while (child != 0) {
    if (child->getNodeType() == XML_ELEMENT_NODE) {
      string str = child->getNodeName();
      // look for the include tag
      if (str == "include") {
        map<string, string> attributes;
        child->getAttributes(attributes);
        string href = attributes["href"];

        // not absolute path, append href to directory
        if (href[0] != '/')
          href = directory + href;
        if (href == "")
          throw ProblemSetupException("No href attributes in include tag", __FILE__, __LINE__);
        
        // open the file, read it, and replace the index node
        ProblemSpecReader *psr = scinew ProblemSpecReader(href);
        ProblemSpecP include = psr->readInputFile();
        delete psr;
        // nodes to be substituted must be enclosed in a 
        // "Uintah_Include" node

        if (include->getNodeName() == "Uintah_Include" || 
            include->getNodeName() == "Uintah_specification") {
          ProblemSpecP incChild = include->getFirstChild();
          while (incChild != 0) {
            //make include be created from same document that created params
            ProblemSpecP newnode = child->importNode(incChild, true);
            resolveIncludes(newnode);
            xmlAddPrevSibling(child->getNode(), newnode->getNode());
            incChild = incChild->getNextSibling();
          }
          ProblemSpecP temp = child->getNextSibling();
          params->removeChild(child);
          child = temp;
          continue;
        }
        else {
          throw ProblemSetupException("No href attributes in include tag", __FILE__, __LINE__);
        }
      }
      // recurse on child's children
      resolveIncludes(child);
    }
    child = child->getNextSibling();
  } // end while (child != 0)

} // end resolveIncludes()
