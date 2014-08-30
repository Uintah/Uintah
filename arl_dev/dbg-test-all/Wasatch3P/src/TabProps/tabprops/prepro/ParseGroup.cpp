/*
 * Copyright (c) 2012 The University of Utah
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

#include "ParseGroup.h"

#include <iostream>
#include <stdexcept>

#include <boost/property_tree/xml_parser.hpp>
namespace bpt = boost::property_tree;
#include <boost/foreach.hpp>

using std::string;

//===================================================================

ParseGroup::ParseGroup( const string fileName )
: name_( fileName ),
  tree_( new Entry() ),
  builtIt_( true )
{
  try {
    bpt::read_xml( fileName,
                   const_cast<Entry&>(*tree_),
                   bpt::xml_parser::trim_whitespace|bpt::xml_parser::no_comments );
  }
  catch( std::exception& err ){
    std::ostringstream msg;
    msg << "Error parsing '" << fileName << "'" << std::endl
        << err.what() << std::endl;
    throw std::runtime_error( msg.str() );
  }
}

ParseGroup::ParseGroup( std::istream& istr )
: name_(""),
  tree_( new Entry() ),
  builtIt_( true )
{
  try{
    bpt::read_xml( istr,
                   const_cast<Entry&>(*tree_),
                   bpt::xml_parser::trim_whitespace|bpt::xml_parser::no_comments );
  }
  catch( std::exception& err ){
    std::ostringstream msg;
    msg << "Error parsing from supplied istream" << std::endl
        << err.what() << std::endl;
    throw std::runtime_error( msg.str() );
  }
}

ParseGroup::ParseGroup( const string key, const Entry* const tree )
: name_( key  ),
  tree_( tree ),
  builtIt_( false )
{}

ParseGroup::ParseGroup( const ParseGroup& other )
: name_( other.name_ ),
  tree_( other.tree_ ),
  builtIt_( false )
{}

ParseGroup&
ParseGroup::operator=( const ParseGroup& other )
{
  name_ = other.name_;
  tree_ = other.tree_;
  builtIt_ = false;
  return *this;
}

//-------------------------------------------------------------------

ParseGroup::~ParseGroup()
{
  if( builtIt_ ) delete tree_;
}

//-------------------------------------------------------------------

void
ParseGroup::display( std::ostream& os ) const
{
  ParseGroup::display( os, 0, *tree_ );
//  bpt::xml_parser::write_xml(os,*tree_);
}

void
ParseGroup::display( std::ostream& os, const unsigned depth, const Entry& tree )
{
  using namespace std;

//  os << tree.get_value<string>() << endl;

  BOOST_FOREACH( const Entry::value_type& v, tree.get_child("") ) {

    const Entry& subtree = v.second;
    const string nodestr = tree.get<string>(v.first);
    const string indent = string("").assign(depth*2,' ');
    const bool isAttr = ( v.first == "<xmlattr>" );

    // special handling for comments
    if( v.first == "<xmlcomment>" ){
      os << "<!--" << nodestr << "-->" << endl;
      continue;
    }
    // special handling for attributes
    if( isAttr ){
      // skip the placeholder <xmlattr> tag and suck out the attributes
      BOOST_FOREACH( const Entry::value_type& v2, subtree.get_child("") ){
        const string nstr = subtree.get<string>(v2.first);
        os << " " << v2.first;
        if ( nstr.length() > 0 ) os << "=\"" << nstr <<"\"";
      }
      os << ">" << std::endl;
    }
    else{
      os << indent << "<" << v.first;
      if( nodestr.length() > 0 ) os << ">" << nodestr << "</" << v.first;

      if( subtree.find("<xmlattr>") == subtree.not_found() ){
        os << ">" << std::endl;
      }
      display(os,depth+1,subtree);  // recurse down the hierarchy

      if( nodestr.length() == 0 )
        os << indent << "</" << v.first << ">" << endl;
    }
  }
}

//-------------------------------------------------------------------

bool
ParseGroup::has_child( const std::string& key ) const
{
  return ( tree_->find(key) != tree_->not_found() );
}

//-------------------------------------------------------------------

std::string
ParseGroup::name() const{
  return name_;
}

//-------------------------------------------------------------------

void
ParseGroup::check_exists( const string& key ) const
{
  if( !has_child(key) ){
    std::ostringstream msg;
    msg << "Parser error: could not find requested child '" << key << "' in XML group '"
        << name() << "'" << std::endl;
    throw std::invalid_argument( msg.str() );
  }
}

//-------------------------------------------------------------------

ParseGroup
ParseGroup::get_child( const string key ) const
{
  check_exists(key);
  return ParseGroup( key, &tree_->get_child(key) );
}

//-------------------------------------------------------------------

template<typename T> T
ParseGroup::get_value() const{
  return tree_->get_value<T>();
}

template<typename T> T
ParseGroup::get_value( const string key ) const{
  check_exists(key);
  return get_child(key).get_value<T>();
}

template<typename T> T
ParseGroup::get_value( const string key, T t ) const{
  if( tree_->find(key) == tree_->not_found() ){
    std::cout << "NOTE: setting default value of '" << t << "' for " << key << std::endl;
    return t;
  }
  return tree_->get<T>(key);
}

//-------------------------------------------------------------------

template<typename T> T
ParseGroup::get_attribute( const string key ) const{
  if( !has_child("<xmlattr>") ){
    std::ostringstream msg;
    msg << "Could not find attribute '" << key << "' on node '" << name() << "'" << std::endl;
    throw std::invalid_argument(msg.str());
  }
  return get_child("<xmlattr>").get_value<T>(key);
}

template<typename T> T
ParseGroup::get_attribute( const string key, const T t ) const{
  if( has_child("<xmlattr>") )
    return get_child("<xmlattr>").get_value<T>(key);
  return t;
}

//-------------------------------------------------------------------

#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>

template<typename T>
std::vector<T>
ParseGroup::get_value_vec( const string key ) const{
  return get_child(key).get_value_vec<T>();
}

template<typename T>
std::vector<T>
ParseGroup::get_value_vec() const{
  using namespace std;
  vector<T> values;
  typedef boost::tokenizer< boost::char_separator<char> > Token;
  boost::char_separator<char> sep(" ;,");
  const std::string str = get_value<string>();
  const Token token( str, sep );
  for(Token::const_iterator i=token.begin(); i!=token.end();++i) {
    values.push_back( boost::lexical_cast<T>(*i) );
  }
  return values;
}

//-------------------------------------------------------------------

template<typename T>
std::map<string,T>
ParseGroup::get_children() const{
  std::map<string,T> result;
  BOOST_FOREACH( const Entry::value_type& v2, tree_->get_child("") ){
    result[ v2.first ] = v2.second.get_value<T>();
  }
  return result;
}

//-------------------------------------------------------------------

ParseGroup::const_iterator
ParseGroup::begin( const std::string name ) const
{
  std::pair<Entry::const_assoc_iterator,Entry::const_assoc_iterator> iters = tree_->equal_range(name);
  return const_iterator( iters.first, iters.second );
}

ParseGroup::const_iterator
ParseGroup::end( const std::string name ) const{
  std::pair<Entry::const_assoc_iterator,Entry::const_assoc_iterator> iters = tree_->equal_range(name);
  return const_iterator( iters.second, iters.second );
}

//===================================================================

// explicit template instantiation
#define INSTANTIATE( T )                                                \
  template T ParseGroup::get_value<T>() const;                          \
  template T ParseGroup::get_value<T>(const string) const;              \
  template T ParseGroup::get_value<T>(const string, T) const;           \
  template T ParseGroup::get_attribute<T>(const string) const;          \
  template std::vector<T> ParseGroup::get_value_vec() const;            \
  template std::vector<T> ParseGroup::get_value_vec(const string) const;\
  template std::map<string,T> ParseGroup::get_children() const;

INSTANTIATE( double   )
INSTANTIATE( float    )
INSTANTIATE( int      )
INSTANTIATE( size_t   )
INSTANTIATE( unsigned )
INSTANTIATE( string   )

//-------------------------------------------------------------------

bool
ParseGroup::operator==(const ParseGroup& other)
{
  return ( *tree_ == *other.tree_ );
}

//-------------------------------------------------------------------

std::ostream& operator<<( std::ostream& os, const ParseGroup& pg )
{
  pg.display(os);
  return os;
}

//===================================================================

ParseGroupIterator::ParseGroupIterator( Iter ii, Iter iend )
: iter_( ii ),
  iend_( iend ),
  pg_( ii==iend ? "END" : ii->first,
       ii==iend ? NULL : &ii->second )
{}

ParseGroupIterator::ParseGroupIterator( const ParseGroupIterator& other )
: iter_( other.iter_ ),
  iend_( other.iend_ ),
  pg_  ( other.pg_   )
{}

bool
ParseGroupIterator::operator!=( const ParseGroupIterator& other ) const{
  return iter_ != other.iter_;
}

const ParseGroup*
ParseGroupIterator::operator->() const{
  assert( iter_ != iend_ );
  return &pg_;
}

const ParseGroup&
ParseGroupIterator::operator*() const{
  assert( iter_ != iend_ );
  return pg_;
}

void
ParseGroupIterator::operator++(){
  assert( iter_ != iend_ );
  ++iter_;
  if( iter_ != iend_ )  pg_ = ParseGroup( iter_->first, &iter_->second );
}

//===================================================================
