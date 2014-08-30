/*
 * Copyright (c) 2011 The University of Utah
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

#include <expression/Tag.h>

#include <sstream>
#include <map>

namespace Expr{


  FieldID get_new_id(){
    static FieldID s=0;
    return s++;
  }

  FieldID get_id( const std::string& name ){
    static std::map< std::string, FieldID > tagIdMap;
    std::map<std::string,FieldID>::iterator i=tagIdMap.find( name );
    if( i==tagIdMap.end() ){
      const FieldID id = get_new_id();
      i = tagIdMap.insert( make_pair(name,id) ).first;
      return id;
    }
    return i->second;
  }

  inline FieldID get_id( const std::string& name, const Context c ){
    return get_id( name+context2str(c) );
  }

  inline FieldID get_id( const Tag& tag ){
    return get_id( tag.name(), tag.context() );
  }


  Tag::Tag( const std::string fieldName,
            const Context tag )
    : fieldName_( fieldName ),
      context_  ( tag ),
      id_( get_id(fieldName_,context_) )
  {}

  Tag::Tag( const Tag& s )
    : fieldName_( s.fieldName_ ),
      context_  ( s.context_   ),
      id_( get_id(fieldName_,context_) )
  {}


  Tag::Tag( const Tag& el,
            const std::string s )
    : fieldName_( el.fieldName_ + s ),
      context_  ( el.context_ ),
      id_( get_id(fieldName_,context_) )
  {}

  Tag::Tag()
    : fieldName_(""),
      context_( INVALID_CONTEXT ),
      id_( get_id(fieldName_,context_) )
  {}

  Tag&
  Tag::operator=( const Tag& s )
  {
    fieldName_ = s.fieldName_;
    context_   = s.context_;
    id_        = s.id_;
    return *this;
  }

   std::ostream&
  operator<<( std::ostream& os, const Tag& s )
  {
    os << "( " << s.field_name() << ", " << s.context() << " )";
    return os;
  }

  std::ostream&
  operator<<( std::ostream& os, const TagList& tl )
  {
    if( tl.size() == 0 ) return os;
    TagList::const_iterator it=tl.begin();
    os << *it; ++it;
    for( ; it!=tl.end(); ++it ){
      os << ", " << *it;
    }
    return os;
  }

  std::ostream&
  operator<<( std::ostream& os, const TagSet& tl )
  {
    if( tl.size() == 0 ) return os;
    TagSet::const_iterator it=tl.begin();
    os << *it; ++it;
    for( ; it!=tl.end(); ++it ){
      os << ", " << *it;
    }
    return os;
  }

} // namespace Expr
