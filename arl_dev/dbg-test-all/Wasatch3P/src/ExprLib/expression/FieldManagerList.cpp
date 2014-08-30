/*
 * FieldManagerLIst.cpp
 *
 *  Created on: Oct 26, 2012
 *      Author: "James C. Sutherland"
 *
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


#include "FieldManagerList.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace Expr{

FieldManagerList*
extract_field_manager_list( FMLMap& fmls, const int id )
{
  // If we only have one FieldManagerList in the map, don't look for a match,
  // just return it. This maintains backward compatibility.
  if( fmls.size() == 1 ) return fmls.begin()->second;

  FMLMap::iterator ifml = fmls.find(id);
  if( ifml == fmls.end() ){
    std::ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << std::endl
        << "ERROR!  When using multiple FieldManagerLists, you must set the id to" << std::endl
        << "        use with each expression!  When registering the expression, call" << std::endl
        << "         factory.set_field_manager_list( exprID, listID );" << std::endl;
    throw std::runtime_error( msg.str() );
  }
  return ifml->second;
}

const FieldManagerList*
extract_field_manager_list( const FMLMap& fmls, const int id )
{
  // If we only have one FieldManagerList in the map, don't look for a match,
  // just return it. This maintains backward compatibility.
  if( fmls.size() == 1 ) return fmls.begin()->second;

  FMLMap::const_iterator ifml = fmls.find(id);
  if( ifml == fmls.end() ){
    std::ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << std::endl
        << "ERROR!  When using multiple FieldManagerLists, you must set the id to" << std::endl
        << "        use with each expression!  When registering the expression, call" << std::endl
        << "         factory.set_field_manager_list( exprID, listID );" << std::endl;
    throw std::runtime_error( msg.str() );
  }
  return ifml->second;
}

//===================================================================

void
FieldManagerList::deallocate_fields()
{
  for( iterator i=begin(); i!=end(); ++i ){
    i->second->deallocate_fields();
  }
}

//--------------------------------------------------------------------

void
FieldManagerList::
dump_fields( std::ostream& os ) const
{
  os << std::endl
     << "***********************************************************" << std::endl
     << "** Fields registered on field manager list named: " << listName_ << std::endl
     << "***********************************************************"
     << std::endl;

  for( const_iterator i=begin(); i!=end(); ++i ){
    i->second->dump_fields(os);
    os << std::endl;
  }
}

//--------------------------------------------------------------------

std::string
FieldManagerList::name_counter()
{
  static int n=0;
  std::ostringstream s;
  s << "Field Manager List " << n++;
  return s.str();
}

//--------------------------------------------------------------------

} // namespace Expr
