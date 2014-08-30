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
#include "FieldDeps.h"

#include <expression/FieldManagerList.h>

namespace Expr{

  //------------------------------------------------------------------

  std::ostream& operator<<( std::ostream& out, const FieldDeps& fd )
  {
    const FieldDeps::FldHelpers& fh = fd.field_helpers();
    FieldDeps::FldHelpers::const_iterator ifh=fh.begin();
    out << (*ifh)->tag();
    ++ifh;
    for( ; ifh!=fh.end(); ++ifh ){
      out << ", " << (*ifh)->tag();
    }
    return out;
  }

  //------------------------------------------------------------------

  FieldDeps::FieldDeps(){}

  //------------------------------------------------------------------

  FieldDeps::FieldDeps( const FieldDeps& deps )
  {
    for( FldHelpers::const_iterator i=deps.fldHelpers_.begin(); i!=deps.fldHelpers_.end(); ++i ){
      fldHelpers_.push_back( (*i)->clone() );
    }
  }

  //------------------------------------------------------------------

  FieldDeps::~FieldDeps()
  {
    for( FldHelpers::iterator i=fldHelpers_.begin(); i!=fldHelpers_.end(); ++i ){
      delete *i;
    }
  }

  //------------------------------------------------------------------

  void
  FieldDeps::register_fields( FieldManagerList& fml )
  {
    for( FldHelpers::iterator i=fldHelpers_.begin(); i!=fldHelpers_.end(); ++i ){
      (*i)->register_field(fml);
    }
  }

  //------------------------------------------------------------------

  void
  FieldDeps::prep_field_for_consuption( FieldManagerList& fml, short int deviceIndex ){
    for( FldHelpers::iterator i = fldHelpers_.begin(); i != fldHelpers_.end(); ++i){
      (*i)->prep_field_for_consumption( fml, deviceIndex );
    }
  }

  //------------------------------------------------------------------

  void
  FieldDeps::validate_field_location( FieldManagerList& fml, short int deviceIndex ){
    for( FldHelpers::iterator i = fldHelpers_.begin(); i != fldHelpers_.end(); ++i){
      (*i)->validate_field_location( fml, deviceIndex );
    }
  }

  //------------------------------------------------------------------

  void
  FieldDeps::set_active_field_location( FieldManagerList& fml, short int deviceIndex ){
    for( FldHelpers::iterator i = fldHelpers_.begin(); i != fldHelpers_.end(); ++i){
      (*i)->set_active_field_location( fml, deviceIndex );
    }
  }

  //------------------------------------------------------------------

  void
  FieldDeps::set_memory_manager( FieldManagerList& fml, const MemoryManager m, const short int deviceIndex )
  {
    for( FldHelpers::iterator i=fldHelpers_.begin(); i!=fldHelpers_.end(); ++i ){
      (*i)->set_field_memory_manager(fml, m, deviceIndex);
    }
  }

  //------------------------------------------------------------------

  bool
  FieldDeps::release_fields( FieldManagerList& fml )
  {
    bool ok = true;
    for( FldHelpers::iterator i=fldHelpers_.begin(); i!=fldHelpers_.end(); ++i ){
      ok = ok & (*i)->release_field(fml);
    }
    return ok;
  }

  //------------------------------------------------------------------
  bool
  FieldDeps::lock_fields( FieldManagerList& fml )
  {
    bool ok = true;
    for( FldHelpers::iterator i=fldHelpers_.begin(); i!=fldHelpers_.end(); ++i ){
      ok = ok & (*i)->lock_field(fml);
    }
    return ok;
  }

  //------------------------------------------------------------------
  bool
  FieldDeps::unlock_fields( FieldManagerList& fml )
  {
    bool ok = true;
    for( FldHelpers::iterator i=fldHelpers_.begin(); i!=fldHelpers_.end(); ++i ){
      ok = ok & (*i)->unlock_field(fml);
    }
    return ok;
  }


  //------------------------------------------------------------------


} // namespace Expr
