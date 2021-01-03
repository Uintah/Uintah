/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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

#include <CCA/Components/Wasatch/Expressions/ScalarEOSHelper.h>

namespace ScalarEOS {

  Expr::Tag
  resolve_tag( const FieldSelector field,
               const FieldTagInfo& info )
  {
    Expr::Tag tag;
    const FieldTagInfo::const_iterator ifld = info.find( field );
    if( ifld != info.end() ) tag = ifld->second;
    return tag;
  }

  Expr::TagList
  resolve_tag_list( const FieldSelector     field,
                    const FieldTagListInfo& info )
  {
    Expr::TagList tagList;
    const FieldTagListInfo::const_iterator ifld = info.find( field );
    if( ifld != info.end() ) tagList = ifld->second;
    return tagList;
  }

  //------------------------------------------------------------------

  Expr::TagList
  assemble_src_tags( const Expr::TagList srcTags,
                     const FieldTagInfo  info )
  {
    const FieldTagInfo::const_iterator ifld = info.find( SOURCE_TERM );
    if( ifld != info.end() ){
      const Expr::Tag tag = ifld->second;

      // ensure that info[SOURCE_TERM] is not appended to srcTags if it's
      // found in srcTags
      if( std::find( srcTags.begin(), srcTags.end(), tag ) == srcTags.end() ){
        return Expr::tag_list( srcTags, tag );
      }

      else{
        std::ostringstream msg;
        msg << "ERROR: tag in info[SOURCE_TERM] was found in list of source tags"
        << std::endl;
        throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }
    }
    return srcTags;
  }

  //------------------------------------------------------------------
}
