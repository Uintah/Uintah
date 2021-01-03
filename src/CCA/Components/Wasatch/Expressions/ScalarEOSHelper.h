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

//-- Uintah Includes --//
#include <Core/Exceptions/InvalidValue.h>
#include <CCA/Components/Wasatch/Expressions/ScalarEOSCoupling.h>
#include <Core/Exceptions/ProblemSetupException.h>

#ifndef ScalarEOSHelper_h
#define ScalarEOSHelper_h

/**
 *  \ingroup  Expressions
 *  \class    ScalarEOSHelper
 *  \author   Josh McConnell
 *  \date     October, 2018
 *
 *  \brief Helper functions for ScalarEOSCoupling
 */

namespace ScalarEOS {

  Expr::Tag
  resolve_tag( const FieldSelector field,
               const FieldTagInfo& info );

  Expr::TagList
  resolve_tag_list( const FieldSelector     field,
                    const FieldTagListInfo& info );

  Expr::TagList
  assemble_src_tags( const Expr::TagList srcTags,
                     const FieldTagInfo  info );

}
#endif
