/**
 * \file ExprDeps.h
 * \author James C. Sutherland
 *
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
#ifndef ExprDeps_h
#define ExprDeps_h

#include <set>

#include <expression/ExpressionID.h>

namespace Expr{

class Tag;                 // forward declaration


/**
 *  @class  ExprDeps
 *  @author James C. Sutherland
 *  @date   March, 2008
 *
 *  @brief Provides functionality for an Expression to record its
 *  dependencies.
 */
class ExprDeps
{
  typedef std::set<Tag> ExprSet;

public:

  ExprDeps(){}

  void requires_expression( const Tag& label ){ deps_.insert( label ); }
  void requires_expression( const TagList& labels )
  {
    for( TagList::const_iterator itl=labels.begin(); itl!=labels.end(); ++itl ){
      deps_.insert( *itl );
    }
  }

  typedef ExprSet::iterator iterator;
  typedef ExprSet::const_iterator const_iterator;

  iterator begin(){ return deps_.begin(); }
  iterator   end(){ return deps_.end(); }

  const_iterator begin() const{ return deps_.begin(); }
  const_iterator   end() const{ return deps_.end(); }

private:

  ExprSet deps_;
};


} // namespace Expr

#endif
