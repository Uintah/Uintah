/**
 * \file ExpressionID.h
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
#ifndef ExpressionID_h
#define ExpressionID_h

#include <iostream> // for output of expression ids


namespace Expr{


  //==================================================================

  /**
   *  @class  ExpressionID
   *  @author James C. Sutherland
   *  @date   May, 2007
   *
   *  @brief Provides unique identification tags for expressions.
   *
   *  It is critical to have unique identifiers for all expressions,
   *  regardless of type.  Each call to the default constructor
   *  provides a unique identifier.
   *
   *  Note that copy construction and assignment are allowed.
   */
  class ExpressionID
  {
  public:

    /**
     *  Generates a unique identifier.  This is the preferred
     *  constructor.
     */
    inline ExpressionID()
      : id_( get_id() )
    {}

    /**
     *  Copy constructor.
     */
    inline ExpressionID( const ExpressionID & id )
      : id_(id.id_)
    {}

    inline ExpressionID& operator=(const ExpressionID& src)
    {
      id_ = src.id_;
      return *this;
    }

    /**
     *  Obtain a null (invalid) expression id.  Use this for temporary
     *  expressions or for expressions that must be built prior to
     *  obtaining a valid id.
     */
    static ExpressionID null_id(){ return ExpressionID(-1); }


    /** @name comparison operators */
    //@{

    inline bool operator==(const ExpressionID &rhs) const { return id_==rhs.id_; }
    inline bool operator< (const ExpressionID &rhs) const { return id_< rhs.id_; }
    inline bool operator> (const ExpressionID &rhs) const { return id_> rhs.id_; }
    inline bool operator!=(const ExpressionID &rhs) const { return id_!=rhs.id_; }

    //}@

    /** write the expression id to the stream */
    void put(std::ostream& os) const { os << id_; }

    /**
     *  Generate an expression from the requested integer.  Note that
     *  this can be dangerous because it does not guarantee uniqueness
     *  in any way!
     */
    inline explicit ExpressionID( const int id )
      : id_(id)
    {}

    /** @return the value */
    int value() const{return id_;}


  private:

    /** obtain a new unique id. */
    static int get_id(){static int count=0;  return count++;}

  private:
    int id_;
  };


  inline std::ostream& operator<<( std::ostream&os, const ExpressionID& id )
  {
    id.put(os);
    return os;
  }


  //==================================================================


} // namespace Expr

#endif
