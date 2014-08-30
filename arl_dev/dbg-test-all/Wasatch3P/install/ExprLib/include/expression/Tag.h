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

/**
 *  \file   Tag.h
 *  \author James C. Sutherland
 */

#ifndef Expr_Tag_h
#define Expr_Tag_h

#include <string>

#include <expression/ExprFwd.h>

namespace Expr{

  /**
   *  @class Tag
   *  @author James C. Sutherland
   *
   *  @brief Label for an Expression or field.
   */
  class Tag
  {
    std::string fieldName_; ///< the name of the field
    Context     context_;   ///< the Context
    FieldID     id_;        ///< enables fast comparison of Tag objects

  public:

    /**
     * Construct a Tag
     * @param fieldName the name of the field
     * @param c the Context for the field
     */
    Tag( const std::string fieldName, const Context c );

    Tag( const Tag& s );

    /**
     * Copy an existing tag, ammending its name with the supplied string
     * @param t the Tag to copy
     * @param s the string to append to the Tag name.
     */
    Tag( const Tag& t, const std::string s );

    Tag();

    Tag& operator=( const Tag& );

    /** @brief Obtain the name of the field evaluated by this expression. */
    const std::string& field_name() const{ return fieldName_; }
    const std::string& name() const{ return fieldName_; }
    std::string& name(){ return fieldName_; }

    /**
     * @return a unique identifier for this field
     */
    FieldID id() const{ return id_; }

    /** @brief Obtain the name of the tag for this expression. */
    const Context& context() const{ return context_; }
    Context& context(){ return context_; }

    inline bool operator<( const Tag& rhs ) const{ return (id_<rhs.id_); }

    inline bool operator==( const Tag& rhs ) const{ return ( id_ == rhs.id_ ); }
    inline bool operator!=( const Tag& rhs ) const{ return !(*this==rhs); }

  };

  /**
   * Prepend the given Tag onto the given TagList
   */
  inline TagList tag_list( const Tag tag, TagList& tags ){
    tags.insert( tags.begin(), tag );
    return tags;
  }

  /**
   * Append the given Tag onto the given TagList
   */
  inline TagList tag_list( TagList& tags, const Tag tag ){
    tags.push_back( tag );
    return tags;
  }

  inline TagList tag_list( const Tag tag ){
    TagList tags; tags.push_back(tag);
    return tags;
  }

  inline TagList tag_list( const Tag t1, const Tag t2 ){
    TagList tags = tag_list(t1);
    tags.push_back(t2);
    return tags;
  }

  inline TagList tag_list( const Tag t1, const Tag t2, const Tag t3 ){
    TagList tags = tag_list(t1,t2);
    tags.push_back(t3);
    return tags;
  }

  inline TagList tag_list( const Tag t1, const Tag t2, const Tag t3, const Tag t4 ){
    TagList tags = tag_list(t1,t2,t3);
    tags.push_back(t4);
    return tags;
  }

  inline TagList tag_list( const Tag t1, const Tag t2, const Tag t3, const Tag t4, const Tag t5 ){
    TagList tags = tag_list(t1,t2,t3,t4);
    tags.push_back(t5);
    return tags;
  }

  inline TagList tag_list( const Tag t1, const Tag t2, const Tag t3, const Tag t4, const Tag t5, const Tag t6 ){
    TagList tags = tag_list(t1,t2,t3,t4,t5);
    tags.push_back(t6);
    return tags;
  }

}

#endif
