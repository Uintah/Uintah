/**
 *  \file   ParseGroup.h
 *
 *  \date   May 25, 2012
 *  \author James C. Sutherland
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

#ifndef PARSER_H_
#define PARSER_H_

#include <string>
#include <iosfwd>
#include <vector>
#include <map>

#include <boost/property_tree/ptree.hpp>


class ParseGroupIterator;

/** \addtogroup Parser
 *  @{
 */

/**
 *  \class ParseGroup
 *  \author James C. Sutherland
 *  \date   May 25, 2012
 *
 *  \brief XML Parser for the LBMS code.  Note that this parser does
 *   not currently support DTDs.
 */
class ParseGroup
{
  typedef boost::property_tree::ptree Entry;

  std::string name_;
  const Entry* tree_;
  bool builtIt_;

  static void display( std::ostream&, const unsigned, const Entry& );

  void check_exists( const std::string& key ) const;

public:

  ParseGroup( const std::string name, const Entry* const node );

  typedef ParseGroupIterator const_iterator;

  /**
   * \brief load a file from disk and parse it
   * \param fileName the name of the file to load
   */
  ParseGroup( const std::string fileName );

  /**
   * @brief Build a ParseGroup from a stream containing the XML
   * @param s the stream containing the XML to parse
   */
  ParseGroup( std::istream& s );

  ParseGroup( const ParseGroup& );
  ParseGroup& operator=( const ParseGroup& );

  ~ParseGroup();

  void display( std::ostream& ) const;

  /** \brief query if the given child exists */
  bool has_child( const std::string& key ) const;

  /** \brief obtain the name of this ParseGroup */
  std::string name() const;

  /**
   * @param key the name of the child group to return
   * @return the child ParseGroup
   */
  ParseGroup get_child( const std::string key ) const;

  /** @return the value of this node */
  template<typename T> T get_value() const;

  /**
   * @param key the name of the child
   * @return the value to extract from the child tag
   */
  template<typename T> T get_value( const std::string key ) const;

  /**
   * @param key the name of the child
   * @return the vector of values extracted from the child
   */
  template<typename T> std::vector<T> get_value_vec( const std::string key ) const;

  /**
   * @return the vector of values extracted from this node
   */
  template<typename T> std::vector<T> get_value_vec() const;

  /**
   * @param key the name of the child tag
   * @param t a default value to return if there is no tag with the given name found.
   * @return the value in the tag, or the default value
   */
  template<typename T> T get_value( const std::string key, T t ) const;

  /**
   * @brief obtain the requested attribute of a given XML node
   * @param name the name of the attribute to obtain
   * @return the attribute value
   * @todo need to insert error checking
   */
  template<typename T> T get_attribute( const std::string name ) const;
  template<typename T> T get_attribute( const std::string name, const T ) const;

  /** @brief obtain the iterator for the first entry in this group */
  const_iterator begin( const std::string name ) const;

  /** @brief obtain the end iterator for this group */
  const_iterator end( const std::string name ) const;

  /**
   * @brief obtain the children as a map<name,value>.
   */
  template< typename T > std::map<std::string,T> get_children() const;

  bool operator==( const ParseGroup& );
  const Entry* entry() const{ return tree_; }
};

class ParseGroupIterator
{
public:
  typedef boost::property_tree::ptree Entry;
  typedef Entry::const_assoc_iterator Iter;
  ParseGroupIterator( Iter iter, Iter end );
  ParseGroupIterator( const ParseGroupIterator& other );
  const ParseGroup* operator->() const;
  const ParseGroup& operator*() const;
  bool operator!=( const ParseGroupIterator& other ) const;
  void operator++();
private:
  ParseGroup pg_;
  Iter iter_, iend_;
};

std::ostream& operator<<( std::ostream&, const ParseGroup& );


/** @} */

#endif /* PARSER_H_ */
