/**
 *  \file   PropertyStash.h
 *  \date   April, 2012
 *  \author James C. Sutherland
 *
 * Copyright (c) 2014 The University of Utah
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

#ifndef PropertyStash_h
#define PropertyStash_h

#include <map>
#include <string>

#include <boost/variant.hpp>

#include <tabprops/TabProps.h>


/**
 *  \class  PropertyStash
 *  \author James C. Sutherland
 *  \date   April, 2012
 *
 *  \brief Allows values of various types to be stashed/retrieved by a string key.
 */
class PropertyStash
{
  typedef boost::variant<
      int,
      double,
      std::string
  > types; // supported types

  typedef std::map<std::string,types> Stash;
  Stash stash_;

public:
  PropertyStash(){}
  ~PropertyStash(){}

  /**
   * @brief set a property
   * @param key the name (key) for the property
   * @param t the property to set
   */
  template<typename T>
  void set( const std::string key, const T t );

  /**
   * @brief Retrieve a property
   *
   * @param key the name (key) for the property
   * @return the value of the requested property
   *
   * @tparam the type for the requested property
   *
   * Note that if the requested property does not exist,
   * an exception will be thrown.
   *
   * Example:
   * \code
   *   PropertyStash stash;
   *   Property prop; // ... set property ...
   *   stash.set( "myProperty", prop );
   *   Property prop2 = stash.get<Property>("myProperty");
   * \endcode
   */
  template<typename T>
  T get( const std::string& key ) const;

  template<typename Archive>
  void serialize( Archive& ar, const unsigned int version );
};

#endif /* PropertyStash_h */
