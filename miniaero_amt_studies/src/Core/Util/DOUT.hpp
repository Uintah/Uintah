/*
 * The MIT License
 *
 * Copyright (c) 1997-2016 The University of Utah
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

#ifndef UINTAH_CORE_UTIL_DOUT_HPP
#define UINTAH_CORE_UTIL_DOUT_HPP

#include <cstdio>
#include <cstring>
#include <cstdlib>

#include <string>
#include <sstream>

#define DOUT( cond, ... )           \
  if (cond) {                       \
    std::ostringstream msg;         \
    msg << __VA_ARGS__;             \
    printf("%s\n",msg.str().c_str()); \
  }

#define DDOUT( cond, ... )          \
  if (cond) {                       \
    std::ostringstream msg;         \
    msg << __FILE__ << ":";         \
    msg << __LINE__ << " : ";       \
    msg << __VA_ARGS__;             \
    printf("%s\n",msg.str().c_str()); \
  }

namespace Uintah {

class Dout
{
public:
  Dout( std::string const & name, bool default_active )
    : m_name{ ',' + name + ':' }
    , m_active{ is_active(default_active) }
  {}

  explicit operator bool() const { return m_active; }

private:
  bool is_active( bool default_active ) const
  {
    const char * sci_debug = std::getenv("SCI_DEBUG");
    const char * name = m_name.c_str();
    size_t n = m_name.size();

    if ( !sci_debug ) return default_active;

    const char * sub =  strstr(sci_debug, name);
    if ( !sub ) {
      sub = strstr( sci_debug, name+1);
      --n;
      // name not found
      if ( !sub || sub != sci_debug ) return default_active;
    }
    return sub[n] == '+';
  }

private:
  const std::string m_name;
  const bool        m_active;

private:
  Dout( const Dout & ) = delete;
  Dout & operator=( const Dout & ) = delete;
  Dout( Dout && ) = delete;
  Dout & operator=( Dout && ) = delete;
};

extern Dout ddbg;
extern Dout ddbgst;
extern Dout dtimeout;
extern Dout dreductionout;
extern Dout dtaskorder;
extern Dout dwaitout;
extern Dout dexecout;
extern Dout dtaskdbg;
extern Dout dtaskLevel_dbg;
extern Dout dmpidbg;

} //namespace Uintah

#endif //UINTAH_CORE_UTIL_DOUT_HPP
