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
#include <expression/Context.h>

#include <map>
#include <algorithm>

namespace Expr{

  typedef std::map<std::string,Context> StrContMap;
  StrContMap scmap;

  void set_map()
  {
    if( !scmap.empty() ) return;
    scmap[      "STATE_NONE"    ] = STATE_NONE;
    scmap["EXPR::STATE_NONE"    ] = STATE_NONE;
    scmap[      "STATE_N"       ] = STATE_N;
    scmap["EXPR::STATE_N"       ] = STATE_N;
    scmap[      "STATE_NP1"     ] = STATE_NP1;
    scmap["EXPR::STATE_NP1"     ] = STATE_NP1;
    scmap[      "STATE_DYNAMIC" ] = STATE_DYNAMIC;
    scmap["EXPR::STATE_DYNAMIC" ] = STATE_DYNAMIC;
    scmap[      "CARRY_FORWARD" ] = CARRY_FORWARD;
    scmap["EXPR::CARRY_FORWARD" ] = CARRY_FORWARD;
  }

  Context str2context( std::string s )
  {
    set_map();
    std::transform( s.begin(), s.end(), s.begin(), ::toupper );
    StrContMap::iterator i=scmap.find(s);
    if( i==scmap.end() ){
      return INVALID_CONTEXT;
    }
    return i->second;
  }

  std::string context2str( const Context c )
  {
    std::string s;
    switch( c ){
    case STATE_NONE     : s = "STATE_NONE";	 	break;
    case STATE_N        : s = "STATE_N";	 	break;
    case STATE_NP1      : s = "STATE_NP1";	 	break;
    case STATE_DYNAMIC  : s = "STATE_DYNAMIC";	 	break;
    case CARRY_FORWARD  : s = "CARRY_FORWARD";		break;
    case INVALID_CONTEXT: s = "INVALID_CONTEXT";	break;
    }
    return s;
  }

  std::ostream& operator<<( std::ostream& os, const Context c )
  {
    os << context2str(c);
    return os;
  }

}
