
/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


#ifndef SCIRun_Core_GuiInterface_TCLKeysyms
#define SCIRun_Core_GuiInterface_TCLKeysyms 1

namespace SCIRun {

#include <map>
#include <string>
#include <iostream>

typedef map<std::string, wchar_t> TCLKeysym_t;
static TCLKeysym_t tcl_keysym;


  class init_tcl_keysym {
  public:
    //static void init_tcl_keysym () {
    init_tcl_keysym () {
    tcl_keysym["space"] = ' ';
    tcl_keysym["exclam"] = '!';
    tcl_keysym["quotedbl"] = '\"';
    tcl_keysym["numbersign"] = '#';
    tcl_keysym["dollar"] = '$';
    tcl_keysym["percent"] = '%';
    tcl_keysym["ampersand"] = '&';
    tcl_keysym["quoteright"] = '\'';
    tcl_keysym["parenleft"] = '(';
    tcl_keysym["parenright"] = ')';
    tcl_keysym["asterisk"] = '*';
    tcl_keysym["plus"] = '+';
    tcl_keysym["comma"] = ',';
    tcl_keysym["minus"] = '-';
    tcl_keysym["period"] = '.';
    tcl_keysym["slash"] = '/';
    tcl_keysym["colon"] = ':';
    tcl_keysym["semicolon"] = ';';
    tcl_keysym["less"] = '<';
    tcl_keysym["equal"] = '=';
    tcl_keysym["greater"] = '>';
    tcl_keysym["question"] = '\?';
    tcl_keysym["at"] = '@';
    tcl_keysym["bracketleft"] = '[';
    tcl_keysym["backslash"] = '\\';
    tcl_keysym["bracketright"] = ']';
    tcl_keysym["asciicircum"] = '^';
    tcl_keysym["underscore"] = '_';
    tcl_keysym["quoteleft"] = '`';
    tcl_keysym["braceleft"] = '{';
    tcl_keysym["bar"] = '|';
    tcl_keysym["braceright"] = '}';
    tcl_keysym["asciitilde"] = '~';
  }
  };

  static init_tcl_keysym init_tcl_keysymer;

  //  init_tcl_keysym()

}

#endif
