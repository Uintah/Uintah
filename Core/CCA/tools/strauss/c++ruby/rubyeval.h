/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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


#ifndef RUBYEVAL_H
#define RUBYEVAL_H

#include <string>
#include <iostream>

//ruby.h is not included, because it conflicts with many other library headers
typedef unsigned long VALUE;


/** Embedded Ruby interpreter for evaluating Ruby expressions.
  *
  * @author Pirmin Kalberer <pka@sourcepole.ch>
  */
class RubyEval {
public:
  /** Singleton Instance */
  static RubyEval* instance();
  ~RubyEval();
  /** Convert Ruby value to string */
  static std::string val2str(const VALUE rval);
  /** Convert Ruby string value to string */
  static std::string strval2str(const VALUE rval);
  /** Run Ruby interpreter with @p filename */
  void run_file(const char* filename, std::ostream& out = std::cout);
  /** Evaluate code string */
  VALUE eval(const char* code);
  /** Evaluate code string and print errors */
  VALUE eval(const char* code, std::ostream& errout);
  /** Get Ruby error/exception info an print it */
  static void exception_print(std::ostream& errout = std::cerr);
  /** Get Ruby error/exception info as string */
  static std::string exception_info();
  /** Last evaluation was successful */
  bool evalOk();
private:
  /** Singleton */
  RubyEval();
  /** singleton instance */
  static RubyEval* _instance;
  /** last eval status */
  int status;
};

#endif
