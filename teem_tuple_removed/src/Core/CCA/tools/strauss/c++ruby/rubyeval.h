
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
