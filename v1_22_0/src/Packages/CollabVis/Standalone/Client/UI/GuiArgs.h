#ifndef __gui_args_h_
#define __gui_args_h_

#include <vector>
#include <string>

using namespace std;

namespace SemotusVisum {

class GuiArgs {
  vector<string> args_;
public:
  bool have_error_;
  bool have_result_;
  string string_;
  
  GuiArgs(int argc, char* argv[]);
  ~GuiArgs();
  int count();
  string operator[](int i);
  
  void error(const string&);
  void result(const string&);
  void append_result(const string&);
  void append_element(const string&);
  
  static string make_list(const string&, const string&);
  static string make_list(const string&, const string&, const string&);
  static string make_list(const vector<string>&);
};

}
#endif
