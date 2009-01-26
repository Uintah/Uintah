/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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



#ifndef TOKEN_H
#define TOKEN_H 1

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <sgi_stl_warnings_on.h>



using namespace std;



class Token;
typedef vector<string>        string_list;
typedef string_list::iterator sl_iter;
typedef vector<Token*>        token_list;
typedef token_list::iterator  tl_iter;
typedef map<string,Token*>    st_map;
typedef st_map::iterator      stm_iter;



#define IGNORE_UNKNOWN_TOKENS 1
#define DEBUG 0



typedef enum {

  DEL_NONE        = 0,
  DEL_LBRACE      = '{',
  DEL_RBRACE      = '}',
  DEL_LPAREN      = '(',
  DEL_RPAREN      = ')',
  DEL_LBRACKET    = '[',
  DEL_RBRACKET    = ']',
  DEL_LANGLE      = '<',
  DEL_RANGLE      = '>',
  DEL_DOUBLEQUOTE = '"',
  DEL_SINGLEQUOTE = '\''

} DELIMITER;

    

class Token
{

 protected:

  class TokenManager
  {
      
  private:
    
    st_map all_tokens_;
    
  public:
    
    TokenManager() {}
    virtual ~TokenManager() {}
    
    bool AddToken(Token *t) {
      if (all_tokens_.find(t->GetMoniker()) == all_tokens_.end()) {
	all_tokens_.insert(pair<string,Token*>(t->GetMoniker(),t));
#if 0
	cerr << "Token Manager: added token " << t->GetMoniker() << endl;
#endif
	return true;
      } else 
	return false;
    }
    
    Token *MakeToken(const string& s) {
      stm_iter i = all_tokens_.find(s);
      if (i != all_tokens_.end()) {
	return (*i).second->MakeToken();
      }
      return 0;
    }
  };
  
  static TokenManager token_manager_;
  static unsigned indent_;


  Token          *parent_;
  bool           file_;
  string         moniker_;
  token_list     children_;
  string_list    valid_child_monikers_;
  unsigned       nargs_;
  string_list    args_;

  bool valid_child_moniker(const string &s) {
    for (sl_iter i = valid_child_monikers_.begin();
	 i != valid_child_monikers_.end();
	 ++i) {
      if ((*i) == s) 
	return true;
    }
    return false;
  }
  
 public:

  Token(const string &s) 
    : parent_(0), file_(false), moniker_(s), nargs_(0)
    { token_manager_.AddToken(this); }
  virtual ~Token() {
    destroy_children();
  }

  void destroy_children()
  {
    size_t length = children_.size();
    size_t loop;
    for (loop=0; loop<length; ++loop) {
      children_[loop]->destroy_children();
      delete children_[loop];
    }
    children_.resize(0);
  } 

  /* to avoid stupid (read "s-g-i") warnings */
  virtual bool Parse() { return false; }
  virtual void Write() {}

  virtual bool Parse(ifstream &);
  virtual bool ParseArgs(ifstream &);
  virtual bool ParseChildren(ifstream &);

  virtual void Write(ofstream &);
  virtual void Indent(ofstream &);

  bool AddChildMoniker(const string &s) { 
    if (!valid_child_moniker(s)) {
      valid_child_monikers_.push_back(s);
      return true;
    }
    return false;
  }

  void SetParent(Token *p) {
    parent_ = p;
  }
	
  token_list *GetChildren() { return &children_; }
  size_t GetNumChildren() const { return children_.size(); }
  string GetMoniker() const { return moniker_; }
  string_list *GetArgs() { return &args_; }
  unsigned GetNumArgs() const { return nargs_; }
  static bool match(DELIMITER, DELIMITER);
  virtual Token *MakeToken() = 0;

};


// Local Variables:
// c-basic-offset: 2
// c-indent-level: 2
// End:

#endif

