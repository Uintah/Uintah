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



/*
 *  GuiCallback.cc: Interface to user interface
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/GuiInterface/GuiCallback.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Exceptions/GuiException.h>
#include <Core/Malloc/Allocator.h>
#include <tcl.h>

using namespace SCIRun;

GuiArgs::GuiArgs(int argc, TCLCONST char* argv[])
: args_(argc)
{
    for(int i=0;i<argc;i++)
	args_[i] = string(argv[i]);
    have_error_ = false;
    have_result_ = false;
}

GuiArgs::~GuiArgs()
{
}

int GuiArgs::count()
{
    return args_.size();
}

string
GuiArgs::operator[](int i)
{
  if (i < 0 || i >= int(args_.size()))
    SCI_THROW(GuiException("Cannot access GUI Arg #"+to_string(i)));

  return args_[i];
}


string
GuiArgs::get_string(int i)
{
  return operator[](i);
}

int
GuiArgs::get_int(int i)
{
  int ret_val;
  if (!string_to_int(operator[](i), ret_val))
      SCI_THROW(GuiException
		("Cannot convert GUI Argument #" +to_string(i) +
		 ", contents = '" + string(args_[i]) + "' to type of int."));
          
  return ret_val;
}

double
GuiArgs::get_double(int i)
{
  double ret_val;
  if (!string_to_double(operator[](i), ret_val))
      SCI_THROW(GuiException
		("Cannot convert GUI Argument #" +to_string(i) +
		 ", contents = '" + string(args_[i]) + "' to type of int."));
          
  return ret_val;
}


void GuiArgs::error(const string& e)
{
  string_ = e;
  have_error_ = true;
  have_result_ = true;
}

void GuiArgs::result(const string& r)
{
    if(!have_error_){
	string_ = r;
	have_result_ = true;
    }
}

void GuiArgs::append_result(const string& r)
{
    if(!have_error_){
	string_ += r;
	have_result_ = true;
    }
}

void GuiArgs::append_element(const string& e)
{
    if(!have_error_){
	if(have_result_)
	    string_ += ' ';
	string_ += e;
	have_result_ = true;
    }
}

string GuiArgs::make_list(const string& item1, const string& item2)
{
    char* argv[2];
    argv[0]= ccast_unsafe(item1);
    argv[1]= ccast_unsafe(item2);
    char* ilist=Tcl_Merge(2, argv);
    string res(ilist);
    free(ilist);
    return res;
}

string GuiArgs::make_list(const string& item1, const string& item2,
			const string& item3)
{
    char* argv[3];
    argv[0]=ccast_unsafe(item1);
    argv[1]=ccast_unsafe(item2);
    argv[2]=ccast_unsafe(item3);
    char* ilist=Tcl_Merge(3, argv);
    string res(ilist);
    free(ilist);
    return res;
}

string GuiArgs::make_list(const vector<string>& items)
{
    char** argv=scinew char*[items.size()];
    for(unsigned int i=0; i<items.size(); i++)
    {
      argv[i]= ccast_unsafe(items[i]);
    }
    char* ilist=Tcl_Merge(items.size(), argv);
    string res(ilist);
    free(ilist);
    delete[] argv;
    return res;
}


GuiCallback::GuiCallback()
{
}

GuiCallback::~GuiCallback()
{
}
