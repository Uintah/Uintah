#include <UI/GuiArgs.h>
#include <UI/MiscUI.h>

#include <tcl.h>

namespace SemotusVisum {

GuiArgs::GuiArgs(int argc, char* argv[])
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

string GuiArgs::operator[](int i)
{
    return args_[i];
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
    char** argv=new char*[items.size()];
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

}
