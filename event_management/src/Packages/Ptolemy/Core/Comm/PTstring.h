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

//this file will include some helper functions that will enable us to 
//process the strings that are sent to the SCIRun server by the
//SPA workflow.
//by oscar barney


#include <Core/Containers/StringUtil.h>

/*  turns input characters into a vector of strings
    and returns the vector. sets size to be the number
    of things that are in the vector.
*/
vector<string> processCstr(char *input, int *size){
	//change the input into a string then vector 
	string temp = string(input);
	vector<string> v = split_string(temp,';');
	if(temp == "\n"){
		*size = 0;	//case where we do not do anything
	}	else {
		*size = (int)v.size();	
		//cut \n off the end of the last string
		v[*size-1] = v[*size-1].substr(0,v[*size-1].size()-1);
	}	
	return v;
}




