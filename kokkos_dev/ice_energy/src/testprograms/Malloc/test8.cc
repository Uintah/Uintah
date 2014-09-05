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



#ifdef __sgi

#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" int Allocator_try_lock(unsigned long*);

void do_test(void*)
{
    for(int i=0;i<200000;i++){
	void* p=malloc(508);
	free(p);
    }
    fprintf(stderr, "done...\n");
}

main()
{
    sproc(do_test, PR_SADDR, 0);
    sproc(do_test, PR_SADDR, 0);
    sproc(do_test, PR_SADDR, 0);
    sproc(do_test, PR_SADDR, 0);
    int s;
    wait(&s);
    wait(&s);
    wait(&s);
    wait(&s);
    return 0;
}

#else

#include <iostream>
using std::cerr;

int
main(char **, int )
{
    cerr << "This test not available\n";
    return 0;
}

#endif

