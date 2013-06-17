/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include "Core/Util/StackTracer.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

#include <execinfo.h>
#include <cxxabi.h>

#include <sci_defs/mpi_defs.h>

using namespace std;

map<string, bool> variables;

void setVariable(const char *name, bool value){
    variables[name] = value;
}

bool getVariable(const char *name){
    string s = name;
    if (variables.find(s) == variables.end())
        return false;
    return variables[s];
}

void printStacktraceIfTrue(const char *fileName, const char *lastNode, const char *conditionalVariable){
    if (variables.find(conditionalVariable) == variables.end())
        return;
    printStacktrace(fileName, lastNode, 2);
}

void printStacktrace(const char *fileName, const char *lastNode, int skipInStackTop)
{
    FILE *pFile = fopen(fileName, "a");
    //FILE *pFile = stdout;
    fprintf(pFile, "stack_trace: %lf\n", MPI_Wtime());

    // storage array for stack trace address data
    void* addrlist[1028];

    // retrieve current stack addresses
    int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

    if (addrlen == 0) {
       fprintf(pFile, "  <Stack trace empty, possibly corrupt>\n");
       return;
    }

    // resolve addresses into strings containing "filename(function+address)",
    // this array must be free()-ed
    char** symbollist = backtrace_symbols(addrlist, addrlen);

    // allocate string which will be filled with the demangled function name
    size_t funcnamesize = 256;
    char* funcname = (char*)malloc(funcnamesize);

    // iterate over the returned symbol lines. skip the first, it is the
    // address of this function.
    for (int i = skipInStackTop; i < addrlen; i++)
    {
       char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

       // find parentheses and +address offset surrounding the mangled name:
       // ./module(function+0x15c) [0x8048a6d]
       for (char *p = symbollist[i]; *p; ++p)
       {
           if (*p == '(')
               begin_name = p;
           else if (*p == '+')
               begin_offset = p;
           else if (*p == ')' && begin_offset) {
               end_offset = p;
               break;
           }
       }

       if (begin_name && begin_offset && end_offset
           && begin_name < begin_offset)
       {
           *begin_name++ = '\0';
           *begin_offset++ = '\0';
           *end_offset = '\0';

           // mangled name is now in [begin_name, begin_offset) and caller
           // offset in [begin_offset, end_offset). now apply
           // __cxa_demangle():

           int status;
           char* ret = abi::__cxa_demangle(begin_name,
                                           funcname, &funcnamesize, &status);
           if (status == 0) {
               funcname = ret; // use possibly realloc()-ed string
               //fprintf(out, "  %s : %s+%s\n", symbollist[i], funcname, begin_offset);
               if (i == 1)
                   fprintf(pFile, "%s+%s\n", funcname, lastNode);
               else
                   fprintf(pFile, "%s+%s\n", funcname, begin_offset);
           }
           else {
               // demangling failed. Output function name as a C function withno arguments.
               //fprintf(out, "  *%s : %s()+%s\n", symbollist[i], begin_name, begin_offset);
               if (i == 1)
                   fprintf(pFile, "%s()+%s\n", begin_name, lastNode);
               else
                   fprintf(pFile, "%s()+%s\n", begin_name, begin_offset);
           }
       }
       else
       {
           // couldn't parse the line? print the whole line.
           if (i == 1)
               fprintf(pFile, "%s+%s\n", symbollist[i], lastNode);
           else
               fprintf(pFile, "%s\n", symbollist[i]);
       }
    }

    fclose(pFile);
    free(funcname);
    free(symbollist);
}
