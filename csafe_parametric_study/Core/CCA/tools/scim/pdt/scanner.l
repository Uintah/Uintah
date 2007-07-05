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


%{
#include "../IR.h"
#include "parser.h"
#include <string.h>

int pdt_lineno;
char* pdt_curfile;
int rejectOL = 0; 
#define YY_DECL extern "C" int pdtlex (void)
%}

%%

"%map"           { rejectOL = pdt_lineno; return MAP; }
"%ininterface"   { rejectOL = pdt_lineno; return ININTERFACE; }
"%outinterface"  { rejectOL = pdt_lineno; return OUTINTERFACE; }
"%/ininterface"   { rejectOL = pdt_lineno; return CLOSEININTERFACE; }
"%/outinterface"  { rejectOL = pdt_lineno; return CLOSEOUTINTERFACE; }
"%inoutremap"    { rejectOL = pdt_lineno; return INOUTREMAP; }
"%inoutomit"     { rejectOL = pdt_lineno; return INOUTOMIT; }
"->"             return ARROW;
">>"             { rejectOL = pdt_lineno; return DBLANGLE; }


[A-Za-z][A-Za-z0-9._]* { pdtlval.ident = strdup(yytext);
             	         return IDENTIFIER;
		       }

#[\n\t ]*[0-9]+[\n\t ]*\"[^\"]*\"[^\n]*\n {
                     char * p;
                     pdt_lineno=strtol(yytext+1, &p, 0);
                     while(*p != 0 && *p == '"' || *p == ' ' || *p == '\t' || *p == '\n'){
                         p++;
                     }
                     pdt_curfile=strdup(p);
                     p=pdt_curfile+strlen(pdt_curfile)-1;
                     while(p >= pdt_curfile && *p != '"'){
                         *p--=0;
                     }
                     while(p >= pdt_curfile && *p == '"' || *p == ' ' || *p == '\t' || *p == '\n'){
                         *p--=0;
                     }
                     }

\/\/[^\n]*\n {
         pdt_lineno++;
     }

" "  { }
"\n" { pdt_lineno++; }
">"  { rejectOL = pdt_lineno; return '>'; }

[^ \t\n>][^\n]* { if(pdt_lineno == rejectOL) REJECT;
                    pdtlval.oline = yytext;
                    return OTHERLINE;
                  }

. { fprintf(stderr, "Illegal character: %c (%d)\n", yytext[0],
            yytext[0]);
        }


%%
extern "C" {

int yywrap()
{
  return 1;
}

}




















