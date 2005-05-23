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
#include <Core/CCA/tools/sidl/Spec.h>
#include <Core/CCA/tools/sidl/parser.h>
#include <string.h>
int lineno;
char* curfile;
#define YY_DECL extern "C" int yylex (void)
%}

integer [0-9]+

/* Babel extension for overloading methods in languages that do not support method overloading. Ignore for now.  */
/* babel_ext [\[][A-Za-z][A-Za-z0-9_]*[\[] */
babel_ext [A-Za-z][A-Za-z0-9_]*

%%

"abstract"	return ABSTRACT;
"array"		return ARRAY;
"bool"		return BOOL;
"char"		return CHAR;
"class"		return CLASS;
"collective"    return COLLECTIVE;
"copy"          return COPY;
"dcomplex"	return DCOMPLEX;
"distribution"  return DISTRIBUTION;
"double"	return DOUBLE;
"enum"          return ENUM;
"extends"	return EXTENDS;
"fcomplex"	return FCOMPLEX;
"final"		return FINAL;
"float"		return FLOAT;
"implements-all" return IMPLEMENTSALL;
"implements"	return IMPLEMENTS;
"import"        return IMPORT;
"in"		return IN;
"inout"		return INOUT;
"int"		return INT;
"interface"	return INTERFACE;
"local"         return LOCAL;
"long"          return LONG;
"oneway"        return ONEWAY;
"opaque"        return OPAQUE;
"out"		return OUT;
"package"	return PACKAGE;
"static"	return STATIC;
"string"	return STRING;
"throws"	return THROWS;
"version"       return VERSION;
"void"		return VOID;

("["){babel_ext}("]") {
                    fprintf(stderr, "Match %s\n", yytext);
                    return BABEL_EXT;
                  }

-?{integer} { yylval.number = strtol(yytext, 0, 0);
	 return INTEGER;
       }

{integer}("."{integer})+ {
         yylval.string = strdup(yytext);
         return VERSION_STRING;
       }

[A-Za-z][A-Za-z0-9_]* { yylval.ident = strdup(yytext);
		        return IDENTIFIER;
		      }


#[\n\t ]*[0-9]+[\n\t ]*\"[^\"]*\"[^\n]*\n {
		     char * p;
                     lineno=strtol(yytext+1, &p, 0);
		     while(*p != 0 && *p == '"' || *p == ' ' || *p == '\t' || *p == '\n'){
		         p++;
		     }
		     curfile=strdup(p);
		     p=curfile+strlen(curfile)-1;
		     while(p >= curfile && *p != '"'){
			 *p--=0;
		     }
		     while(p >= curfile && *p == '"' || *p == ' ' || *p == '\t' || *p == '\n'){
		         *p--=0;
	             }
		     }
\/\/[^\n]*\n {
         lineno++;
     }


" "  { }
"\t" { }
"\n" { lineno++; }
";"  { return ';'; }
"{"  { return '{'; }
"}"  { return '}'; }
"."  { return '.'; }
","  { return ','; }
"("  { return '('; }
")"  { return ')'; }
"<"  { return '<'; }
">"  { return '>'; }
"="  { return '='; }

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
