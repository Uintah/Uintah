%{
#include "Spec.h"
#include "parser.h"
#include <string.h>
int lineno;
char* curfile;
#define YY_DECL extern "C" int yylex (void);
%}

%%

"package"	return PACKAGE;
"class"		return CLASS;
"extends"	return EXTENDS;
"implements"	return IMPLEMENTS;
"interface"	return INTERFACE;
"abstract"	return ABSTRACT;
"final"		return FINAL;
"static"	return STATIC;
"void"		return VOID;
"throws"	return THROWS;
"bool"		return BOOL;
"char"		return CHAR;
"dcomplex"	return DCOMPLEX;
"double"	return DOUBLE;
"fcomplex"	return FCOMPLEX;
"float"		return FLOAT;
"int"		return INT;
"string"	return STRING;
"array"		return ARRAY;
"in"		return IN;
"out"		return OUT;
"inout"		return INOUT;

[0-9]+ { yylval.number = strtol(yytext, 0, 0);
	 return NUMBER;
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

. { fprintf(stderr, "Illegal character: %c (%d)\n", yytext[0],
            yytext[0]); 
	}
