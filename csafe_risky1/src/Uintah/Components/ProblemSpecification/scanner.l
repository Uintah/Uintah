%{
#include "parser.h"
#include <stdlib.h>
#include <string.h>
int lineno;
char* curfile;
extern "C" int yyerror(char*);
#define YY_DECL extern "C" int yylex (void)
%}

alpha [A-Za-z]
digit [0-9]
special [_]
slash [\\]
quote ["]
id ({alpha}|{special})({alpha}|{special}|{digit})*
number {digit}+
substring [^\n\"]*
escquote {slash}{quote}
string ({substring}|{escquote})*
expon ([eEdD][\+\-]?{digit}+)
%%

"variable"	return VARIABLE;
"FRONT"	return FRONT;
"BACK"	return BACK;
"TOP"	return TOP;
"BOTTOM"	return BOTTOM;
"LEFT"	return LEFT;
"RIGHT"	return RIGHT;

0[0-7]+ {
            return INT;
        }
{number}    return INT;
((\.?{number})|({number}\.{digit}*)){expon}? {
            return FLOAT;
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
"/*"        {
                 int c;
   
		 for ( ; ; ) {
		     while ( (c = yyinput()) != '*' &&  c != EOF )
#ifdef FLEX_SCANNER
                         if(c=='\n')
			     yylineno++;
#else
		         ;    /* eat up text of comment */
#endif

                     if ( c == '*' ) {
			 while ( (c = yyinput()) == '*' )
#ifdef FLEX_SCANNER
                         if(c=='\n')
			     yylineno++;
#else
			     ;
#endif
			 if ( c == '/' )
			     break;    /* found the end */
		     }
		 
		     if ( c == EOF ) {
			 yyerror( "EOF in comment" );
			 break;
		     }
		 }
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
"["  { return '['; }
"]"  { return ']'; }
"="  { return '='; }
"+"  { return '+'; }
"-"  { return '-'; }
"*"  { return '*'; }
"/"  { return '/'; }
"&"  { return '&'; }

\"{string}\"  {
                    yylval.str=strdup(yytext);
                    return STRING;
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

//
// $Log$
// Revision 1.1  2000/03/01 18:33:48  sparker
// New Uintah Problem Specification parser
//
//
