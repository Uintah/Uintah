
%{

#include <stdio.h>
extern "C" {
extern int yylex(void);
int yyerror(char*);
}
extern char* curfile;
extern int lineno;
#define YYDEBUG 1
#include <stdlib.h>
#include <iostream>
using std::cerr;
%}

%union {
    char* ident;
    int number;
    char* str;
};

%start specification

%token VARIABLE
%token IDENTIFIER
%token FLOAT
%token INT
%token FRONT BACK TOP BOTTOM LEFT RIGHT

%token<str> STRING
%token '='

%left '-' '+'
%left '*' '/' '%'
/* %left '|' '&' '^'
%left T_ANDAND T_OROR
%right '!'
%right '~'
*/
%left UMINUS

%%

specification: definition_star
	       ;


definition_star: /* empty */
                 |
                 definition_star definition
		 ;

definition: VARIABLE IDENTIFIER '=' expression ';'
            |
            IDENTIFIER '=' expression ';'
            |
            IDENTIFIER '{' definition_star  '}'
            |
            IDENTIFIER IDENTIFIER '{' definition_star '}'
            |
            box_syntax
            ;

box_syntax: IDENTIFIER index_list '{' definition_star '}'
            ;

index_list: '[' face_index ']' '[' region_index ']' '[' matl_index ']'
            |
            '[' region_index ']' '[' matl_index ']'
            ;

face_index: facelist
            ;

region_index: IDENTIFIER
              |
              '*'
              ;

matl_index: IDENTIFIER
            |
            '*'
            ;

facelist:   face
            |
            facelist ',' face
            ;

face: TOP | BOTTOM | FRONT | BACK | LEFT | RIGHT ;


expression: FLOAT
            |
            INT
            |
            STRING
{
  cerr << "found string: " << $1 << '\n';
}

            |
            IDENTIFIER
            |
            vector
            |
            expression '*' expression
            |
            expression '/' expression
            |
            expression '-' expression
            |
            expression '+' expression
            |
            expression '%' expression
            |
            '-' expression %prec UMINUS
            ;

vector: '[' expression ',' expression ',' expression ']'
        ;


%%

int yyerror(char* s)
{
  extern int lineno;
  extern char* curfile;
  fprintf(stderr, "%s: %s at line %d\n" , curfile, s, lineno);
  return 0;
}

//
// $Log$
// Revision 1.1  2000/03/01 18:33:48  sparker
// New Uintah Problem Specification parser
//
//
