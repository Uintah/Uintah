
/*
//=======================
//
// fnparser.y
// David Hart
// July 2000
// SCI group
// University of Utah
//
//=======================
*/

%{

#include <iostream>
#include <string.h>
#include <strings.h>
#include <stdlib.h>
  
#include <vector>
using namespace std;

#include <SCICore/Math/function.h>
using namespace SCICore::Math;


#define YYPARSE_PARAM param

int yylex();			// Defined in the scanner

namespace SCICore {
namespace Math {
  
void yyerror(char *s);		// Defined below and called automatically when
				// bison detects a parse error

extern int linenumber;		// the current line number
extern int waserror;		// true if an error ever happens

extern int labelctr;
extern char *func_name;

int new_err = 0;		// set to the line number of the error
				// if there is currently an unreported
				// one, set to zero otherwise

int last_err = 0;		// set to the line number of the last
				// reported error

int errline;
extern char *yytext;

} //namespace Math {
} //namespace SCICore {

%}

%union {
  double	value;
  char*		text;
  int		var;
  
  Function*	function;
}

%token <text> TSTRING
%token <value> TCONST

%token <var> TX
%token <var> TXP

%token TCOMMA
%token TLPAREN
%token TRPAREN
%token TBAR
%token TCARET
%token TASSIGN
%token TSEMI

%token TSIN
%token TCOS
%token TEXP
%token TABS
%token TSQR
%token TSQRT
%token TLOG
%token TPOW

%left TPLUS TMINUS
%left TTIMES TSLASH
%left NEG
%right TCARET

%type <function> function

%start session

%%
  
session: function {
  *(Function**)param = $1;
}
  
function : TCONST {
  $$ = new Function($1);
}
| TX {
  $$ = new Function($1);
}
| TLPAREN function TRPAREN {
  $$ = $2;
}

| function TPLUS function {
  $$ = new Function(sum, $1, $3);
}
| function TMINUS function {
  $$ = new Function(difference, $1, $3);
}
| function TTIMES function {
  $$ = new Function(product, $1, $3);
}
| function TSLASH function {
  $$ = new Function(quotient, $1, $3);
}

| function TCARET function {
  $$ = new Function(power, $1, $3);
}
| TPOW TLPAREN function TCOMMA function TRPAREN {
  $$ = new Function(power, $3, $5);
}

| TMINUS function %prec NEG {
  $$ = new Function(negation, $2);
}
| TSIN TLPAREN function TRPAREN {
  $$ = new Function(sine, $3);
}
| TCOS TLPAREN function TRPAREN {
  $$ = new Function(cosine, $3);
}
| TEXP TLPAREN function TRPAREN {
  $$ = new Function(exponential, $3);
}
| TLOG TLPAREN function TRPAREN {
  $$ = new Function(logarithm, $3);
}
| TSQR TLPAREN function TRPAREN {
  $$ = new Function(square, $3);
}
| TSQRT TLPAREN function TRPAREN {
  $$ = new Function(squareroot, $3);
}

| TABS TLPAREN function TRPAREN {
  $$ = new Function(absolutevalue, $3);
}
| TBAR function TBAR {
  $$ = new Function(absolutevalue, $2);
}
;


%%

namespace SCICore {
namespace Math {

//----------------------------------------------------------------------
void fnerror (char *s) {
  cerr << "error";
  if (s) cerr << ": " << s;
  cerr << endl;
				// only report one error per line
  if (last_err != linenumber) {
    last_err = new_err = linenumber;
    waserror = 1;
  }
}

} //namespace Math {
} //namespace SCICore {
