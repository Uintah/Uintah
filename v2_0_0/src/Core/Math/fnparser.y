/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
//=======================
// fnparser.y
// David Hart
// July 2000
// SCI group
// University of Utah
//=======================
*/

%{

#if !defined(_AIX) && !defined(__APPLE__)
#  include <alloca.h>
#endif
#include <iostream>
#include <string.h>
#include <strings.h>
#include <stdlib.h>
  
#include <vector>
using namespace std;

#include <Core/Math/function.h>
using namespace SCIRun;

#define YYPARSE_PARAM param

int yylex();			// Defined in the scanner

extern "C" void yyerror(char *s);		// Defined below and called automatically when
				// bison detects a parse error

namespace SCIRun {
  
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

} //namespace SCIRun

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
%token TRANDOM

%left TPLUS TMINUS
%left TTIMES TSLASH
%left NEG
%right TCARET

%type <function> function

%start session

%%
  
session: function {
  *(Function**)param = $1;
};
  
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

| TRANDOM TLPAREN TRPAREN {
  $$ = new Function(randomfunction, 0);
}
;


%%

//----------------------------------------------------------------------
extern "C" void fnerror (char *s) {
  cerr << "error";
  if (s) cerr << ": " << s;
  cerr << endl;
				// only report one error per line
  if (last_err != linenumber) {
    last_err = new_err = linenumber;
    waserror = 1;
  }
}
