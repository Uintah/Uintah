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
// scanner.l
// David Hart
// Copyright 1998
// Program of Computer Graphics
// Cornell University
// July 2000
// SCI group, University of Utah
// All rights reserved
//=======================
*/

%{

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <ctype.h>

#include <iostream>
#include <vector>
using namespace std;

#include <Core/Math/function.h>
using namespace SCIRun;
#include <Core/Math/fnparser.h>

/* for some reason yywrap gets broken by the fn prefix
   this seems to solve the problem */
//#undef yywrap

int yywrap() { return 1;}
#define YY_SKIP_YYWRAP

namespace SCIRun {

int linenumber = 1;
int waserror = 0;

char*	getId(int lowerit);	// Helping function to obtain string
				// lexemes.
 
int	getNumber();		// Helping function to obtain number
				// lexemes.

} //namespace SCIRun

extern int fnparse(void* param);

%}

 /* Definitions of useful regular expressions. */

blank		[ \t]
alpha		[A-Za-z]
digit		[0-9]
alnum		{alpha}|{digit}
stringchar	[^\"\n\\]
commentchar	[^\n]
comment		\/\/{commentchar}*
integer		{digit}+
number		{integer}+("."{integer}+)?
string		{alpha}{alnum}+

%%

{comment} {
  // do nothing for a comment
}

","	{ return(TCOMMA);	}
";"	{ return(TSEMI);	}
"="	{ return(TASSIGN);	}
"+"	{ return(TPLUS);	}
"-"	{ return(TMINUS);	}
"*"	{ return(TTIMES);	}
"/"	{ return(TSLASH);	}
"("	{ return(TLPAREN);	}
")"	{ return(TRPAREN);	}
"^"	{ return(TCARET);	}
"|"	{ return(TBAR);		}
"sin"	{ return(TSIN);		}
"cos"	{ return(TCOS);		}
"exp"	{ return(TEXP);		}
"sqrt"	{ return(TSQRT);	}
"sqr"	{ return(TSQR);		}
"ln"	{ return(TLOG);		}
"log"	{ return(TLOG);		}
"abs"	{ return(TABS);		}
"fabs"	{ return(TABS);		}
"pow"	{ return(TPOW);		}
"random" { return(TRANDOM);      }

"true"	{ fnlval.value = 1; return(TCONST); }
"false" { fnlval.value = 0; return(TCONST); }

"x"{integer} {
				// extract the integer from yytext
  fnlval.var = atoi(&yytext[1]);
  return(TX);
}

"x"{integer}"'" {
				// extract the integer from yytext
  yytext[yyleng-1] = 0;
  fnlval.var = atoi(&yytext[1]);
  return(TXP);
}

"x["{integer}"]"  {
				// extract the integer from yytext
  yytext[yyleng-1] = 0;
  fnlval.var = atoi(&yytext[2]);
  return(TX);
}

"x["{integer}"]'" {
				// extract the integer from yytext
  yytext[yyleng-2] = 0;
  fnlval.var = atoi(&yytext[2]);
  return(TXP);
}

"x"  { fnlval.var = 0; return(TX);  }
"x'" { fnlval.var = 0; return(TXP); }

"y"  { fnlval.var = 1; return(TX);  }
"y'" { fnlval.var = 1; return(TXP); }

"z"  { fnlval.var = 2; return(TX);  }
"z'" { fnlval.var = 2; return(TXP); }

{number} {
				// check to see if a number is too
				// large.  seems like if the number
				// doesn't print to the same string it
				// originated from, then the number
				// must be too large.
  char*	num = getId(0);
  fnlval.value = atof(num);
  delete [] num;
  return TCONST;
}

{string} {
    fnlval.text = getId(1);
    return(TSTRING);
}



[\n]         { ++linenumber; }


{blank}+ {
  // skip whitespace
}

. {
  fprintf(stderr, "Error, line %d:illegal character '%c'\n",
	  linenumber, yytext[0]);
  waserror = 1;
}

%%

namespace SCIRun {

//----------------------------------------------------------------------
				// This procedure returns a pointer to
				// the string containing the lexeme of
				// the current token.  The beginning
				// of the lexeme is pointed to by
				// "yytext," and it contains "yyleng"
				// characters.
char* getId(int lowerit) {
  char* str = new char[yyleng+1];
  int i;
  strncpy(str, yytext, yyleng);
  str[yyleng] = '\0';
  if (lowerit) {
    for (i = 0; i < yyleng; i++) {
      str[i] = tolower(str[i]);
    }
  }
  return(str);
}

//----------------------------------------------------------------------
				// This procedure returns the integer
				// value of the lexeme of the current
				// token, which should be a numeral.
int getNumber() {
  return(atoi(yytext));
}

//----------------------------------------------------------------------
void fnparsestring(const char* str, Function** param) {
  fn_scan_string(str);		// point the parse buffer to str
  fnparse((void*)param);	// parse it
  fn_delete_buffer(YY_CURRENT_BUFFER); // done with the buffer
				// go back to parsing stdin
  fn_switch_to_buffer(fn_create_buffer(stdin,YY_BUF_SIZE));
}

//----------------------------------------------------------------------
void fnparsestdin(Function** param) {
  fnparse((void*)param);
}

} // End namespace SCIRun
