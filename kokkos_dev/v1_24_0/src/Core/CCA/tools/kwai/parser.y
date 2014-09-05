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

#include <stdio.h>
extern "C" {
extern int yylex(void);
int yyerror(char*);
}

extern char* curfile;
extern int lineno;

#define YYDEBUG 1
#include <stdlib.h>
#include <string.h>
#include <string>
#include <Core/CCA/tools/kwai/symTable.h>
using namespace std;
using namespace SCIRun;

symTable* symT; 

%}

%union {
  char* ident;
  char* string;
  int number;

  char* text;
  kPackage* kPkg; 
  kObject* kObj;
  kObjList* kObj_list;
  kMethodList* kMethod_list;
  kMethod* kMeth;
  kArgumentList* kArg_list;
  kArgument* kArg;
  bool isP;
};

%start specification

%token ABSTRACT
%token ARRAY
%token BOOL
%token CHAR
%token CLASS
%token COLLECTIVE
%token COPY
%token DCOMPLEX
%token DISTRIBUTION
%token DOUBLE
%token ENUM
%token EXTENDS
%token FCOMPLEX
%token FINAL
%token FLOAT
%token IMPLEMENTS
%token IMPLEMENTSALL
%token IMPORT
%token IN
%token INOUT
%token INT
%token INTERFACE
%token LOCAL
%token LONG
%token ONEWAY
%token OPAQUE
%token OUT
%token PACKAGE
%token STATIC
%token STRING
%token THROWS
%token VERSION
%token VOID
%token<ident> IDENTIFIER
%token<number> INTEGER
%token<string> VERSION_STRING

%type<text> mode type return_type scoped_identifier dot_identifier_star comma_scoped_identifier_star opt_comma_number
%type<kPkg> package
%type<kObj> class interface definition
%type<kObj_list> definition_star 
%type<kMethod_list> statements statements_star class_statements class_statement_star 
%type<kMeth> method class_method
%type<kArg_list> arguments comma_argument_star
%type<kArg> argument
%type<isP> interface_extends class_inherit class_extends class_implements_all class_implements 
%%

specification: version_star import_star package_star
               {
	       }
	       ;

version_star: /* Empty */
	      {
	      }
	      |
	      version_star version
	      {
	      }
	      ;

version: VERSION IDENTIFIER INTEGER ';'
         {
         }
         |
         VERSION IDENTIFIER VERSION_STRING ';'
         {
         }
         ;

package_star: /* Empty */
              {
                symT = new symTable();
              }
              |
	      package_star package
              {
                symT->addPackage($2);  
	      }
              ;

definition_star: /* Empty */
	       {
                 $$ = new kObjList();
	       }
               |
 	       definition_star definition
	       {
                 $$ = $1; 
                 if($2) $1->addObj($2);                 
               }
	       ;

import_star: /* Empty */
             {
	     }
             |
	     import_star import
             {
	     }
             ;

import: IMPORT scoped_identifier ';'
        {
	}
        ;

definition:    class
	       {
                 $$ = $1;
	       }
	       |
	       enum
	       {
                 $$ = NULL;
	       }
	       |
	       interface
	       {
                 $$ = $1;
	       }
	       |
               package
	       {
                 $$ = $1;
               }
               |
	       /* Not in SSIDL spec but added for MxN soln. purposes */
	       distribution
               {
                 $$ = NULL;
               }
	       ;

distribution:  DISTRIBUTION ARRAY IDENTIFIER '<' type opt_comma_number '>' ';'
               {
               }
               ;

package:       PACKAGE scoped_identifier '{' definition_star '}' opt_semi
	       {
                 $$ = new kPackage($2, $4);
	       }
               ;

scoped_identifier:   opt_dot IDENTIFIER dot_identifier_star
	       {
                 $$ = new char[300];
                 $$ = strcat($$,$2);
                 $$ = strcat($$,$3);
	       }
	       ;

opt_dot:       /* Empty */
	       {
	       }
	       |
	       '.'
	       {
	       }
	       ;

opt_semi:      /* Empty */
	       |
	       ';'
               ;

dot_identifier_star: /* Empty */
		     {
                       $$ = new char[200];
	             }
		     |
		     dot_identifier_star '.' IDENTIFIER
		     {
                       strcat($1,"::");  
                       $$ = strcat($1,$3); 
		     }
		     ;


class:	       class_modifier CLASS IDENTIFIER class_inherit class_statements opt_semi
	       {
                 if($4) { //only if extends port base class
                   $$ = new kPort($3, $5);
                 } else {
                   delete $5;
                   $$ = NULL;
                 }
	       }
	       ;

class_modifier: /* Empty */
                {	
		}
                |
                ABSTRACT
                {
		}
		;

class_inherit: class_extends class_implements_all class_implements
	       {
                 $$=($1||$2||$3);
	       }
	       ;

class_extends: /* Empty */
	       {
                 $$=false;
	       }
	       |
	       EXTENDS scoped_identifier
	       {
                 $$=symT->isPort($2);
	       }
	       ;

class_implements_all: /* Empty */
		      {
                        $$=false;
		      }
		      |
		      IMPLEMENTSALL scoped_identifier comma_scoped_identifier_star
                      {
                        $$=(symT->isPort($2) || symT->isPort($3));
		      }
                      ;

class_implements: /* Empty */
		  {
                    $$=false;
		  }
		  |
		  IMPLEMENTS scoped_identifier comma_scoped_identifier_star
		  {
                    $$=(symT->isPort($2) || symT->isPort($3)); 
		  }
		  ;

comma_scoped_identifier_star: /* Empty */
			{
                          $$ = new char[1000];
			}
			|
			comma_scoped_identifier_star ',' scoped_identifier
			{
                          $$ = strcat($$,$1);
                          $$ = strcat($$,$3);
			}
			;

enum: ENUM IDENTIFIER '{' enumerator_list opt_comma '}' opt_semi
      {
      }
      ;

opt_comma: /* Empty */
           |
           ','
           ;

enumerator_list: enumerator
                 {
		 }
                 |
                 enumerator_list ',' enumerator
                 {
		 }
                 ;

enumerator: IDENTIFIER
            {
	    }
            |
            IDENTIFIER '=' INTEGER
            {
	    }
            ;

interface:     INTERFACE IDENTIFIER interface_extends statements opt_semi
	       {
                 if($3) { //only if extends port base class 
                   $$ = new kPort($2, $4);
                 } else {
                   delete $4;
                   $$ = NULL;
                 }
	       }
	       ;


class_statements: '{' class_statement_star '}'
		  {
                    $$ = $2;
		  }
		  ;

class_statement_star: /* Empty */
		   {
                     $$ = new kMethodList();
		   }
		   |
		   class_statement_star class_method
		   {
                     $$ = $1;
                     $1->addMethod($2); 
		   }
                   |
                   /* Not in SSIDL spec but added for MxN soln. purposes */
                   class_statement_star distribution
                   {
                     $$ = $1;
                   }
                   ;

interface_extends: /* Empty */
		   {
                     $$=false; 
		   }
		   |
		   EXTENDS scoped_identifier comma_scoped_identifier_star
		   {
                     $$=(symT->isPort($2) || symT->isPort($3));
		   }
		   ;

class_method: method_modifier method
              {
                $$ = $2;
	      }
              ;

method_modifier: /* Empty */
                 {
		 }
                 |
		 ABSTRACT
		 {
		 }
		 |
		 FINAL
		 {
		 }
		 |
		 STATIC
		 {
		 }
		 ;

statements: '{' statements_star '}'
	    {
              $$ = $2;
	    }
	    ;

statements_star: /* Empty */
	         {
                   $$ = new kMethodList();
                 }
	         |
	         statements_star method
	         {
                   $$ = $1;
                   $1->addMethod($2);
                 }
                 |
		 /* Not in SSIDL spec but added for MxN soln. purposes */
                 statements_star distribution
                 {
                   $$ = $1;
                 }
	         ; 

method: return_type IDENTIFIER arguments method_modifiers2 opt_throws_clause ';'
	{
	  $$ = new kMethod($2,$1,$3); 
	}
        |
        COLLECTIVE return_type IDENTIFIER arguments method_modifiers2 opt_throws_clause ';'
	{
          $$ = new kMethod($3,$2,$4);
	}
	;
      

method_modifiers2: /* Empty */
                  {
		  }
                  |
                  LOCAL
                  {
		  }
		  |
		  ONEWAY
                  {
		  }
		  ;

opt_throws_clause: /* Empty */
		   {
		   }
		   |
		   throws_clause
		   {
		   }
		   ;

return_type: VOID
	     {
               $$ = "void";
             }
	     |
	     opt_copy type
	     {
               $$ = $2;
	     }
	     ;

arguments: '(' ')'
	   {
            $$ = new kArgumentList();
	   }
	   |
	   '(' argument comma_argument_star ')'
	   {
            $$ = $3;
            $3->addArg($2);
	   }
	   ;

opt_copy: /* Empty */
          {
	  }
          |
	  COPY
          {
	  }
          ;

argument: opt_copy mode type
	  {
            $$ = new kArgument($2,$3," ");
	  }
	  |
	  opt_copy mode type IDENTIFIER
	  {
            $$ = new kArgument($2,$3,$4);
	  }
	  ;

mode: IN
      {
       $$ = "in";
      }
      |
      OUT
      {
       $$ = "out";
      }
      |
      INOUT
      {
       $$ = "inout";
      }
      ;

comma_argument_star: /* Empty */
		     {
                      $$ = new kArgumentList();  
	             }
		     |
		     comma_argument_star ',' argument
 		     {
                      $$ = $1;
                      $1->addArg($3);
		     }
		     ;

throws_clause: THROWS scoped_identifier comma_scoped_identifier_star
	       {
	       }
	       ;

type: BOOL
      {
       $$ = "bool";
      }
      |
      CHAR
      {
       $$ = "char";
      }
      |
      DCOMPLEX
      {
       $$ = "dcomplex";
      }
      |
      DOUBLE
      {
       $$ = "double";
      }
      |
      FCOMPLEX
      {
       $$ = "fcomplex";
      }
      |
      FLOAT
      {
       $$ = "float";
      }
      |
      INT
      {
       $$ = "int";
      }
      |
      LONG
      {
       $$ = "long";
      }
      |
      OPAQUE
      {
       $$ = "opaque";
      }
      |
      STRING
      {
       $$ = "string";
      }
      |
      ARRAY '<' type opt_comma_number '>'
      {
       $$ = new char[300];
       $$ = strcat($$,"array<");
       $$ = strcat($$,$3);
       $$ = strcat($$,$4);
       $$ = strcat($$,">");
      }
      |
      scoped_identifier
      {
       $$ = $1;
      }
      ;

opt_comma_number: /* Empty */
		  {
                   $$ = "";
		  }
		  |
		  ',' INTEGER
		  {
                   $$ = new char[20];
                   $$ = strcat($$,",");
                   char i[2];
                   i[0] = $2+48; i[1] = '\0';
                   $$ = strcat($$,i);
		  }
		  ;

%%

int yyerror(char* s)
{
  extern int lineno;
  extern char* curfile;
  fprintf(stderr, "%s: %s at line %d\n" , curfile, s, lineno);
  return 0;
}

