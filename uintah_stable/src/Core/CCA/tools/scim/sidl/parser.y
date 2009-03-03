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
#include <string>
#include <Core/CCA/tools/scim/IR.h>
using namespace std;

extern IR* ir; 

%}

%union {
  char* ident;
  char* string;
  int number;

  char* text;
  IrPackage* irPkg; 
  IrDefinition* irDef;
  IrDefList* irDef_list;
  IrMethodList* IrMethod_list;
  IrMethod* irMeth;
  IrArgumentList* irArg_list;
  IrArgument* irArg;
  IrNameMapList* irNM_list;
  IrNameMap* irNM; 
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

%token MAP
%token ININTERFACE
%token OUTINTERFACE
%token CLOSEININTERFACE
%token CLOSEOUTINTERFACE
%token INOUTREMAP
%token INOUTOMIT 
%token ARROW
%token DBLANGLE

%type<text> mode type return_type opt_comma_number
%type<irPkg> package
%type<irDef> class interface definition
%type<irDef_list> definition_star
%type<IrMethod_list> statements statement_star
%type<irMeth> method 
%type<irArg_list> arguments comma_argument_star
%type<irArg> argument
%type<irNM_list> namemap_star subnamemap_star
%type<irNM> namemap subnamemap
%%

specification: mainline_star


mainline_star: /* Empty */
               {
		 //ir = new IR();
	       }
               |
               mainline_star mainline
               {

	       }
               ;


mainline: command
          {
	  }
          |
          version
          {
          }
          |
          definition
          {
            ir->addInOutDef($1);
          }
          ;


command: MAP IDENTIFIER ARROW IDENTIFIER namemap_star
         {
           IrMap* map = new IrMap($2,$4,$5,ir);
           ir->addMap(map);
         }
         |
         MAP namemap_star
         {
           IrMap* map = new IrMap($2,NULL);
           ir->setForAllMap(map);
         }
         |
         ININTERFACE definition_star CLOSEININTERFACE 
         {
           ir->addInDefList($2);
	 }
         |
         OUTINTERFACE definition_star CLOSEOUTINTERFACE
         {
           ir->addOutDefList($2);
         }
         |
         INOUTOMIT
         {
           ir->omitInOut();
         }
         |
         INOUTOMIT IDENTIFIER
         {
           ir->omitInOut($2); 
         }
         |
         INOUTREMAP IDENTIFIER
         {
           ir->remapInOut($2);
         } 
         ;


/*SIDL extra*/
version: VERSION IDENTIFIER INTEGER ';'
         {
         }
         |
         VERSION IDENTIFIER VERSION_STRING ';'
         {
         }
         ;

opt_version: /* Empty */
             {
             }
             |
	     VERSION INTEGER 
             {
             }
             |
             VERSION VERSION_STRING
             {
             }
             ;


definition: class 
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
            ;


package: PACKAGE IDENTIFIER opt_version '{' definition_star '}' opt_semi
         {
           $$ = new IrPackage($2, $5);
         }
         ;


definition_star: /* Empty */
               {
                 $$ = new IrDefList();
               }
               |
               definition_star definition
               {
                 $$ = $1;
                 if($2) $1->addDef($2);
               }
               ;


namemap_star: /* Empty */
              {
                $$ = new IrNameMapList();
              }
              |
              namemap_star namemap
              {
                $$ = $1;
                if($2) $1->addNameMap($2);
              }
              ;


namemap: '>' IDENTIFIER ARROW IDENTIFIER subnamemap_star
         {
           $$ = new IrNameMap($2,$4);
	   if($5) $$->addSubList($5);
         }
         |
         '>' IDENTIFIER subnamemap_star
         {
           $$ = new IrNameMap($2,$2);
           if($3) $$->addSubList($3);
         }
         ;


subnamemap_star: /* Empty */
                 {
                   $$ = new IrNameMapList();
                 }
                 |
                 subnamemap_star subnamemap
                 {
                   $$ = $1;
                   if($2) $1->addNameMap($2);
                 }
                 ;


subnamemap: DBLANGLE IDENTIFIER ARROW IDENTIFIER
            {
              $$ = new IrNameMap($2,$4);
            }
            ;


enum: ENUM IDENTIFIER '{' enumerator_list opt_comma '}' opt_semi
      {
      }
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


interface: INTERFACE IDENTIFIER extends statements opt_semi
       {
	 $$ = new IrPort($2, $4);
       }
       ;


class: modifier CLASS IDENTIFIER extends statements opt_semi
       {
	 $$ = new IrPort($3, $5);
       }
       ;


extends: /* Empty */
         {
         }
         |
         EXTENDS IDENTIFIER 
         {
         }
         | 
         IMPLEMENTSALL IDENTIFIER
         {
         }
         |
         IMPLEMENTS IDENTIFIER 
         {
         }
         ;


modifier: /* Empty */
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
	  |
          /*SIDL extra*/ 
          COLLECTIVE
          {
          }
          ;


statements: '{' statement_star '}'
            {
              $$ = $2;
            }
            ;
            

statement_star: /* Empty */
                {
		  $$ = new IrMethodList();
                }
                |
                statement_star method
                {
		  $$ = $1;
		  $1->addMethod($2);
                }
                ;


method: modifier return_type IDENTIFIER arguments throws_clause ';'
        {
          $$ = new IrMethod($3,$2,$4);
        }
        ;


throws_clause: /* Empty */
               {
               }
               |
               THROWS IDENTIFIER comma_identifier_star
	       {
	       }
	       ;


arguments: '(' ')'
           {
            $$ = new IrArgumentList();
           }
           |
           '(' argument comma_argument_star ')'
           {
            $$ = $3;
            $3->addArgToFront($2);
           }
           ;


comma_argument_star: /* Empty */
                     {
                      $$ = new IrArgumentList();
                     }
                     |
                     comma_argument_star ',' argument
                     {
                      $$ = $1;
                      $1->addArg($3);
                     }
                     ;


argument: mode type
          {
            $$ = new IrArgument($1,$2," ");
          }
          |
          mode type IDENTIFIER
          {
            $$ = new IrArgument($1,$2,$3);
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


return_type: VOID
             {
               $$ = "void";
             }
             |
             type
             {
               $$ = $1;
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
       $$ = strcpy($$,"array<");
       $$ = strcat($$,$3);
       $$ = strcat($$,$4);
       $$ = strcat($$,">");
      }
      |
      IDENTIFIER
      {
       $$ = $1;
      }
      ;


comma_identifier_star: /* Empty */
		       {
		       }
		       |
		       comma_identifier_star ',' IDENTIFIER 
		       {
		       }
		       ;


opt_comma: /* Empty */
           |
           ','
           ;


opt_semi: /* Empty */
          |
          ';'
          ;


opt_comma_number: /* Empty */
		  {
                   $$ = "";
		  }
		  |
		  ',' INTEGER
		  {
                   $$ = new char[20];
                   $$[0] = ','; 
                   $$[1] = $2+48; 
                   $$[2] = '\0';
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

