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


%{

#include <stdio.h>
extern "C" {
extern int yylex(void);
int yyerror(char*);
}
#include <Core/CCA/tools/sidl/Spec.h>
Specification specs;
extern char* curfile;
extern int lineno;
#define YYDEBUG 1
#include <stdlib.h>
#include <iostream>

%}

%union {
    char* ident;
    int number;
    DefinitionList* definition_list;
    Definition* definition;
    Interface* interface;
    Package* package;
    ScopedName* scoped_name;
    Class* c_class;
    struct {
       ScopedName* class_extends;
       ScopedNameList* class_implements;
    } class_inherit;
    bool boolean;
    ScopedNameList* scoped_name_list;
    MethodList* method_list;
    Method* method;
    Method::Modifier method_modifier;
    Type* type;
    ArgumentList* argument_list;
    Argument* argument;
    Argument::Mode mode;
};

%start specification

%token PACKAGE
%token<ident> IDENTIFIER
%token CLASS
%token EXTENDS
%token IMPLEMENTS
%token INTERFACE
%token ABSTRACT
%token FINAL
%token STATIC
%token VOID
%token THROWS
%token BOOL
%token CHAR
%token DCOMPLEX
%token DOUBLE
%token FCOMPLEX
%token FLOAT
%token INT
%token STRING
%token ARRAY
%token<number> NUMBER
%token IN
%token OUT
%token INOUT

%type<definition_list> definition_star
%type<definition> definition
%type<interface> interface
%type<package> package
%type<scoped_name> scoped_name dot_identifier_star
%type<c_class> class
%type<class_inherit> class_inherit
%type<scoped_name> class_extends
%type<scoped_name_list> class_implements
%type<boolean> opt_dot
%type<method_list> modified_methods modified_method_star
%type<scoped_name_list> comma_scoped_name_star
%type<scoped_name_list> interface_extends
%type<method_list> methods method_star
%type<method> method
%type<method_modifier> method_modifier
%type<type> return_type type
%type<argument_list> arguments comma_argument_star
%type<scoped_name_list> opt_throws_clause throws_clause
%type<argument> argument
%type<mode> mode
%type<number> opt_comma_number

%%

specification: definition_star
	       {
                   specs.add($1);
	       }
	       ;

definition_star: /* Empty */
	       {
	           $$=new DefinitionList();
	       }
               |
	       definition_star definition
	       {
	          $$=$1;
		  $$->add($2);
               }
	       ;

definition:    package ';'
	       {
	          $$=$1;
	       }
	       |
	       class ';'
	       {
	          $$=$1;
	       }
	       |
	       interface ';'
	       {
	          $$=$1;
	       }
	       ;

package:       PACKAGE IDENTIFIER '{' definition_star '}'
	       {
	           $$=new Package(curfile, lineno, $2, $4);
	       }
	       ;

scoped_name:   opt_dot IDENTIFIER dot_identifier_star
	       {
	           $$=$3;
		   $$->prepend($2);
		   $$->set_leading_dot($1);
	       }
	       ;

opt_dot:       /* Empty */
	       {
		   $$=false;
	       }
	       |
	       '.'
	       {
		   $$=true;
	       }
	       ;

dot_identifier_star: /* Empty */
		     {
		         $$=new ScopedName();
	             }
		     |
		     dot_identifier_star '.' IDENTIFIER
		     {
		         $$=$1;
			 $$->add($3);
		     }
		     ;


class:	       CLASS IDENTIFIER
	       {
	           $$=new Class(curfile, lineno, $2);
	       }
	       |
	       CLASS IDENTIFIER class_inherit modified_methods
	       {
	           $$=new Class(curfile, lineno, $2,
				$3.class_extends, $3.class_implements, $4);
	       }
	       ;

class_inherit: class_extends class_implements
	       {
	           $$.class_extends=$1;
		   $$.class_implements=$2;
	       }
	       ;

class_extends: /* Empty */
	       {
	           $$=0;
	       }
	       |
	       EXTENDS scoped_name
	       {
	           $$=$2;
	       }
	       ;

class_implements: /* Empty */
		  {
		     $$=0;
		  }
		  |
		  IMPLEMENTS scoped_name comma_scoped_name_star
		  {
		     $3->prepend($2);
		     $$=$3;
		  }
		  ;

comma_scoped_name_star: /* Empty */
			{
			    $$=new ScopedNameList();
			}
			|
			comma_scoped_name_star ',' scoped_name
			{
			    $$=$1;
			    $$->add($3);
			}
			;

interface:     INTERFACE IDENTIFIER
               {
	           $$=new Interface(curfile, lineno, $2);
	       }
	       |
	       INTERFACE IDENTIFIER interface_extends methods
	       {
	           $$=new Interface(curfile, lineno, $2, $3, $4);
	       }
	       ;

modified_methods: '{' modified_method_star '}'
		  {
		      $$=$2;
		  }
		  ;

modified_method_star: /* Empty */
		      {
		          $$=new MethodList();
		      }
		      |
		      modified_method_star method ';'
		      {
		          $$=$1;
			  $$->add($2);
		      }
		      |
		      modified_method_star method_modifier method ';'
		      {
		          $$=$1;
			  $3->setModifier($2);
			  $$->add($3);
		      }
		      ;

interface_extends: /* Empty */
		   {
		       $$=0;
		   }
		   |
		   EXTENDS scoped_name comma_scoped_name_star
		   {
		       $3->prepend($2);
		       $$=$3;
		   }
		   ;

method_modifier: ABSTRACT
		 {
		     $$=Method::Abstract;
		 }
		 |
		 FINAL
		 {
		     $$=Method::Final;
		 }
		 |
		 STATIC
		 {
		     $$=Method::Static;
		 }
		 ;

methods: '{' method_star '}'
	 {
	     $$=$2;
	 }
	 ;

method_star: /* Empty */
	     {
	         $$=new MethodList();
             }
	     |
	     method_star method ';'
	     {
	         $$=$1;
		 $$->add($2);
             }
	     ;

method: return_type IDENTIFIER arguments opt_throws_clause
	{
	    $$=new Method(curfile, lineno, $1, $2, $3, $4);
	}
	;

opt_throws_clause: /* Empty */
		   {
		       $$=0;
		   }
		   |
		   throws_clause
		   {
		       $$=$1;
		   }
		   ;

return_type: VOID
	     {
	         $$=Type::voidtype();
             }
	     |
	     type
	     {
	         $$=$1;
	     }
	     ;

arguments: '(' ')'
	   {
	      $$=new ArgumentList();
	   }
	   |
	   '(' argument comma_argument_star ')'
	   {
	      $$=$3;
              $$->prepend($2);
	   }
	   ;

argument: mode type
	  {
	      $$=new Argument($1, $2);
	  }
	  |
	  mode type IDENTIFIER
	  {
	      $$=new Argument($1, $2, $3);
	  }
	  ;

mode: IN
      {
         $$=Argument::In;
      }
      |
      OUT
      {
         $$=Argument::Out;
      }
      |
      INOUT
      {
         $$=Argument::InOut;
      }
      ;

comma_argument_star: /* Empty */
		     {
		        $$=new ArgumentList();
	             }
		     |
		     comma_argument_star ',' argument
		     {
		        $$=$1;
			$$->add($3);
		     }
		     ;

throws_clause: THROWS scoped_name comma_scoped_name_star
	       {
	           $3->prepend($2);
	           $$=$3;
	       }
	       ;

type: BOOL
      {
         $$=Type::booltype();
      }
      |
      CHAR
      {
         $$=Type::chartype();
      }
      |
      DCOMPLEX
      {
         $$=Type::dcomplextype();
      }
      |
      DOUBLE
      {
         $$=Type::doubletype();
      }
      |
      FCOMPLEX
      {
         $$=Type::fcomplextype();
      }
      |
      FLOAT
      {
         $$=Type::floattype();
      }
      |
      INT
      {
         $$=Type::inttype();
      }
      |
      STRING
      {
         $$=Type::stringtype();
      }
      |
      ARRAY '<' type opt_comma_number '>'
      {
         $$=Type::arraytype($3, $4);
      }
      |
      scoped_name
      {
         $$=new NamedType(curfile, lineno, $1);
      }
      ;

opt_comma_number: /* Empty */
		  {
		      $$=0;
		  }
		  |
		  ',' NUMBER
		  {
		      $$=$2;
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

