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
TODO:
class
*/


%{

#include <stdio.h>
extern "C" {
extern int yylex(void);
int yyerror(char*);
}
#include <Core/CCA/tools/sidl/Spec.h>
 Specification* parse_spec;
extern char* curfile;
extern int lineno;
#define YYDEBUG 1
#include <stdlib.h>
#include <iostream>

 using namespace std;
 
%}

%union {
  char* ident;
  char* string;
  int number;
  DefinitionList* definition_list;
  Definition* definition;
  BaseInterface* interface;
  Package* package;
  ScopedName* scoped_identifier;
  Class* c_class;
  Class::Modifier class_modifier;
  DistributionArray* distributionarray;
  Enum* c_enum;
  Enumerator* enumerator;
  struct {
    ScopedName* class_extends;
    ScopedNameList* class_implementsall;
    ScopedNameList* class_implements;
  } class_inherit;
  bool boolean;
  ScopedNameList* scoped_identifier_list;
  struct {
    MethodList* method_list;
    DistributionArrayList* distarray_list;
  } statement_list;
  Method* method;
  Method::Modifier method_modifier;
  Method::Modifier2 method_modifiers2;
  Type* type;
  struct {
    Type* type;
    bool copy;
  } return_type;
  ArgumentList* argument_list;
  Argument* argument;
  Argument::Mode mode;
  Version* version;
  VersionList* version_list;
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

%type<argument> argument
%type<argument_list> arguments comma_argument_star
%type<boolean> opt_dot opt_copy
%type<c_class> class
%type<class_inherit> class_inherit
%type<class_modifier> class_modifier
%type<definition> definition
%type<definition_list> definition_star
%type<distributionarray> distribution
%type<enumerator> enumerator
%type<c_enum> enumerator_list enum
%type<interface> interface
%type<scoped_identifier_list> import_star
%type<scoped_identifier> import
%type<method> method class_method
%type<statement_list> statements statements_star
%type<statement_list> class_statements class_statement_star
%type<method_modifier> method_modifier
%type<method_modifiers2> method_modifiers2
%type<mode> mode
%type<number> opt_comma_number
%type<package> package
%type<definition_list> package_star
%type<scoped_identifier> class_extends
%type<scoped_identifier> scoped_identifier dot_identifier_star
%type<scoped_identifier_list> class_implements class_implements_all
%type<scoped_identifier_list> comma_scoped_identifier_star
%type<scoped_identifier_list> interface_extends
%type<scoped_identifier_list> opt_throws_clause throws_clause
%type<type> type
%type<return_type> return_type;
%type<version> version
%type<version_list> version_star

%%

specification: version_star import_star package_star
               {
		 parse_spec=new Specification($1, $2, $3);
	       }
	       ;

version_star: /* Empty */
	      {
		$$=new VersionList();
	      }
	      |
	      version_star version
	      {
		$$=$1;
		$$->add($2);
	      }
	      ;

version: VERSION IDENTIFIER INTEGER ';'
         {
	   $$=new Version($2, $3);
         }
         |
         VERSION IDENTIFIER VERSION_STRING ';'
         {
	   $$=new Version($2, $3);
         }
         ;

package_star: /* Empty */
              {
		$$=new DefinitionList();
	      }
              |
	      package_star package
              {
		$$=$1;
		$$->add($2);
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

import_star: /* Empty */
             {
	       $$=new ScopedNameList();
	     }
             |
	     import_star import
             {
	       $$=$1;
	       $$->add($2);
	     }
             ;

import: IMPORT scoped_identifier ';'
        {
	  $$=$2;
	}
        ;

definition:    class
	       {
		 $$=$1;
	       }
	       |
	       enum
	       {
		 $$=$1;
	       }
	       |
	       interface
	       {
		 $$=$1;
	       }
	       |
               package
	       {
		 $$=$1;
	       }
               |
	       /* Not in SSIDL spec but added for MxN soln. purposes */
	       distribution
               {
		 $$=$1;
               }
	       ;

distribution:  DISTRIBUTION ARRAY IDENTIFIER '<' type opt_comma_number '>' ';'
               {
		 ArrayType* arr_t = new ArrayType($5,$6);
		 $$=new DistributionArray(curfile,lineno,$3,arr_t);
               }
               ;

package:       PACKAGE scoped_identifier '{' definition_star '}' opt_semi
	       {
		 Package* pack = new Package(curfile, lineno, $2->name($2->nnames()-1), $4);
		 for(int i=$2->nnames()-2;i>=0;i--){
		   DefinitionList* defs = new DefinitionList();
		   defs->add(pack);
		   pack = new Package(curfile, lineno, $2->name(i), defs);
		 }
		 $$=pack;
	       }
               ;

scoped_identifier:   opt_dot IDENTIFIER dot_identifier_star
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

opt_semi:      /* Empty */
	       |
	       ';'
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


class:	       class_modifier CLASS IDENTIFIER class_inherit class_statements opt_semi
	       {
		 $$=new Class(curfile, lineno, $1, $3,
			      $4.class_extends, $4.class_implementsall,
			      $4.class_implements, $5.method_list, $5.distarray_list);
	       }
	       ;

class_modifier: /* Empty */
                {	
		  $$=Class::None;
		}
                |
                ABSTRACT
                {
		  $$=Class::Abstract;
		}
		;

class_inherit: class_extends class_implements_all class_implements
	       {
		 $$.class_extends=$1;
		 $$.class_implementsall=$2;
		 $$.class_implements=$3;
	       }
	       ;

class_extends: /* Empty */
	       {
		 $$=0;
	       }
	       |
	       EXTENDS scoped_identifier
	       {
		 $$=$2;
	       }
	       ;

class_implements_all: /* Empty */
		      {
			$$=0;
		      }
		      |
		      IMPLEMENTSALL scoped_identifier comma_scoped_identifier_star
                      {
			$3->prepend($2);
			$$=$3;
		      }
                      ;

class_implements: /* Empty */
		  {
		     $$=0;
		  }
		  |
		  IMPLEMENTS scoped_identifier comma_scoped_identifier_star
		  {
		    $3->prepend($2);
		    $$=$3;
		  }
		  ;

comma_scoped_identifier_star: /* Empty */
			{
			  $$=new ScopedNameList();
			}
			|
			comma_scoped_identifier_star ',' scoped_identifier
			{
			  $$=$1;
			  $$->add($3);
			}
			;

enum: ENUM IDENTIFIER '{' enumerator_list opt_comma '}' opt_semi
      {
	$$=$4;
        $$->setName($2);
      }
      ;

opt_comma: /* Empty */
           |
           ','
           ;

enumerator_list: enumerator
                 {
		   $$=new Enum(curfile, lineno, "tmp_enum");
		   $$->add($1);
		 }
                 |
                 enumerator_list ',' enumerator
                 {
		   $$=$1;
		   $$->add($3);
		 }
                 ;

enumerator: IDENTIFIER
            {
	      $$=new Enumerator(curfile, lineno, $1);
	    }
            |
            IDENTIFIER '=' INTEGER
            {
	      $$=new Enumerator(curfile, lineno, $1, $3);
	    }
            ;

interface:     INTERFACE IDENTIFIER interface_extends statements opt_semi
	       {
		 $$=new BaseInterface(curfile, lineno, $2, $3, $4.method_list, $4.distarray_list);
	       }
	       ;


class_statements: '{' class_statement_star '}'
		  {
		    $$=$2;
		  }
		  ;

class_statement_star: /* Empty */
		   {
		     $$.method_list = new MethodList();
		     $$.distarray_list = new DistributionArrayList();
		   }
		   |
		   class_statement_star class_method
		   {
		     $$=$1;
		     ($$.method_list)->add($2);
		   }
                   |
                   /* Not in SSIDL spec but added for MxN soln. purposes */
                   class_statement_star distribution
                   {
		     $$=$1;
		     ($$.distarray_list)->add($2);
                   }
                   ;

interface_extends: /* Empty */
		   {
		     $$=0;
		   }
		   |
		   EXTENDS scoped_identifier comma_scoped_identifier_star
		   {
		     $3->prepend($2);
		     $$=$3;
		   }
		   ;

class_method: method_modifier method
              {
		$$=$2;
		$$->setModifier($1);
	      }
              ;

method_modifier: /* Empty */
                 {
		   $$=Method::NoModifier;
		 }
                 |
		 ABSTRACT
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

statements: '{' statements_star '}'
	    {
	      $$=$2;
	    }
	    ;

statements_star: /* Empty */
	         {
	           $$.method_list = new MethodList();
		   $$.distarray_list = new DistributionArrayList();  
                 }
	         |
	         statements_star method
	         {
	           $$=$1;
	           ($$.method_list)->add($2);
                 }
                 |
		 /* Not in SSIDL spec but added for MxN soln. purposes */
                 statements_star distribution
                 {
		   $$=$1;
		   ($$.distarray_list)->add($2);
                 }
	         ; 

method: return_type IDENTIFIER arguments method_modifiers2 opt_throws_clause ';'
	{
	  $$=new Method(curfile, lineno, $1.copy, $1.type, $2, $3, $4, $5);
	  $$->isCollective = false;

	}
        |
        COLLECTIVE return_type IDENTIFIER arguments method_modifiers2 opt_throws_clause ';'
	{
	  $$=new Method(curfile, lineno, $2.copy, $2.type, $3, $4, $5, $6);
	  $$->isCollective = true;
		
	}
	;
      

method_modifiers2: /* Empty */
                  {
		    $$=Method::None;
		  }
                  |
                  LOCAL
                  {
		    $$=Method::Local;
		  }
		  |
		  ONEWAY
                  {
		    $$=Method::Oneway;
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
	       $$.type=Type::voidtype();
	       $$.copy=false;
             }
	     |
	     opt_copy type
	     {
	       $$.type=$2;
	       $$.copy=$1;
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

opt_copy: /* Empty */
          {
	    $$=false;
	  }
          |
	  COPY
          {
	    $$=true;
	  }
          ;

argument: opt_copy mode type
	  {
	    $$=new Argument($1, $2, $3);
	  }
	  |
	  opt_copy mode type IDENTIFIER
	  {
	    $$=new Argument($1, $2, $3, $4);
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

throws_clause: THROWS scoped_identifier comma_scoped_identifier_star
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
      LONG
      {
	$$=Type::longtype();
      }
      |
      OPAQUE
      {
	$$=Type::opaquetype();
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
      scoped_identifier
      {
	$$=new NamedType(curfile, lineno, $1);
      }
      ;

opt_comma_number: /* Empty */
		  {
		    $$=0;
		  }
		  |
		  ',' INTEGER
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

