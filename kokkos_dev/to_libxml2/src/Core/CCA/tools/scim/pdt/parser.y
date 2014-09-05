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
extern int pdtlex(void);
int pdterror(char*);
}

extern char* pdt_curfile;
extern int pdt_lineno;

#define YYDEBUG 1
#include <stdlib.h>
#include <string>
#include "../IR.h"
#include "pdtParser.h"
using namespace std;

extern IR* ir; 
string outCodeR; /* Code Repository */
string inCodeR;
string inoutCodeR;
 

void parseToIr(std::string mode) {

  char execline[100];
  FILE* workfile;
  std::string filename;	 
  filename = pdt_curfile; 
  filename.append(".");
  filename.append(mode.c_str());	
  sprintf(execline,"cxxparse %s",filename.c_str());

  if(mode == "in") {
    if(inCodeR.length() == 0) return;
    workfile = fopen(filename.c_str(),"w");
    if(workfile == NULL) return; 
    fprintf(workfile,"%s \n",inCodeR.c_str());
    fclose(workfile);
    system(execline); 
    filename.append(".pdb");
    pdtParser* pdt = new pdtParser();
    IrDefList* defL = pdt->pdtParse(filename.c_str());
    if(defL != NULL) ir->addInDefList(defL);
  }
  else if(mode == "out") {
    if(outCodeR.length() == 0) return;
    workfile = fopen(filename.c_str(),"w");
    if(workfile == NULL) return;
    fprintf(workfile,"%s \n",outCodeR.c_str());
    fclose(workfile);
    system(execline);
    filename.append(".pdb");
    pdtParser* pdt = new pdtParser();
    IrDefList* defL = pdt->pdtParse(filename.c_str());
    if(defL != NULL) ir->addOutDefList(defL);
  }    
  else if(mode == "inout") {
    if(inoutCodeR.length() == 0) return;
    workfile = fopen(filename.c_str(),"w");
    if(workfile == NULL) return;
    fprintf(workfile,"%s \n",inoutCodeR.c_str());
    fclose(workfile);
    system(execline);
    filename.append(".pdb");
    pdtParser* pdt = new pdtParser();
    IrDefList* defL = pdt->pdtParse(filename.c_str());
    if(defL != NULL) {
      for(int i=0; i<defL->getSize(); i++) {
	ir->addInOutDef(defL->getDef(i));
      }
    }
  }
  else {
    return;
  }

  sprintf(execline,"rm -f %s*",filename.substr(0,filename.size()-4).c_str());
  system(execline);
}  

%}

%union {
  char* ident;
  char* oline;

  std::string* text;
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

%token<ident> IDENTIFIER
%token<oline> OTHERLINE

%token MAP
%token ININTERFACE
%token OUTINTERFACE
%token CLOSEININTERFACE
%token CLOSEOUTINTERFACE 
%token INOUTREMAP
%token INOUTOMIT 
%token ARROW
%token DBLANGLE

%type<oline> otherline
%type<text> otherline_star
%type<irNM_list> namemap_star subnamemap_star
%type<irNM> namemap subnamemap
%%

specification: mainline_star
               {
                 parseToIr("in");
                 parseToIr("out");
		 parseToIr("inout"); 
               }  

mainline_star: /* Empty */
               {
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
          otherline
          {
            inoutCodeR.append($1);
            inoutCodeR.append("\n");
          }
          ;




otherline_star: /* Empty */ 
              {
                $$ = new string();  
              }
              | 
              otherline_star otherline 
              {
                $1->append($2);
                $1->append("\n");
              }
              ;

otherline: OTHERLINE
         {
           $$ = $1;
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
         ININTERFACE otherline_star CLOSEININTERFACE
         {
	   inCodeR.append($2->c_str());
           inCodeR.append("\n");
	 }
         |
         OUTINTERFACE otherline_star CLOSEOUTINTERFACE
         {
           outCodeR.append($2->c_str());
           outCodeR.append("\n");
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



%%

int pdterror(char* s)
{
  extern int pdt_lineno;
  extern char* pdt_curfile;
  fprintf(stderr, "%s: %s at line %d\n" , pdt_curfile, s, pdt_lineno);
  return -1;
}

