/*************************************************************
Copyright (C) 1990, 1991, 1993 Andy C. Hung, all rights reserved.
PUBLIC DOMAIN LICENSE: Stanford University Portable Video Research
Group. If you use this software, you agree to the following: This
program package is purely experimental, and is licensed "as is".
Permission is granted to use, modify, and distribute this program
without charge for any purpose, provided this license/ disclaimer
notice appears in the copies.  No warranty or maintenance is given,
either expressed or implied.  In no event shall the author(s) be
liable to you or a third party for any special, incidental,
consequential, or other damages, arising out of the use or inability
to use the program for any purpose (or the loss of data), even if we
have been advised of such possibilities.  Any public reference or
advertisement of this source code should refer to it as the Portable
Video Research Group (PVRG) code, and not by any author(s) (or
Stanford University) name.
*************************************************************/
%{

/*LABEL lexer.c */

/* Redefine the yywrap so that we don't have
   to worry about lex library */

# define yywrap() (1)

static char *ReservedWords[] = {
"ADD",
"SUB",
"MUL",
"DIV",
"NOT",
"AND",
"OR",
"XOR",
"LT",
"LTE",
"EQ",
"GT",
"GTE",
"NEG",
"SQRT",
"ABS",
"FLOOR",
"CEIL",
"ROUND",
"DUP",
"POP",
"EXCH",
"COPY",
"ROLL",
"INDEX",
"CLEAR",
"STO",
"RCL",
"GOTO",
"IFG",
"IFNG",
"EXIT",
"EXE",
"ABORT",
"PRINTSTACK",
"PRINTPROGRAM",
"PRINTIMAGE",
"PRINTFRAME",
"ECHO",
"OPEN",
"CLOSE",
"EQU",
"VAL",
"STREAMNAME",
"COMPONENT",
"PICTURERATE",
"FRAMESKIP",
"QUANTIZATION",
"SEARCHLIMIT",
"NTSC",
"CIF",
"QCIF",
""};

#define R_ADD 1
#define R_SUB 2
#define R_MUL 3
#define R_DIV 4
#define R_NOT 5
#define R_AND 6
#define R_OR 7
#define R_XOR 8
#define R_LT 9
#define R_LTE 10
#define R_EQ 11
#define R_GT 12
#define R_GTE 13
#define R_NEG 14
#define R_SQRT 15
#define R_ABS 16
#define R_FLOOR 17
#define R_CEIL 18
#define R_ROUND 19
#define R_DUP 20
#define R_POP 21
#define R_EXCH 22
#define R_COPY 23
#define R_ROLL 24
#define R_INDEX 25
#define R_CLEAR 26
#define R_STO 27
#define R_RCL 28
#define R_GOTO 29
#define R_IFG 30
#define R_IFNG 31
#define R_EXIT 32
#define R_EXE 33
#define R_ABORT 34
#define R_PRINTSTACK 35
#define R_PRINTPROGRAM 36
#define R_PRINTIMAGE 37
#define R_PRINTFRAME 38
#define R_ECHO 39
#define R_OPEN 40
#define R_CLOSE 41
#define R_EQU 42
#define R_VAL 43
#define R_STREAMNAME 44
#define R_COMPONENT 45
#define R_PICTURERATE 46
#define R_FRAMESKIP 47
#define R_QUANTIZATION 48
#define R_SEARCHLIMIT 49
#define R_NTSC 50
#define R_CIF 51
#define R_QCIF 52


#define R_INTEGER 1000
#define R_LBRACKET 1001
#define R_RBRACKET 1002
#define R_ID 1003
#define R_STRING 1004
#define R_REAL 1005


static char *EquLabels[] = {
"SQUANT",
"MQUANT",
"PTYPE",
"MTYPE",
"BD",
"FDBD",
"BDBD",
"IDBD",
"VAROR",
"FVAR",
"BVAR",
"IVAR",
"DVAR",
"RATE",
"BUFFERSIZE",
"BUFFERCONTENTS",
"QDFACT",
"QOFFS",
""};


#define L_SQUANT 1
#define L_MQUANT 2
#define L_PTYPE 3
#define L_MTYPE 4
#define L_BD 5
#define L_FDBD 6
#define L_BDBD 7
#define L_IDBD 8
#define L_VAROR 9
#define L_FVAR 10
#define L_BVAR 11
#define L_IVAR 12
#define L_DVAR 13
#define L_RATE 14
#define L_BUFFERSIZE 15
#define L_BUFFERCONTENTS 16
#define L_QDFACT 17
#define L_QOFFS 18

int CommentDepth = 0;  /* depth of comment nesting */
int yyint=0;           /* Return value for integers */
int LexDebug=0;        /* Status of lex debugging */

#define PRIME 211
#define EOS '\0'

#define MakeStructure(S) (S *) malloc(sizeof(S))
#define InsertLink(link,list){\
if(!list){list=link;}else{link->next=list;list=link;}}

#define ID struct id
#define LINK struct link_def
ID {         /* Default id structure */
  char *name;
  int tokentype;
  int count;
  int value;
};

LINK {              /* A link for the hash buckets */
ID *lid;
LINK *next;
};

ID *Cid=NULL;

/*PUBLIC*/

extern void initparser();
extern void parser();
extern void Execute();

static int hashpjw();
static LINK * MakeLink();
static ID * enter();
static char * getstr();
static void PrintProgram();
static void MakeProgram();
static void CompileProgram();
static int mylex();

/*PRIVATE*/

/*NOPROTO*/

%}

Delim		[ \t\n]
WhiteSpace	{Delim}+
Letter	  	[a-zA-Z]
Digit	 	[0-9]
HexDigit  	({Digit}|[a-fA-F])
OctalDigit	[0-7]
Id		{Letter}({Letter}|{Digit})*(:)?
DecInteger 	[-+]?{Digit}+
HexInteger 	[-+]?0[xX]{HexDigit}+
OctInteger 	[-+]?0[oO]{OctalDigit}+
HexInteger2	[-+]?{HexDigit}+[Hh]
OctInteger2	[-+]?{OctalDigit}+[BCObco]
CharInteger     [-+]?'([^\\]|\\([\n^\n]|{OctalDigit}{1,3}))'
ScaleFactor	[eE][-+]?{Digit}+
Real1		({Digit}+"."{Digit}*({ScaleFactor})?)
Real2		({Digit}*"."{Digit}+({ScaleFactor})?)
Real3		({Digit}+{ScaleFactor})
Real		{Real1}|{Real2}|{Real3}
String		\"([^\"]|\\\")*\"

%S NORMAL COMMENT
%%

<NORMAL>{WhiteSpace}	{}

<NORMAL>{Id}	{Cid = enter(0,yytext,yyleng); 
		 if (LexDebug)
		   {
		     printf("%s : %s (%d)\n",
			    yytext,
			    ((Cid->tokentype) ? "RESERVED" : "IDENTIFIER"),
			    Cid->count);
		   }
		 if (Cid->tokentype)
		   {
		     return(Cid->tokentype);
		   }
		 else
		   {
		     yyint = Cid->value;
		     return(R_ID);
		   }
	       }
<NORMAL>{Real}	        {if (LexDebug)
			   {
			     printf("%s : %s\n", yytext, "REAL");
			   }
			 return(R_REAL);
		       }
<NORMAL>{DecInteger}	{if (LexDebug)
			   {             
			     printf("%s : %s\n", yytext, "INTEGER");
			   }
			 yyint = atoi(yytext);
			 return(R_INTEGER);}
<NORMAL>{HexInteger}	{if (LexDebug)
			   {
			     printf("%s : %s\n", yytext, "(HEX)INTEGER");
			   }
			 yyint = strtol(yytext+2,NULL,16);
			 return(R_INTEGER);}
<NORMAL>{HexInteger2}	{if (LexDebug)
			   {
			     printf("%s : %s\n", yytext, "(HEX)INTEGER");
			   }
			 yyint = strtol(yytext,NULL,16);
			 return(R_INTEGER);}
<NORMAL>{OctInteger}	{if (LexDebug)
			   {
			     printf("%s : %s\n", yytext, "(OCT)INTEGER");
			   }
			 yyint = strtol(yytext+2,NULL,8);
			 return(R_INTEGER);}
<NORMAL>{OctInteger2}	{if (LexDebug)
			   {
			     printf("%s : %s\n", yytext, "(OCT)INTEGER");
			   }
			 yyint = strtol(yytext,NULL,8);
			 return(R_INTEGER);}
<NORMAL>{CharInteger}	{if (LexDebug)
			   {
			     printf("%s : %s\n", yytext, "(CHAR)INTEGER");
			   }
			 if (yyleng>4)
			   {
			     yyint = strtol(yytext+2,NULL,8);
			   }
			 else
			   {
			     if (*(yytext+1)=='\\')
			       {
				 switch(*(yytext+2))
				   {
				   case '0':
				     yyint=0;
				     break;
				   case 'b':
				     yyint = 0x8;
				     break;
				   case 'i':
				     yyint = 0x9;
				     break;
				   case 'n':
				     yyint = 0xa;
				     break;
				   case 'v':
				     yyint = 0xb;
				     break;
				   case 'f':
				     yyint = 0xc;
				     break;
				   case 'r':
				     yyint = 0xd;
				     break;
				   default:
				     yyint=(*yytext+2);
				     break;
				   }
			       }
			     else
			       {
				 yyint = *(yytext+1);
			       }
			   }
			 return(R_INTEGER);}
<NORMAL>\[	        {if (LexDebug)
			   {
			     printf("%s : %s\n", yytext, "LBRACKET");
			   }
			 return(R_LBRACKET);}
<NORMAL>\]	        {if (LexDebug)
			   {
			     printf("%s : %s\n", yytext, "RBRACKET");
			   }
			 return(R_RBRACKET);}
<NORMAL>{String}	{if (LexDebug)
			   {
			     printf("%s : %s\n", yytext, "STRING");
			   }
			 return(R_STRING);}

<NORMAL,COMMENT>"/*"	{CommentDepth++; BEGIN COMMENT;}

<COMMENT>"*/"		{CommentDepth--;if(!CommentDepth) BEGIN NORMAL;}

<NORMAL>.	  	  	{
		  	    /* None of the above rules applicable, so
			       it's a bad symbol. */
                              printf("Bad input char '%c' on line %d\n",
  	  	  	  	    yytext[0],
  	  	  	  	    yylineno);
  	  	  	}

<COMMENT>.|\n		{} /*Everything's AOK */

%%

/*PROTO*/
#define NUMBER_PROGRAMS 10
#define MEMTOP 1024
#define MAXIMUM_LINES 2000

LINK *HashTable[PRIME];  /* My little hash table */
int DataLevel = 0;       /* pointer within stack */
double DataStack[MEMTOP]; /* The data stack */
double *DataPtr;

int NextVal=0;            /* the  number of values to load directly
			     from the program */

double Memory[MEMTOP];

int LocalLevel=0;
int CommandLevel=0;
double *LocalStack;
int *CommandStack;

int CurrentLine=0;
int CurrentProgram=0;
double ProgramLocalStack[NUMBER_PROGRAMS][MAXIMUM_LINES];
int ProgramCommandStack[NUMBER_PROGRAMS][MAXIMUM_LINES];
int ProgramLevel[NUMBER_PROGRAMS];
int ProgramLocalLevel[NUMBER_PROGRAMS];

int PProgram=0;
int PLevel=0;
int PLLevel=0;
int *PCStack=NULL;
double *PLStack=NULL;

int LabelLevel=0;
ID *LabelStack[1000];

int SourceLevel=0;
int SourceProgramStack[16];
int SourceLineStack[16];

/*START*/

/*BFUNC

initparser() is used to place the Reserved Words into the hash table.
It must be called before the parser command is called.

EFUNC*/

void initparser()
{
  char i,**sptr;
  BEGIN NORMAL;

  for(i=1,sptr=ReservedWords;**sptr!='\0';i++,sptr++) 
    {     /* Add Reserved Words */
      enter(i,*sptr,strlen(*sptr));
    }
  for(i=1,sptr=EquLabels;**sptr!='\0';i++,sptr++) 
    {     /* Add defined labels */
      equname(i,*sptr);
    }
  for(i=0;i<NUMBER_PROGRAMS;i++)
    {
      ProgramLevel[i]=0;
    }
  DataLevel=0;
  DataPtr = DataStack;
}

#undef BEGIN
#undef MakeStructure
#include "globals.h"
#include "stream.h"
#include <math.h>

#define pushdata(val) *(DataPtr++) = (double) (val); DataLevel++;

extern FRAME *CFrame;
extern IMAGE *CImage;
extern int ErrorValue;
extern int Prate;
extern int FrameSkip;
extern int SearchLimit;
extern int ImageType;
extern int InitialQuant;

/*BFUNC

hashpjw() returns a hash value for a string input.

EFUNC*/

static int hashpjw(s)
     char *s;
{
  BEGIN("hashpjw");
  char *p;
  unsigned int h=0,g;

  for(p=s;*p!=EOS;p++)
    {
      h = (h << 4) + *p;
      if (g = h&0xf0000000)
	{
	  h = h ^(g >> 24);
	  h = h ^ g;
	}
    }
  return(h % PRIME);
}

/*BFUNC

MakeLink() is used to construct a link object. The link
is used for the hash table construct.

EFUNC*/


static LINK *MakeLink(tokentype,str,len)
     int tokentype;
     char *str;
     int len;
{
  BEGIN("MakeLink");
  LINK *temp;
  
  if (!(temp = MakeStructure(LINK)))
    {
      WHEREAMI();
      printf("Cannot make a LINK.\n");
      exit(ERROR_MEMORY);
    }
  if (!(temp->lid = MakeStructure(ID)))
    {
      printf("Cannot make an id.\n");
      exit(ERROR_MEMORY);
    }
  temp->next = NULL;
  if (!(temp->lid->name =(char *)calloc(len+1,sizeof(char))))
    {
      printf("Cannot make a string space for the link.\n");
      exit(ERROR_MEMORY);
    }
  strcpy(temp->lid->name,str);
  temp->lid->tokentype = tokentype;
  temp->lid->count = 1;
  temp->lid->value = -1; /* Unreferenced yet. */
  return(temp);
}

/*BFUNC

enter() is used to enter a Reserved Word or ID into the hash table.

EFUNC*/

static ID *enter(tokentype,str,len)
     int tokentype;
     char *str;
     int len;
{
  BEGIN("enter");
  int hashnum;
  LINK *temp,*current;
  char *ptr;

  for(ptr=str;*ptr!='\0';ptr++)
    {
      if ((*ptr>='a') && (*ptr<='z'))
	{
	  *ptr = *ptr - ('a'-'A');
	}
    }
  hashnum = hashpjw(str);  /* Check if in hash table */
  for(temp=NULL,current=HashTable[hashnum];
      current!= NULL;
      current=current->next)
    {
      if (strcmp(str,current->lid->name) == 0)
	{
	  temp=current;
	  break;
	}
    }
  if (temp)   /* Yes, found ID then return */
    {
      temp->lid->count++;
      return(temp->lid);
    }
  else        /* Else make our own ID and return that*/
    {
      temp = MakeLink(tokentype,str,len);
      InsertLink(temp,HashTable[hashnum]);
      return(temp->lid);
    }
}

equname(number,name)
     int number;
     char *name;
{
  ID *temp;
  temp = enter(0,name,strlen(name));
  temp->value=number;
}


/*BFUNC

getstr() gets a string from the input. It copies the string to
temporary storage before it returns the pointer.

EFUNC*/

static char *getstr()
{
  BEGIN("getstr");
  char *tmp,*ptr,*bptr;
  int i,accum,flag;
  if (mylex() != R_STRING)
    {
      printf("String expected.\n");
      if (!(tmp=(char *) malloc(sizeof(char))))
	{
	  WHEREAMI();
	  printf("Cannot allocate for null string.\n");
	  exit(ERROR_MEMORY);
	}
      *tmp='\0';
      return(tmp);
    }
  if (!(tmp=(char *)calloc(strlen(yytext)+1,sizeof(char))))
    {
      WHEREAMI();
      printf("Cannot allocate %d string space.\n",yyleng);
      exit(ERROR_MEMORY);
    }
  for(bptr=yytext+1,ptr=tmp;*bptr!='"';bptr++,ptr++)
    {
      if (*bptr=='\\')
	{
	  bptr++;
	  for(flag=0,accum=0,i=0;i<3;i++)  /* Octal character lookahead */
	    {
	      if ((*bptr>='0')&&(*bptr<='7'))
		{
		  accum = (accum<<3)+(*bptr-'0');
		  bptr++;
		  flag=1;
		}
	      else {break;}
	    }
	  if (flag) {bptr--;*ptr=accum;}
	  else
	    {
	      switch(*(bptr))
		{
		case '0':
		  *ptr = 0;
		  break;
		case 'b':
		  *ptr = 0x8;
		  break;
		case 'i':
		  *ptr = 0x9;
		  break;
		case 'n':
		  *ptr = 0xa;
		  break;
		case 'v':
		  *ptr = 0xb;
		  break;
		case 'f':
		  *ptr = 0xc;
		  break;
		case 'r':
		  *ptr = 0xd;
		  break;
		default:
		  *ptr=(*bptr);
		}
	    }
	}
      else {*ptr = (*bptr);}
    }
  *ptr='\0';
  return(tmp);
}

/*BFUNC

parser() handles all of the parsing required for the Program
Interpreter.  It is basically a {\tt while} statement with a very
large case statement for every input.  All unmatched values-- strings,
brackets, etc. are ignored.

EFUNC*/

#define ARRAYBEGIN if (ntoken==R_LBRACKET)\
	    {\
	      arrayflag=1;\
	      ntoken=mylex();\
	    }\
	  if (ntoken!=R_INTEGER)\
	    {\
	      WHEREAMI();\
	      printf("Expected integer.\n");\
	      break;\
	    }\
	  while(1)\
	    {

#define ARRAYEND  if (arrayflag)\
		{\
		  if ((ntoken=mylex())==R_RBRACKET) break;\
		  else if (ntoken!=R_INTEGER)\
		    {\
		      WHEREAMI();\
		      printf("Expected integer or right bracket.\n");\
		      break;\
		    }\
		}\
	      else break;\
	      }

#define BINARYOP(operation)  if (DataLevel<2)\
	    {\
	      printf("Not enough operands on stack.\n");\
	      break;\
	    }\
	  accum = *(--DataPtr);\
	  *(--DataPtr) operation accum;\
	  DataPtr++;\
	  DataLevel--;

#define RELOP(operation)  if (DataLevel<2)\
	    {\
	      printf("Not enough operands on stack.\n");\
	      break;\
	    }\
	  accum = *(--DataPtr); DataPtr--;\
	  if (*(DataPtr) operation (accum)) *(DataPtr++) = 1.0;\
          else *(DataPtr++) = 0.0;\
	  DataLevel--;

#define openprogram(value)\
  PProgram=(value);\
  PLStack = ProgramLocalStack[(value)];\
  PCStack = ProgramCommandStack[(value)];\
  PLLevel = ProgramLocalLevel[(value)];\
  PLevel = ProgramLevel[(value)];


#define pushprogram(program,line)\
  SourceProgramStack[SourceLevel] = CurrentProgram;\
  SourceLineStack[SourceLevel] = CurrentLine;\
  SourceLevel++;\
  CurrentProgram = program;\
  CurrentLine = line;\
  CommandStack = ProgramCommandStack[CurrentProgram];\
  LocalStack = ProgramLocalStack[CurrentProgram];\
  LocalLevel = ProgramLocalLevel[CurrentProgram];\
  CommandLevel = ProgramLevel[CurrentProgram];

#define popprogram()\
  SourceLevel--;\
  CurrentProgram = SourceProgramStack[SourceLevel];\
  CurrentLine = SourceLineStack[SourceLevel];\
  CommandStack = ProgramCommandStack[CurrentProgram];\
  LocalStack = ProgramLocalStack[CurrentProgram];\
  LocalLevel = ProgramLocalLevel[CurrentProgram];\
  CommandLevel = ProgramLevel[CurrentProgram];


#define GETINTEGER(retval)  (retval)=mylex();\
  if ((retval)!=R_INTEGER)\
    {WHEREAMI();\
     printf("Integer expected.\n");\
     break;}


void parser()
{
  BEGIN("parser");
  int i,dest,value,token,ntoken,arrayflag;
  double accum;
  int hold;
  char *sptr;

  while(token=mylex())
    {
      arrayflag=0;
      switch(token)
 	{
	case R_INTEGER:
 	  pushdata((double) yyint);
 	  break;
 	case R_REAL:
 	  pushdata(atof(yytext));
 	  break;

 	case R_ADD:
 	  BINARYOP(+=); 
	  break;
 	case R_SUB:
 	  BINARYOP(-=);
 	  break;
	case R_MUL:
 	  BINARYOP(*=);
 	  break;
 	case R_DIV:
 	  BINARYOP(/=);
 	  break;
 	case R_NOT:
	  accum = *(--DataPtr);
 	  *(DataPtr++) = (accum ? 0.0 : 1.0);
 	  break;
 	case R_AND:
 	  RELOP(&&);
 	  break;
 	case R_OR:
 	  RELOP(||);
 	  break;
 	case R_XOR:
	  if (DataLevel<2)
	    {
	      printf("Not enough operands on stack.\n");
	      break;
	    }
	  accum = *(--DataPtr); DataPtr--;
	  if ((*(DataPtr) && !(accum))||
	      (!(*(DataPtr)) && (accum))) *(DataPtr++) = 1.0;
          else *(DataPtr++) = 0.0;
	  DataLevel--;
 	  break;
 	case R_LT:
	  RELOP(<);
 	  break;
 	case R_LTE:
 	  RELOP(<=);
 	  break; 
	case R_EQ:
 	  RELOP(==);
 	  break;
 	case R_GT:
 	  RELOP(>);
 	  break;
 	case R_GTE:
 	  RELOP(>=);
 	  break;

 	case R_NEG:
 	  accum = *(--DataPtr);
 	  *(DataPtr++) = -(accum); 
	  break;
 	case R_SQRT:
 	  accum = *(--DataPtr);
	  *(DataPtr++) = sqrt(accum);
 	  break;
 	case R_ABS:
 	  accum = *(--DataPtr);
	  *(DataPtr++) = fabs(accum);
 	  break;
 	case R_FLOOR:
 	  accum = *(--DataPtr);
	  *(DataPtr++) = floor(accum);
 	  break;
 	case R_CEIL:
 	  accum = *(--DataPtr);
	  *(DataPtr++) = ceil(accum);
 	  break;
 	case R_ROUND:
 	  accum = *(--DataPtr);
	  *(DataPtr++) = ((accum<0)?ceil(accum-0.5):floor(accum+0.5));
 	  break;

	case R_DUP:
 	  *(DataPtr) = DataPtr[-1];
	  DataPtr++;
 	  DataLevel++;
 	  break;
 	case R_POP: 
	  if (DataLevel)
 	    {
 	      DataLevel--;
 	      DataPtr--;
 	    }
 	  else 	{printf("Not enough stack elements.\n");}
 	  break;
 	case R_EXCH:
 	  *DataPtr = DataPtr[-1];
 	  DataPtr[-1] = DataPtr[-2];
 	  DataPtr[-2] = *DataPtr;
 	  break;
 	case R_COPY:
	  GETINTEGER(ntoken);
 	  if (DataLevel<yyint)
 	    {
 	      WHEREAMI();
 	      printf("Not enough elements\n");
 	      break;
 	    }
 	  for(i=0;i<yyint;i++)
 	    {
 	      *(DataPtr) = DataPtr[-yyint];
 	      DataPtr++;
 	      DataLevel++;
 	    }
 	  break;
 	case R_ROLL:
	  GETINTEGER(ntoken);
	  dest=yyint;
 	  GETINTEGER(ntoken);
	  value=yyint;
	  value = value % dest;
 	  if (value<0) {value+= dest;}
 	  for(i=0;i<value;i++)
	    {DataPtr[i] = DataPtr[i-value];}
 	  for(i=0;i<dest-value;i++)
 	    {DataPtr[-i-1] = DataPtr[-value-i-1];}
 	  for(i=0;i<value;i++)
 	    {DataPtr[i-dest] = DataPtr[i];}
 	  break;
 	case R_INDEX:
	  GETINTEGER(ntoken);
	  if (yyint > DataLevel)
 	    {
	      WHEREAMI();
 	      printf("Index out of bounds\n");
 	      break;
 	    }
 	  *DataPtr = DataPtr[-yyint];
 	  DataPtr++;
 	  DataLevel++;
 	  break;
 	case R_CLEAR:
 	  DataLevel=0; 
	  DataPtr=DataStack;
 	  break;

 	case R_STO:
 	  if (!DataLevel)
 	    {
 	      printf("Not enough stack elements.\n");
 	    }
 	  ntoken = mylex();
 	  if ((ntoken!=R_ID)&&(ntoken!=R_INTEGER))
 	    { 	 
	      printf("Integer or label expected.\n");
 	      break;
 	    }
 	  Memory[yyint]= *(--DataPtr);
 	  DataLevel--;
	  break;
 	case R_RCL:
 	  ntoken = mylex();
 	  if ((ntoken!=R_ID)&&(ntoken!=R_INTEGER))
 	    {
	      printf("Integer or label expected.\n");
 	      break;
 	    }
 	  pushdata(Memory[yyint]);
 	  break;

	case R_GOTO:
	case R_IFG:
	case R_IFNG:
	case R_EXIT:
	  WHEREAMI();
	  printf("Program commands not available on top-level.\n");
	  break;

 	case R_EXE:
 	  ntoken = mylex();
 	  if ((ntoken!=R_ID)&&(ntoken!=R_INTEGER))
	    { 
	      printf("Integer or label expected.\n");
	      break;
	    }
	  pushprogram(yyint,0);
	  break;
	case R_ABORT:
	  GETINTEGER(ntoken);
	  exit(yyint);
	  break;

 	case R_PRINTSTACK:
 	  for(i=0;i<DataLevel;i++)
 	    {
 	      printf("%d: %f\n",i,DataStack[i]);
 	    }
 	  break;
 	case R_PRINTPROGRAM:
 	  ntoken = mylex();
 	  if ((ntoken!=R_ID)&&(ntoken!=R_INTEGER))
 	    {
 	      printf("Integer or label expected.\n");
 	      break;
	    }
 	  openprogram(yyint);
 	  PrintProgram();
 	  break;
	case R_PRINTIMAGE:
	  PrintImage();
	  break;
	case R_PRINTFRAME:
	  PrintFrame();
	  break;

	case R_ECHO:
	  printf("%s\n",getstr());
	  break;
 	case R_OPEN:
 	  ntoken = mylex();
 	  if ((ntoken!=R_ID)&&(ntoken!=R_INTEGER))
 	    {
 	      printf("Integer or label expected.\n");
 	      break;
 	    }
 	  hold = yyint;
	  openprogram(hold);
 	  PLevel=0;
 	  MakeProgram();
 	  CompileProgram();
 	  ProgramLevel[hold]=PLevel;
 	  ProgramLocalLevel[hold]=PLLevel;
	  break;
	case R_CLOSE:
	  WHEREAMI();
	  printf("Close not available on top level.\n");
	  break;

 	case R_EQU: 
	  if (!DataLevel)
 	    {
 	      printf("Not enough stack elements.\n");
 	    }
 	  ntoken = mylex();
 	  if ((ntoken!=R_ID))
 	    {
 	      printf("Label expected.\n"); 
	      break; 
	    }
 	  Cid->value = (int) *(--DataPtr);
 	  DataLevel--;
 	  break;
	case R_VAL:
	  WHEREAMI();
	  printf("VAL is not a valid id on top level.\n");
	  break;

	case R_STREAMNAME:
	  CImage->StreamFileName=getstr();
	  break;
	case R_COMPONENT:
	  ntoken=mylex();
	  ARRAYBEGIN;
	  dest = yyint;
	  ntoken=mylex();
	  if (ntoken!=R_LBRACKET)
	    {
	      WHEREAMI();
	      printf("Left bracket expected.\n");
	      break;
	    }
	  sptr=getstr();
	  strcpy(CFrame->ComponentFilePrefix[dest],sptr);
	  sptr=getstr();
	  strcpy(CFrame->ComponentFileSuffix[dest],sptr);
	  ntoken=mylex();
	  if (ntoken!=R_RBRACKET)
	    {
	      WHEREAMI();
	      printf("Right bracket expected.\n");
	      break;
	    }
	  ARRAYEND;
	  break;
        case R_PICTURERATE:
	  GETINTEGER(ntoken);
	  Prate = yyint;
	  break;
	case R_FRAMESKIP:
	  GETINTEGER(ntoken);
	  /* FrameSkip = yyint; */  /* Currently disallowed */
	  break;
	case R_QUANTIZATION:
	  GETINTEGER(ntoken);
	  InitialQuant = yyint;
	  break;
	case R_SEARCHLIMIT:
	  GETINTEGER(ntoken);
	  /* SearchLimit = yyint; */ /* Currently disallowed */
	  /* BoundValue(SearchLimit,1,31,"SearchLimit"); */
	  break;
	case R_NTSC:
	  ImageType=IT_NTSC;
	  break;
	case R_CIF:
	  ImageType=IT_CIF;
	  break;
	case R_QCIF:
	  ImageType=IT_QCIF;
	  break;
	default:
	  WHEREAMI();
	  printf("Illegal token type encountered: %d\n",token);
	  break;
	}
    }
}

/*BFUNC

PrintProgram() prints out a program that is loaded as current.

EFUNC*/

static void PrintProgram()
{
  BEGIN("PrintProgram");
  int i; 

  for(i=0;i<PLevel;i++)
    {
      switch(PCStack[i])
	{
	case R_ADD:
	case R_SUB:
	case R_MUL:
	case R_DIV:

	case R_NOT:
	case R_AND:
	case R_OR:
	case R_XOR:
	case R_LT:
	case R_LTE:
	case R_EQ:
	case R_GT:
	case R_GTE:

	case R_NEG:
	case R_SQRT:
	case R_ABS:
	case R_FLOOR:
	case R_CEIL:
	case R_ROUND:

	case R_DUP:
	case R_POP:
	case R_EXCH:
	case R_CLEAR:
	case R_EXIT:
	case R_PRINTSTACK:
	case R_PRINTIMAGE:
	case R_PRINTFRAME:
	  printf("%d: %s\n",
		 i,
		 ReservedWords[PCStack[i]-1]);
	  break;
	case R_COPY:
	case R_INDEX:
	case R_STO:
	case R_RCL:
	case R_EXE:
	case R_ABORT:
	case R_PRINTPROGRAM:
	  printf("%d: %s %d\n",
		 i,
		 ReservedWords[PCStack[i]-1],
		 PCStack[i+1]);
	  i++;
	  break;
	case R_ROLL:
	  printf("%d: %s %d %d\n",
		 i,
		 ReservedWords[PCStack[i]-1],
		 PCStack[i+1],
		 PCStack[i+2]);
	  i+=2;
	  break;
	case R_GOTO:
	case R_IFG:
	case R_IFNG:
	  printf("%d: %s %d\n",
		 i,
		 ReservedWords[PCStack[i]-1],
		 PCStack[i+1]);
	  i++;
	  break;
	case R_VAL:
	  printf("%d: %s %f\n",
		 i,
		 ReservedWords[PCStack[i]-1],
		 PLStack[PCStack[i+1]]);
	  i++;
	  break;
	case R_ECHO:
	case R_OPEN:
	case R_CLOSE:
	case R_EQU:
	case R_STREAMNAME:
	case R_COMPONENT:
	case R_PICTURERATE:
	case R_FRAMESKIP:
	case R_QUANTIZATION:
	case R_SEARCHLIMIT:
	case R_NTSC:
	case R_CIF:
	case R_QCIF:
	  WHEREAMI();
	  printf("Top-level token occurring in program: %s.\n",
		 ReservedWords[PCStack[i]-1]);
	  break;
	default:
	  WHEREAMI();
	  printf("Bad token type %d\n",PCStack[i]);
	  break;
	}
    }
}

/*BFUNC

MakeProgram() makes a program from the input from mylex().

EFUNC*/

static void MakeProgram()
{
  BEGIN("MakeProgram");
  int ntoken;

  while((ntoken=mylex())!= R_CLOSE)
    {
      switch(ntoken)
	{
	case 0:
	  exit(-1);
	  break;
	case R_ADD:
	case R_SUB:
	case R_MUL:
	case R_DIV:

	case R_NOT:
	case R_AND:
	case R_OR:
	case R_XOR:
	case R_LT:
	case R_LTE:
	case R_EQ:
	case R_GT:
	case R_GTE:

	case R_NEG:
	case R_SQRT:
	case R_ABS:
	case R_FLOOR:
	case R_CEIL:
	case R_ROUND:

	case R_DUP:
	case R_POP:
	case R_EXCH:
	case R_CLEAR:

	case R_EXIT:
	case R_PRINTSTACK:
	case R_PRINTIMAGE:
	case R_PRINTFRAME:
	  PCStack[PLevel++] = ntoken;
	  break;
	case R_COPY:
	case R_INDEX:
	case R_STO:
	case R_RCL:
	case R_EXE:
	case R_ABORT:
	case R_PRINTPROGRAM:
	  PCStack[PLevel++] = ntoken;
	  ntoken = mylex();
	  if ((ntoken==R_INTEGER)||(ntoken==R_ID))
	    {
	      PCStack[PLevel++] = yyint;
	    }
	  else
	    {
	      PCStack[PLevel++] = 0;
	      printf("Integer expected.\n");
	    }
	  break;
	case R_ROLL:
	  PCStack[PLevel++] = ntoken;
	  ntoken = mylex();
	  if ((ntoken==R_INTEGER)||(ntoken==R_ID))
	    {
	      PCStack[PLevel++] = yyint;
	    }
	  else
	    {
	      PCStack[PLevel++] = 0;
	      printf("Integer expected.\n");
	    }
	  ntoken = mylex();
	  if ((ntoken==R_INTEGER)||(ntoken==R_ID))
	    {
	      PCStack[PLevel++] = yyint;
	    }
	  else
	    {
	      PCStack[PLevel++] = 0;
	      printf("Integer expected.\n");
	    }
	  break;
	case R_GOTO:
	case R_IFG:
	case R_IFNG:
	  PCStack[PLevel++] = ntoken;
	  ntoken = mylex();
	  if (ntoken==R_ID)
	    {
	      LabelStack[LabelLevel] = Cid;
	      PCStack[PLevel++] = LabelLevel++;
	    }
	  else
	    {
	      printf("Id expected.\n");
	    }
	  break;
	case R_VAL:
	  PCStack[PLevel++] = ntoken;
	  PLStack[PLLevel]=(double) *(--DataPtr);  /* Take from Top of stack */
	  DataLevel--;
	  PCStack[PLevel++] = PLLevel++;
	  break;
	case R_INTEGER:
	  PCStack[PLevel++] = R_VAL;
	  PLStack[PLLevel]=(double) yyint;
	  PCStack[PLevel++] = PLLevel++;
	  break;
	case R_REAL:
	  PCStack[PLevel++] = R_VAL;
	  PLStack[PLLevel] = atof(yytext);
	  PCStack[PLevel++] = PLLevel++;
	  break;
	case R_ID:
	  if (Cid->value>=0)
	    {
	      WHEREAMI();
	      printf("Attempt to redefine label.\n");
	      break;
	    }
	  Cid->value = PLevel;
	  break;
	default:
	  WHEREAMI();
	  printf("Token type %d not allowed in programs.\n",ntoken);
	  break;
	}
    }
}

/*BFUNC

CompileProgram() assigns values to the labels in a program.

EFUNC*/

static void CompileProgram()
{
  BEGIN("CompileProgram");
  int i;

  for(i=0;i<PLevel;i++)
    {
      switch(PCStack[i])
	{
	case R_ADD:
	case R_SUB:
	case R_MUL:
	case R_DIV:

	case R_NOT:
	case R_AND:
	case R_OR:
	case R_XOR:
	case R_LT:
	case R_LTE:
	case R_EQ:
	case R_GT:
	case R_GTE:

	case R_NEG:
	case R_SQRT:
	case R_ABS:
	case R_FLOOR:
	case R_CEIL:
	case R_ROUND:

	case R_DUP:
	case R_POP:
	case R_EXCH:
	case R_CLEAR:

	case R_EXIT:
	case R_PRINTSTACK:
	case R_PRINTIMAGE:
	case R_PRINTFRAME:
	  break;
	case R_COPY:
	case R_INDEX:
	case R_STO:
	case R_RCL:
	case R_EXE:
	case R_ABORT:
	case R_VAL:
	case R_PRINTPROGRAM:
	  i++;
	  break;
	case R_ROLL:
	  i+=2;
	  break;
	case R_GOTO:
	case R_IFG:
	case R_IFNG:
	  i++;
	  if (!LabelStack[PCStack[i]]->value)
	    {
	      printf("Bad reference to label!\n");
	      break;
	    }
	  PCStack[i] = LabelStack[PCStack[i]]->value;
	  break;
	default:
	  WHEREAMI();
	  printf("Invalid program compilation token: %d.\n",PCStack[i]);
	  break;
	}
    }
}

/*BFUNC

mylex() reads either from the yylex() routine or from the currently
active program depending on what source is active.

EFUNC*/

static int mylex()
{
  BEGIN("mylex");
  int token;

  while(1)
    {
      if (!SourceLevel)
	{
	  return(yylex());
	}
      token = CommandStack[CurrentLine++];
/*
      printf("Token:%d  CommandStack:%x  CurrentLine:%d\n",
	     token,CommandStack,CurrentLine-1);
*/
      if (NextVal)
	{
	  NextVal--;
	  yyint = token;
	  return(R_INTEGER);
	}
      switch(token)
	{
	case 0:
	  printf("Abnormal break at: %d\n",CurrentLine);
	  popprogram();
	  break;
	case R_VAL:
	  pushdata(LocalStack[CommandStack[CurrentLine++]]);
	  break;
	case R_GOTO:
	  CurrentLine = CommandStack[CurrentLine];
	  break;
	case R_IFG:
	  DataLevel--;
	  if (*(--DataPtr))
	    {
	      CurrentLine = CommandStack[CurrentLine];
	    }
	  else CurrentLine++;
	  break;
	case R_IFNG:
	  DataLevel--;
	  if (!(*(--DataPtr)))
	    {
	      CurrentLine = CommandStack[CurrentLine];
	    }
	  else CurrentLine++;
	  break;
	case R_EXIT:
	  popprogram();
	  break;
	case R_COPY:
	case R_INDEX:
	case R_STO:
	case R_RCL:
	case R_EXE:
	case R_ABORT:
	case R_PRINTPROGRAM:
	  NextVal = 1;  /* Notify to take the next integer straight */
	  return(token);
	  break;
	case R_ROLL:
	  NextVal = 2;
	  return(token);
	  break;
	default:
	  return(token);
	  break;
	}
    }
}

/*BFUNC

Execute() calls the program interpreter to execute a particular
program location.

EFUNC*/

void Execute(pnum)
     int pnum;
{
  BEGIN("Execute");

  if (ProgramLevel[pnum])
    {
      pushprogram(pnum,0);
      parser();
    }
}

/*NOPROTO*/
/*END*/

