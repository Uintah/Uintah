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
# include "stdio.h"
#include "string.h"
# define U(x) ((x)&0377)
# define NLSTATE yyprevious=YYNEWLINE
# define BEGIN yybgin = yysvec + 1 +
# define INITIAL 0
# define YYLERR yysvec
# define YYSTATE (yyestate-yysvec-1)
# define YYOPTIM 1
# define YYLMAX 200
# define output(c) putc(c,yyout)
# define input() (((yytchar=yysptr>yysbuf?U(*--yysptr):getc(yyin))==10?(yylineno++,yytchar):yytchar)==EOF?0:yytchar)
# define unput(c) {yytchar= (c);if(yytchar=='\n')yylineno--;*yysptr++=yytchar;}
# define yymore() (yymorfg=1)
# define ECHO fprintf(yyout, "%s",yytext)
# define REJECT { nstr = yyreject(); goto yyfussy;}
int yyleng; extern char yytext[];
int yymorfg;
extern char *yysptr, yysbuf[];
int yytchar;

#ifdef LINUX
FILE *yyin = 0, *yyout = 0;
#else
FILE *yyin ={stdin}, *yyout ={stdout};
#endif

extern int yylineno;
struct yysvf { 
	struct yywork *yystoff;
	struct yysvf *yyother;
	int *yystops;};
struct yysvf *yyestate;
extern struct yysvf yysvec[], *yybgin;

void equname(int, char*);

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

# define NORMAL 2
# define COMMENT 4
# define YYNEWLINE 10
yylex(){

int nstr; extern int yyprevious;

#ifdef LINUX
yyout = stdout;
yyin = stdin;
#endif

while((nstr = yylook()) >= 0)
yyfussy: switch(nstr){
case 0:
if(yywrap()) return(0); break;
case 1:
{}
break;
case 2:
{Cid = enter(0,yytext,yyleng); 
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
case 3:
        {if (LexDebug)
			   {
			     printf("%s : %s\n", yytext, "REAL");
			   }
			 return(R_REAL);
		       }
case 4:
{if (LexDebug)
			   {             
			     printf("%s : %s\n", yytext, "INTEGER");
			   }
			 yyint = atoi(yytext);
			 return(R_INTEGER);}
case 5:
{if (LexDebug)
			   {
			     printf("%s : %s\n", yytext, "(HEX)INTEGER");
			   }
			 yyint = strtol(yytext+2,NULL,16);
			 return(R_INTEGER);}
case 6:
{if (LexDebug)
			   {
			     printf("%s : %s\n", yytext, "(HEX)INTEGER");
			   }
			 yyint = strtol(yytext,NULL,16);
			 return(R_INTEGER);}
case 7:
{if (LexDebug)
			   {
			     printf("%s : %s\n", yytext, "(OCT)INTEGER");
			   }
			 yyint = strtol(yytext+2,NULL,8);
			 return(R_INTEGER);}
case 8:
{if (LexDebug)
			   {
			     printf("%s : %s\n", yytext, "(OCT)INTEGER");
			   }
			 yyint = strtol(yytext,NULL,8);
			 return(R_INTEGER);}
case 9:
{if (LexDebug)
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
case 10:
        {if (LexDebug)
			   {
			     printf("%s : %s\n", yytext, "LBRACKET");
			   }
			 return(R_LBRACKET);}
case 11:
        {if (LexDebug)
			   {
			     printf("%s : %s\n", yytext, "RBRACKET");
			   }
			 return(R_RBRACKET);}
case 12:
{if (LexDebug)
			   {
			     printf("%s : %s\n", yytext, "STRING");
			   }
			 return(R_STRING);}
case 13:
{CommentDepth++; BEGIN COMMENT;}
break;
case 14:
	{CommentDepth--;if(!CommentDepth) BEGIN NORMAL;}
break;
case 15:
  	  	{
		  	    /* None of the above rules applicable, so
			       it's a bad symbol. */
                              printf("Bad input char '%c' on line %d\n",
  	  	  	  	    yytext[0],
  	  	  	  	    yylineno);
  	  	  	}
break;
case 16:
	{}
break;
case -1:
break;
default:
fprintf(yyout,"bad switch yylook %d",nstr);
} return(0); }
/* end of yylex */

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

void equname(number,name)
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
	case R_ROLL:
	  NextVal = 2;
	  return(token);
	default:
	  return(token);
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

int yyvstop[] ={
0,

15,
0,

1,
15,
0,

1,
0,

15,
0,

15,
0,

15,
0,

15,
0,

15,
0,

4,
15,
0,

4,
15,
0,

4,
15,
0,

2,
15,
0,

2,
15,
0,

10,
15,
0,

11,
15,
0,

16,
0,

16,
0,

16,
0,

12,
0,

4,
0,

4,
0,

4,
0,

3,
0,

13,
0,

3,
0,

4,
0,

4,
0,

8,
0,

6,
0,

8,
0,

8,
0,

2,
0,

2,
0,

2,
0,

2,
6,
0,

14,
0,

12,
0,

9,
0,

3,
0,

3,
0,

7,
0,

5,
0,

3,
0,

3,
0,

3,
0,

3,
0,
0};
# define YYTYPE unsigned char
struct yywork { YYTYPE verify, advance; } yycrank[] ={
0,0,	0,0,	3,7,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	3,8,	3,9,	
0,0,	8,9,	8,9,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	3,10,	
8,9,	27,51,	0,0,	0,0,	
3,11,	28,52,	0,0,	0,0,	
3,12,	14,36,	0,0,	3,13,	
3,14,	3,15,	3,16,	3,16,	
3,16,	3,16,	3,16,	3,16,	
3,16,	3,17,	6,23,	23,50,	
0,0,	0,0,	0,0,	6,24,	
0,0,	0,0,	3,18,	3,18,	
0,0,	0,0,	3,18,	4,11,	
3,19,	3,19,	35,54,	0,0,	
0,0,	0,0,	4,13,	4,14,	
3,19,	4,16,	4,16,	4,16,	
4,16,	4,16,	4,16,	4,16,	
5,22,	3,19,	55,64,	0,0,	
3,20,	3,7,	3,21,	3,7,	
5,22,	5,22,	0,0,	10,25,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	35,54,	10,25,	
10,25,	13,35,	13,35,	13,35,	
13,35,	13,35,	13,35,	13,35,	
13,35,	13,35,	13,35,	0,0,	
0,0,	5,22,	55,64,	4,20,	
0,0,	4,21,	0,0,	0,0,	
0,0,	5,23,	5,22,	0,0,	
10,26,	0,0,	5,24,	5,22,	
0,0,	0,0,	0,0,	0,0,	
11,28,	10,25,	0,0,	5,22,	
0,0,	0,0,	10,25,	0,0,	
11,28,	11,28,	0,0,	0,0,	
5,22,	5,22,	10,25,	0,0,	
5,22,	0,0,	5,22,	5,22,	
0,0,	0,0,	0,0,	10,25,	
10,25,	0,0,	5,22,	10,25,	
0,0,	10,25,	10,25,	0,0,	
0,0,	11,28,	0,0,	5,22,	
0,0,	10,25,	0,0,	5,22,	
0,0,	5,22,	11,28,	0,0,	
0,0,	0,0,	10,25,	11,28,	
0,0,	0,0,	10,27,	0,0,	
10,25,	0,0,	0,0,	11,28,	
43,59,	43,59,	43,59,	43,59,	
43,59,	43,59,	43,59,	43,59,	
11,28,	11,28,	0,0,	0,0,	
11,28,	0,0,	11,28,	11,28,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	11,28,	0,0,	
29,28,	0,0,	0,0,	0,0,	
0,0,	12,30,	0,0,	11,28,	
0,0,	0,0,	0,0,	11,29,	
0,0,	11,28,	12,31,	12,32,	
12,32,	12,32,	12,32,	12,32,	
12,32,	12,32,	12,33,	12,33,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	12,34,	
12,34,	12,34,	12,34,	12,34,	
12,34,	0,0,	29,53,	29,53,	
29,53,	29,53,	29,53,	29,53,	
29,53,	29,53,	0,0,	0,0,	
0,0,	0,0,	15,37,	0,0,	
15,38,	15,38,	15,38,	15,38,	
15,38,	15,38,	15,38,	15,38,	
15,39,	15,39,	0,0,	12,34,	
12,34,	12,34,	12,34,	12,34,	
12,34,	15,34,	15,40,	15,40,	
15,34,	15,41,	15,34,	0,0,	
15,42,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	15,43,	
29,28,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	0,0,	
15,44,	61,28,	61,28,	61,28,	
61,28,	61,28,	61,28,	61,28,	
61,28,	15,34,	15,40,	15,40,	
15,34,	15,41,	15,34,	0,0,	
15,42,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	15,43,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	16,37,	
15,44,	16,38,	16,38,	16,38,	
16,38,	16,38,	16,38,	16,38,	
16,38,	16,39,	16,39,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	16,34,	16,40,	
16,40,	16,34,	16,41,	16,34,	
0,0,	16,42,	0,0,	0,0,	
0,0,	0,0,	0,0,	0,0,	
16,45,	34,34,	34,34,	34,34,	
34,34,	34,34,	34,34,	34,34,	
34,34,	34,34,	34,34,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	16,34,	16,40,	
16,40,	16,34,	16,41,	16,34,	
0,0,	16,42,	0,0,	0,0,	
0,0,	0,0,	0,0,	17,37,	
16,45,	17,39,	17,39,	17,39,	
17,39,	17,39,	17,39,	17,39,	
17,39,	17,39,	17,39,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	17,34,	17,34,	
17,34,	17,34,	17,41,	17,34,	
0,0,	17,42,	40,34,	40,34,	
40,34,	40,34,	40,34,	40,34,	
40,34,	40,34,	40,34,	40,34,	
57,67,	57,67,	57,67,	57,67,	
57,67,	57,67,	57,67,	57,67,	
57,67,	57,67,	0,0,	0,0,	
0,0,	0,0,	17,34,	17,34,	
17,34,	17,34,	17,41,	17,34,	
0,0,	17,42,	18,46,	18,46,	
18,46,	18,46,	18,46,	18,46,	
18,46,	18,46,	18,46,	18,46,	
18,47,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	18,46,	
18,46,	18,46,	18,46,	18,46,	
18,46,	18,48,	18,49,	18,48,	
18,48,	18,48,	18,48,	18,48,	
18,48,	18,48,	18,48,	18,48,	
18,48,	18,48,	18,48,	18,48,	
18,48,	18,48,	18,48,	18,48,	
18,48,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	18,46,	
18,46,	18,46,	18,46,	18,46,	
18,46,	18,48,	18,49,	18,48,	
18,48,	18,48,	18,48,	18,48,	
18,48,	18,48,	18,48,	18,48,	
18,48,	18,48,	18,48,	18,48,	
18,48,	18,48,	18,48,	18,48,	
18,48,	19,48,	19,48,	19,48,	
19,48,	19,48,	19,48,	19,48,	
19,48,	19,48,	19,48,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	19,48,	19,48,	
19,48,	19,48,	19,48,	19,48,	
0,0,	19,48,	0,0,	0,0,	
37,55,	37,55,	37,55,	37,55,	
37,55,	37,55,	37,55,	37,55,	
37,55,	37,55,	58,58,	58,58,	
58,58,	58,58,	58,58,	58,58,	
58,58,	58,58,	58,58,	58,58,	
0,0,	37,56,	19,48,	19,48,	
19,48,	19,48,	19,48,	19,48,	
0,0,	19,48,	31,32,	31,32,	
31,32,	31,32,	31,32,	31,32,	
31,32,	31,32,	31,33,	31,33,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	31,34,	
31,40,	31,40,	31,34,	31,34,	
31,34,	37,56,	31,42,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	31,43,	53,52,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	31,44,	53,61,	
53,61,	53,61,	53,61,	53,61,	
53,61,	53,61,	53,61,	31,34,	
31,40,	31,40,	31,34,	31,34,	
31,34,	0,0,	31,42,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	31,43,	0,0,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	31,44,	32,32,	
32,32,	32,32,	32,32,	32,32,	
32,32,	32,32,	32,32,	32,33,	
32,33,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	0,0,	
32,34,	32,40,	32,40,	32,34,	
32,34,	32,34,	0,0,	32,42,	
0,0,	0,0,	41,57,	0,0,	
41,57,	0,0,	32,45,	41,58,	
41,58,	41,58,	41,58,	41,58,	
41,58,	41,58,	41,58,	41,58,	
41,58,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	0,0,	
32,34,	32,40,	32,40,	32,34,	
32,34,	32,34,	0,0,	32,42,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	32,45,	33,33,	
33,33,	33,33,	33,33,	33,33,	
33,33,	33,33,	33,33,	33,33,	
33,33,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	0,0,	
33,34,	33,34,	33,34,	33,34,	
33,34,	33,34,	0,0,	33,42,	
44,60,	44,60,	44,60,	44,60,	
44,60,	44,60,	44,60,	44,60,	
44,60,	44,60,	0,0,	0,0,	
0,0,	0,0,	0,0,	0,0,	
0,0,	44,60,	44,60,	44,60,	
44,60,	44,60,	44,60,	0,0,	
33,34,	33,34,	33,34,	33,34,	
33,34,	33,34,	0,0,	33,42,	
0,0,	0,0,	0,0,	0,0,	
0,0,	0,0,	48,48,	48,48,	
48,48,	48,48,	48,48,	48,48,	
48,48,	48,48,	48,48,	48,48,	
0,0,	44,60,	44,60,	44,60,	
44,60,	44,60,	44,60,	48,48,	
48,48,	48,48,	48,48,	48,48,	
48,48,	54,62,	48,48,	54,62,	
0,0,	0,0,	54,63,	54,63,	
54,63,	54,63,	54,63,	54,63,	
54,63,	54,63,	54,63,	54,63,	
62,63,	62,63,	62,63,	62,63,	
62,63,	62,63,	62,63,	62,63,	
62,63,	62,63,	0,0,	48,48,	
48,48,	48,48,	48,48,	48,48,	
48,48,	0,0,	48,48,	49,48,	
49,48,	49,48,	49,48,	49,48,	
49,48,	49,48,	49,48,	49,48,	
49,48,	0,0,	0,0,	0,0,	
0,0,	0,0,	0,0,	0,0,	
49,48,	49,48,	49,48,	49,48,	
49,48,	49,48,	56,65,	49,48,	
56,65,	0,0,	0,0,	56,66,	
56,66,	56,66,	56,66,	56,66,	
56,66,	56,66,	56,66,	56,66,	
56,66,	65,66,	65,66,	65,66,	
65,66,	65,66,	65,66,	65,66,	
65,66,	65,66,	65,66,	0,0,	
49,48,	49,48,	49,48,	49,48,	
49,48,	49,48,	64,68,	49,48,	
64,68,	0,0,	0,0,	64,69,	
64,69,	64,69,	64,69,	64,69,	
64,69,	64,69,	64,69,	64,69,	
64,69,	68,69,	68,69,	68,69,	
68,69,	68,69,	68,69,	68,69,	
68,69,	68,69,	68,69,	0,0,	
0,0};
struct yysvf yysvec[] ={
0,	0,	0,
yycrank+0,	0,		0,	
yycrank+0,	0,		0,	
yycrank+-1,	0,		0,	
yycrank+-32,	yysvec+3,	0,	
yycrank+-87,	0,		0,	
yycrank+-16,	yysvec+5,	0,	
yycrank+0,	0,		yyvstop+1,
yycrank+4,	0,		yyvstop+3,
yycrank+0,	yysvec+8,	yyvstop+6,
yycrank+-98,	0,		yyvstop+8,
yycrank+-139,	0,		yyvstop+10,
yycrank+186,	0,		yyvstop+12,
yycrank+61,	0,		yyvstop+14,
yycrank+3,	0,		yyvstop+16,
yycrank+224,	0,		yyvstop+18,
yycrank+297,	0,		yyvstop+21,
yycrank+361,	0,		yyvstop+24,
yycrank+418,	0,		yyvstop+27,
yycrank+493,	yysvec+18,	yyvstop+30,
yycrank+0,	0,		yyvstop+33,
yycrank+0,	0,		yyvstop+36,
yycrank+0,	0,		yyvstop+39,
yycrank+12,	0,		yyvstop+41,
yycrank+0,	yysvec+14,	yyvstop+43,
yycrank+0,	yysvec+10,	0,	
yycrank+0,	0,		yyvstop+45,
yycrank+-3,	yysvec+10,	0,	
yycrank+2,	0,		0,	
yycrank+210,	0,		0,	
yycrank+0,	yysvec+11,	0,	
yycrank+550,	0,		yyvstop+47,
yycrank+623,	0,		yyvstop+49,
yycrank+687,	0,		yyvstop+51,
yycrank+329,	yysvec+33,	0,	
yycrank+5,	yysvec+13,	yyvstop+53,
yycrank+0,	0,		yyvstop+55,
yycrank+520,	0,		yyvstop+57,
yycrank+0,	yysvec+16,	yyvstop+59,
yycrank+0,	yysvec+17,	yyvstop+61,
yycrank+386,	yysvec+33,	yyvstop+63,
yycrank+655,	yysvec+33,	0,	
yycrank+0,	0,		yyvstop+65,
yycrank+148,	0,		yyvstop+67,
yycrank+712,	0,		0,	
yycrank+0,	0,		yyvstop+69,
yycrank+0,	yysvec+18,	yyvstop+71,
yycrank+0,	0,		yyvstop+73,
yycrank+750,	yysvec+18,	yyvstop+75,
yycrank+807,	yysvec+18,	yyvstop+77,
yycrank+0,	0,		yyvstop+80,
yycrank+0,	yysvec+10,	yyvstop+82,
yycrank+0,	0,		yyvstop+84,
yycrank+591,	0,		0,	
yycrank+778,	0,		0,	
yycrank+21,	yysvec+37,	yyvstop+86,
yycrank+835,	0,		0,	
yycrank+396,	0,		0,	
yycrank+530,	yysvec+33,	yyvstop+88,
yycrank+0,	yysvec+43,	yyvstop+90,
yycrank+0,	yysvec+44,	yyvstop+92,
yycrank+265,	yysvec+53,	0,	
yycrank+788,	0,		0,	
yycrank+0,	yysvec+62,	yyvstop+94,
yycrank+867,	0,		0,	
yycrank+845,	0,		0,	
yycrank+0,	yysvec+65,	yyvstop+96,
yycrank+0,	yysvec+57,	yyvstop+98,
yycrank+877,	0,		0,	
yycrank+0,	yysvec+68,	yyvstop+100,
0,	0,	0};
struct yywork *yytop = yycrank+934;
struct yysvf *yybgin = yysvec+1;
char yymatch[] ={
00  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
01  ,011 ,012 ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
011 ,01  ,'"' ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,'+' ,01  ,'+' ,01  ,01  ,
'0' ,'0' ,'0' ,'0' ,'0' ,'0' ,'0' ,'0' ,
'8' ,'8' ,01  ,01  ,01  ,01  ,01  ,01  ,
01  ,'A' ,'B' ,'B' ,'A' ,'E' ,'A' ,'G' ,
'H' ,'G' ,'G' ,'G' ,'G' ,'G' ,'G' ,'O' ,
'G' ,'G' ,'G' ,'G' ,'G' ,'G' ,'G' ,'G' ,
'X' ,'G' ,'G' ,01  ,0134,01  ,'^' ,01  ,
01  ,'A' ,'B' ,'B' ,'A' ,'E' ,'A' ,'G' ,
'H' ,'G' ,'G' ,'G' ,'G' ,'G' ,'G' ,'O' ,
'G' ,'G' ,'G' ,'G' ,'G' ,'G' ,'G' ,'G' ,
'X' ,'G' ,'G' ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
01  ,01  ,01  ,01  ,01  ,01  ,01  ,01  ,
0};
char yyextra[] ={
0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,
0};
/*	ncform	4.1	83/08/11	*/

int yylineno =1;
# define YYU(x) x
# define NLSTATE yyprevious=YYNEWLINE
char yytext[YYLMAX];
struct yysvf *yylstate [YYLMAX], **yylsp, **yyolsp;
char yysbuf[YYLMAX];
char *yysptr = yysbuf;
int *yyfnd;
extern struct yysvf *yyestate;
int yyprevious = YYNEWLINE;
yylook(){
	register struct yysvf *yystate, **lsp;
	register struct yywork *yyt;
	struct yysvf *yyz;
	int yych;
	struct yywork *yyr;
# ifdef LEXDEBUG
	int debug;
# endif
	char *yylastch;
	/* start off machines */
# ifdef LEXDEBUG
	debug = 0;
# endif
	if (!yymorfg)
		yylastch = yytext;
	else {
		yymorfg=0;
		yylastch = yytext+yyleng;
		}
	for(;;){
		lsp = yylstate;
		yyestate = yystate = yybgin;
		if (yyprevious==YYNEWLINE) yystate++;
		for (;;){
# ifdef LEXDEBUG
			if(debug)fprintf(yyout,"state %d\n",yystate-yysvec-1);
# endif
			yyt = yystate->yystoff;
			if(yyt == yycrank){		/* may not be any transitions */
				yyz = yystate->yyother;
				if(yyz == 0)break;
				if(yyz->yystoff == yycrank)break;
				}
			*yylastch++ = yych = input();
		tryagain:
# ifdef LEXDEBUG
			if(debug){
				fprintf(yyout,"unsigned char ");
				allprint(yych);
				putchar('\n');
				}
# endif
			yyr = yyt;
			if ( (int)yyt > (int)yycrank){
				yyt = yyr + yych;
				if (yyt <= yytop && yyt->verify+yysvec == yystate){
					if(yyt->advance+yysvec == YYLERR)	/* error transitions */
						{unput(*--yylastch);break;}
					*lsp++ = yystate = yyt->advance+yysvec;
					goto contin;
					}
				}
# ifdef YYOPTIM
			else if((int)yyt < (int)yycrank) {		/* r < yycrank */
				yyt = yyr = yycrank+(yycrank-yyt);
# ifdef LEXDEBUG
				if(debug)fprintf(yyout,"compressed state\n");
# endif
				yyt = yyt + yych;
				if(yyt <= yytop && yyt->verify+yysvec == yystate){
					if(yyt->advance+yysvec == YYLERR)	/* error transitions */
						{unput(*--yylastch);break;}
					*lsp++ = yystate = yyt->advance+yysvec;
					goto contin;
					}
				yyt = yyr + YYU(yymatch[yych]);
# ifdef LEXDEBUG
				if(debug){
					fprintf(yyout,"try fall back character ");
					allprint(YYU(yymatch[yych]));
					putchar('\n');
					}
# endif
				if(yyt <= yytop && yyt->verify+yysvec == yystate){
					if(yyt->advance+yysvec == YYLERR)	/* error transition */
						{unput(*--yylastch);break;}
					*lsp++ = yystate = yyt->advance+yysvec;
					goto contin;
					}
				}
			if ((yystate = yystate->yyother) && (yyt= yystate->yystoff) != yycrank){
# ifdef LEXDEBUG
				if(debug)fprintf(yyout,"fall back to state %d\n",yystate-yysvec-1);
# endif
				goto tryagain;
				}
# endif
			else
				{unput(*--yylastch);break;}
		contin:
# ifdef LEXDEBUG
			if(debug){
				fprintf(yyout,"state %d char ",yystate-yysvec-1);
				allprint(yych);
				putchar('\n');
				}
# endif
			;
			}
# ifdef LEXDEBUG
		if(debug){
			fprintf(yyout,"stopped at %d with ",*(lsp-1)-yysvec-1);
			allprint(yych);
			putchar('\n');
			}
# endif
		while (lsp-- > yylstate){
			*yylastch-- = 0;
			if (*lsp != 0 && (yyfnd= (*lsp)->yystops) && *yyfnd > 0){
				yyolsp = lsp;
				if(yyextra[*yyfnd]){		/* must backup */
					while(yyback((*lsp)->yystops,-*yyfnd) != 1 && lsp > yylstate){
						lsp--;
						unput(*yylastch--);
						}
					}
				yyprevious = YYU(*yylastch);
				yylsp = lsp;
				yyleng = yylastch-yytext+1;
				yytext[yyleng] = 0;
# ifdef LEXDEBUG
				if(debug){
					fprintf(yyout,"\nmatch ");
					sprint(yytext);
					fprintf(yyout," action %d\n",*yyfnd);
					}
# endif
				return(*yyfnd++);
				}
			unput(*yylastch);
			}
		if (yytext[0] == 0  /* && feof(yyin) */)
			{
			yysptr=yysbuf;
			return(0);
			}
		yyprevious = yytext[0] = input();
		if (yyprevious>0)
			output(yyprevious);
		yylastch=yytext;
# ifdef LEXDEBUG
		if(debug)putchar('\n');
# endif
		}
	}
yyback(p, m)
	int *p;
{
if (p==0) return(0);
while (*p)
	{
	if (*p++ == m)
		return(1);
	}
return(0);
}
	/* the following are only used in the lex library */
yyinput(){
	return(input());
	}
void yyoutput(c)
     int c; {
       output(c);
}
void yyunput(c)
     int c; {
       unput(c);
}
