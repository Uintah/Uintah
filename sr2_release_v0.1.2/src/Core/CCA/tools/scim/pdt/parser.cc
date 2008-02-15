/* A Bison parser, made by GNU Bison 1.875.  */

/* Skeleton parser for Yacc-like parsing with Bison,
   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  */

/* As a special exception, when this file is copied by Bison into a
   Bison output file, you may use that output file without restriction.
   This special exception was added by the Free Software Foundation
   in version 1.24 of Bison.  */

/* Written by Richard Stallman by simplifying the original so called
   ``semantic'' parser.  */

/* All symbols defined below should begin with pdt or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0



/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum pdttokentype {
     IDENTIFIER = 258,
     OTHERLINE = 259,
     MAP = 260,
     ININTERFACE = 261,
     OUTINTERFACE = 262,
     CLOSEININTERFACE = 263,
     CLOSEOUTINTERFACE = 264,
     INOUTREMAP = 265,
     INOUTOMIT = 266,
     ARROW = 267,
     DBLANGLE = 268
   };
#endif
#define IDENTIFIER 258
#define OTHERLINE 259
#define MAP 260
#define ININTERFACE 261
#define OUTINTERFACE 262
#define CLOSEININTERFACE 263
#define CLOSEOUTINTERFACE 264
#define INOUTREMAP 265
#define INOUTOMIT 266
#define ARROW 267
#define DBLANGLE 268




/* Copy the first part of user declarations.  */
#line 31 "pdt/parser.y"


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



/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

#if ! defined (YYSTYPE) && ! defined (YYSTYPE_IS_DECLARED)
#line 115 "pdt/parser.y"
typedef union YYSTYPE {
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
} YYSTYPE;
/* Line 191 of yacc.c.  */
#line 200 "y.tab.c"
# define pdtstype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 214 of yacc.c.  */
#line 212 "y.tab.c"

#if ! defined (pdtoverflow) || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# if YYSTACK_USE_ALLOCA
#  define YYSTACK_ALLOC alloca
# else
#  ifndef YYSTACK_USE_ALLOCA
#   if defined (alloca) || defined (_ALLOCA_H)
#    define YYSTACK_ALLOC alloca
#   else
#    ifdef __GNUC__
#     define YYSTACK_ALLOC __builtin_alloca
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning. */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
# else
#  if defined (__STDC__) || defined (__cplusplus)
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   define YYSIZE_T size_t
#  endif
#  define YYSTACK_ALLOC malloc
#  define YYSTACK_FREE free
# endif
#endif /* ! defined (pdtoverflow) || YYERROR_VERBOSE */


#if (! defined (pdtoverflow) \
     && (! defined (__cplusplus) \
	 || (YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union pdtalloc
{
  short pdtss;
  YYSTYPE pdtvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union pdtalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (short) + sizeof (YYSTYPE))				\
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  register YYSIZE_T pdti;		\
	  for (pdti = 0; pdti < (Count); pdti++)	\
	    (To)[pdti] = (From)[pdti];		\
	}					\
      while (0)
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T pdtnewbytes;						\
	YYCOPY (&pdtptr->Stack, Stack, pdtsize);				\
	Stack = &pdtptr->Stack;						\
	pdtnewbytes = pdtstacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	pdtptr += pdtnewbytes / sizeof (*pdtptr);				\
      }									\
    while (0)

#endif

#if defined (__STDC__) || defined (__cplusplus)
   typedef signed char pdtsigned_char;
#else
   typedef short pdtsigned_char;
#endif

/* YYFINAL -- State number of the termination state. */
#define YYFINAL  3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   28

/* YYNTOKENS -- Number of terminals. */
#define YYNTOKENS  15
/* YYNNTS -- Number of nonterminals. */
#define YYNNTS  11
/* YYNRULES -- Number of rules. */
#define YYNRULES  23
/* YYNRULES -- Number of states. */
#define YYNSTATES  37

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   268

#define YYTRANSLATE(YYX) 						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? pdttranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const unsigned char pdttranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,    14,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const unsigned char pdtprhs[] =
{
       0,     0,     3,     5,     6,     9,    11,    13,    14,    17,
      19,    25,    28,    32,    36,    38,    41,    44,    45,    48,
      54,    58,    59,    62
};

/* YYRHS -- A `-1'-separated list of the rules' RHS. */
static const pdtsigned_char pdtrhs[] =
{
      16,     0,    -1,    17,    -1,    -1,    17,    18,    -1,    21,
      -1,    20,    -1,    -1,    19,    20,    -1,     4,    -1,     5,
       3,    12,     3,    22,    -1,     5,    22,    -1,     6,    19,
       8,    -1,     7,    19,     9,    -1,    11,    -1,    11,     3,
      -1,    10,     3,    -1,    -1,    22,    23,    -1,    14,     3,
      12,     3,    24,    -1,    14,     3,    24,    -1,    -1,    24,
      25,    -1,    13,     3,    12,     3,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const unsigned short pdtrline[] =
{
       0,   151,   151,   159,   162,   168,   172,   183,   187,   194,
     200,   206,   212,   218,   224,   229,   234,   242,   246,   254,
     260,   269,   273,   281
};
#endif

#if YYDEBUG || YYERROR_VERBOSE
/* YYTNME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals. */
static const char *const pdttname[] =
{
  "$end", "error", "$undefined", "IDENTIFIER", "OTHERLINE", "MAP", 
  "ININTERFACE", "OUTINTERFACE", "CLOSEININTERFACE", "CLOSEOUTINTERFACE", 
  "INOUTREMAP", "INOUTOMIT", "ARROW", "DBLANGLE", "'>'", "$accept", 
  "specification", "mainline_star", "mainline", "otherline_star", 
  "otherline", "command", "namemap_star", "namemap", "subnamemap_star", 
  "subnamemap", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const unsigned short pdttoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,    62
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const unsigned char pdtr1[] =
{
       0,    15,    16,    17,    17,    18,    18,    19,    19,    20,
      21,    21,    21,    21,    21,    21,    21,    22,    22,    23,
      23,    24,    24,    25
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const unsigned char pdtr2[] =
{
       0,     2,     1,     0,     2,     1,     1,     0,     2,     1,
       5,     2,     3,     3,     1,     2,     2,     0,     2,     5,
       3,     0,     2,     4
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const unsigned char pdtdefact[] =
{
       3,     0,     2,     1,     9,    17,     7,     7,     0,    14,
       4,     6,     5,     0,    11,     0,     0,    16,    15,     0,
       0,    18,    12,     8,    13,    17,    21,    10,     0,    20,
      21,     0,    22,    19,     0,     0,    23
};

/* YYDEFGOTO[NTERM-NUM]. */
static const pdtsigned_char pdtdefgoto[] =
{
      -1,     1,     2,    10,    15,    23,    12,    14,    21,    29,
      32
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -6
static const pdtsigned_char pdtpact[] =
{
      -6,     5,    -4,    -6,    -6,     7,    -6,    -6,     8,    10,
      -6,    -6,    -6,     2,     1,     4,     0,    -6,    -6,    13,
      14,    -6,    -6,    -6,    -6,    -6,     6,     1,    16,     9,
      -6,    17,    -6,     9,    11,    18,    -6
};

/* YYPGOTO[NTERM-NUM].  */
static const pdtsigned_char pdtpgoto[] =
{
      -6,    -6,    -6,    -6,    19,    22,    -6,     3,    -6,    -5,
      -6
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const unsigned char pdttable[] =
{
       4,     5,     6,     7,     4,     3,     8,     9,     4,    24,
      13,    17,    22,    18,    19,    20,    25,    26,    28,    30,
      34,    36,    31,    35,    11,    33,    16,     0,    27
};

static const pdtsigned_char pdtcheck[] =
{
       4,     5,     6,     7,     4,     0,    10,    11,     4,     9,
       3,     3,     8,     3,    12,    14,     3,     3,    12,     3,
       3,     3,    13,    12,     2,    30,     7,    -1,    25
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const unsigned char pdtstos[] =
{
       0,    16,    17,     0,     4,     5,     6,     7,    10,    11,
      18,    20,    21,     3,    22,    19,    19,     3,     3,    12,
      14,    23,     8,    20,     9,     3,     3,    22,    12,    24,
       3,    13,    25,    24,     3,    12,     3
};

#if ! defined (YYSIZE_T) && defined (__SIZE_TYPE__)
# define YYSIZE_T __SIZE_TYPE__
#endif
#if ! defined (YYSIZE_T) && defined (size_t)
# define YYSIZE_T size_t
#endif
#if ! defined (YYSIZE_T)
# if defined (__STDC__) || defined (__cplusplus)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# endif
#endif
#if ! defined (YYSIZE_T)
# define YYSIZE_T unsigned int
#endif

#define pdterrok		(pdterrstatus = 0)
#define pdtclearin	(pdtchar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto pdtacceptlab
#define YYABORT		goto pdtabortlab
#define YYERROR		goto pdterrlab1


/* Like YYERROR except do call pdterror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto pdterrlab

#define YYRECOVERING()  (!!pdterrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (pdtchar == YYEMPTY && pdtlen == 1)				\
    {								\
      pdtchar = (Token);						\
      pdtlval = (Value);						\
      pdttoken = YYTRANSLATE (pdtchar);				\
      YYPOPSTACK;						\
      goto pdtbackup;						\
    }								\
  else								\
    { 								\
      pdterror ("syntax error: cannot back up");\
      YYERROR;							\
    }								\
while (0)

#define YYTERROR	1
#define YYERRCODE	256

/* YYLLOC_DEFAULT -- Compute the default location (before the actions
   are run).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)         \
  Current.first_line   = Rhs[1].first_line;      \
  Current.first_column = Rhs[1].first_column;    \
  Current.last_line    = Rhs[N].last_line;       \
  Current.last_column  = Rhs[N].last_column;
#endif

/* YYLEX -- calling `pdtlex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX pdtlex (YYLEX_PARAM)
#else
# define YYLEX pdtlex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (pdtdebug)					\
    YYFPRINTF Args;				\
} while (0)

# define YYDSYMPRINT(Args)			\
do {						\
  if (pdtdebug)					\
    pdtsymprint Args;				\
} while (0)

# define YYDSYMPRINTF(Title, Token, Value, Location)		\
do {								\
  if (pdtdebug)							\
    {								\
      YYFPRINTF (stderr, "%s ", Title);				\
      pdtsymprint (stderr, 					\
                  Token, Value);	\
      YYFPRINTF (stderr, "\n");					\
    }								\
} while (0)

/*------------------------------------------------------------------.
| pdt_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (cinluded).                                                   |
`------------------------------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
pdt_stack_print (short *bottom, short *top)
#else
static void
pdt_stack_print (bottom, top)
    short *bottom;
    short *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (/* Nothing. */; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (pdtdebug)							\
    pdt_stack_print ((Bottom), (Top));				\
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
pdt_reduce_print (int pdtrule)
#else
static void
pdt_reduce_print (pdtrule)
    int pdtrule;
#endif
{
  int pdti;
  unsigned int pdtlineno = pdtrline[pdtrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %u), ",
             pdtrule - 1, pdtlineno);
  /* Print the symbols being reduced, and their result.  */
  for (pdti = pdtprhs[pdtrule]; 0 <= pdtrhs[pdti]; pdti++)
    YYFPRINTF (stderr, "%s ", pdttname [pdtrhs[pdti]]);
  YYFPRINTF (stderr, "-> %s\n", pdttname [pdtr1[pdtrule]]);
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (pdtdebug)				\
    pdt_reduce_print (Rule);		\
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int pdtdebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YYDSYMPRINT(Args)
# define YYDSYMPRINTF(Title, Token, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   SIZE_MAX < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#if YYMAXDEPTH == 0
# undef YYMAXDEPTH
#endif

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef pdtstrlen
#  if defined (__GLIBC__) && defined (_STRING_H)
#   define pdtstrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
#   if defined (__STDC__) || defined (__cplusplus)
pdtstrlen (const char *pdtstr)
#   else
pdtstrlen (pdtstr)
     const char *pdtstr;
#   endif
{
  register const char *pdts = pdtstr;

  while (*pdts++ != '\0')
    continue;

  return pdts - pdtstr - 1;
}
#  endif
# endif

# ifndef pdtstpcpy
#  if defined (__GLIBC__) && defined (_STRING_H) && defined (_GNU_SOURCE)
#   define pdtstpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
#   if defined (__STDC__) || defined (__cplusplus)
pdtstpcpy (char *pdtdest, const char *pdtsrc)
#   else
pdtstpcpy (pdtdest, pdtsrc)
     char *pdtdest;
     const char *pdtsrc;
#   endif
{
  register char *pdtd = pdtdest;
  register const char *pdts = pdtsrc;

  while ((*pdtd++ = *pdts++) != '\0')
    continue;

  return pdtd - 1;
}
#  endif
# endif

#endif /* !YYERROR_VERBOSE */



#if YYDEBUG
/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
pdtsymprint (FILE *pdtoutput, int pdttype, YYSTYPE *pdtvaluep)
#else
static void
pdtsymprint (pdtoutput, pdttype, pdtvaluep)
    FILE *pdtoutput;
    int pdttype;
    YYSTYPE *pdtvaluep;
#endif
{
  /* Pacify ``unused variable'' warnings.  */
  (void) pdtvaluep;

  if (pdttype < YYNTOKENS)
    {
      YYFPRINTF (pdtoutput, "token %s (", pdttname[pdttype]);
# ifdef YYPRINT
      YYPRINT (pdtoutput, pdttoknum[pdttype], *pdtvaluep);
# endif
    }
  else
    YYFPRINTF (pdtoutput, "nterm %s (", pdttname[pdttype]);

  switch (pdttype)
    {
      default:
        break;
    }
  YYFPRINTF (pdtoutput, ")");
}

#endif /* ! YYDEBUG */
/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

#if defined (__STDC__) || defined (__cplusplus)
static void
pdtdestruct (int pdttype, YYSTYPE *pdtvaluep)
#else
static void
pdtdestruct (pdttype, pdtvaluep)
    int pdttype;
    YYSTYPE *pdtvaluep;
#endif
{
  /* Pacify ``unused variable'' warnings.  */
  (void) pdtvaluep;

  switch (pdttype)
    {

      default:
        break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
# if defined (__STDC__) || defined (__cplusplus)
int pdtparse (void *YYPARSE_PARAM);
# else
int pdtparse ();
# endif
#else /* ! YYPARSE_PARAM */
#if defined (__STDC__) || defined (__cplusplus)
int pdtparse (void);
#else
int pdtparse ();
#endif
#endif /* ! YYPARSE_PARAM */



/* The lookahead symbol.  */
int pdtchar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE pdtlval;

/* Number of syntax errors so far.  */
int pdtnerrs;



/*----------.
| pdtparse.  |
`----------*/

#ifdef YYPARSE_PARAM
# if defined (__STDC__) || defined (__cplusplus)
int pdtparse (void *YYPARSE_PARAM)
# else
int pdtparse (YYPARSE_PARAM)
  void *YYPARSE_PARAM;
# endif
#else /* ! YYPARSE_PARAM */
#if defined (__STDC__) || defined (__cplusplus)
int
pdtparse (void)
#else
int
pdtparse ()

#endif
#endif
{
  
  register int pdtstate;
  register int pdtn;
  int pdtresult;
  /* Number of tokens to shift before error messages enabled.  */
  int pdterrstatus;
  /* Lookahead token as an internal (translated) token number.  */
  int pdttoken = 0;

  /* Three stacks and their tools:
     `pdtss': related to states,
     `pdtvs': related to semantic values,
     `pdtls': related to locations.

     Refer to the stacks thru separate pointers, to allow pdtoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  short	pdtssa[YYINITDEPTH];
  short *pdtss = pdtssa;
  register short *pdtssp;

  /* The semantic value stack.  */
  YYSTYPE pdtvsa[YYINITDEPTH];
  YYSTYPE *pdtvs = pdtvsa;
  register YYSTYPE *pdtvsp;



#define YYPOPSTACK   (pdtvsp--, pdtssp--)

  YYSIZE_T pdtstacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE pdtval;


  /* When reducing, the number of symbols on the RHS of the reduced
     rule.  */
  int pdtlen;

  YYDPRINTF ((stderr, "Starting parse\n"));

  pdtstate = 0;
  pdterrstatus = 0;
  pdtnerrs = 0;
  pdtchar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  pdtssp = pdtss;
  pdtvsp = pdtvs;

  goto pdtsetstate;

/*------------------------------------------------------------.
| pdtnewstate -- Push a new state, which is found in pdtstate.  |
`------------------------------------------------------------*/
 pdtnewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed. so pushing a state here evens the stacks.
     */
  pdtssp++;

 pdtsetstate:
  *pdtssp = pdtstate;

  if (pdtss + pdtstacksize - 1 <= pdtssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T pdtsize = pdtssp - pdtss + 1;

#ifdef pdtoverflow
      {
	/* Give user a chance to reallocate the stack. Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *pdtvs1 = pdtvs;
	short *pdtss1 = pdtss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if pdtoverflow is a macro.  */
	pdtoverflow ("parser stack overflow",
		    &pdtss1, pdtsize * sizeof (*pdtssp),
		    &pdtvs1, pdtsize * sizeof (*pdtvsp),

		    &pdtstacksize);

	pdtss = pdtss1;
	pdtvs = pdtvs1;
      }
#else /* no pdtoverflow */
# ifndef YYSTACK_RELOCATE
      goto pdtoverflowlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= pdtstacksize)
	goto pdtoverflowlab;
      pdtstacksize *= 2;
      if (YYMAXDEPTH < pdtstacksize)
	pdtstacksize = YYMAXDEPTH;

      {
	short *pdtss1 = pdtss;
	union pdtalloc *pdtptr =
	  (union pdtalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (pdtstacksize));
	if (! pdtptr)
	  goto pdtoverflowlab;
	YYSTACK_RELOCATE (pdtss);
	YYSTACK_RELOCATE (pdtvs);

#  undef YYSTACK_RELOCATE
	if (pdtss1 != pdtssa)
	  YYSTACK_FREE (pdtss1);
      }
# endif
#endif /* no pdtoverflow */

      pdtssp = pdtss + pdtsize - 1;
      pdtvsp = pdtvs + pdtsize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) pdtstacksize));

      if (pdtss + pdtstacksize - 1 <= pdtssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", pdtstate));

  goto pdtbackup;

/*-----------.
| pdtbackup.  |
`-----------*/
pdtbackup:

/* Do appropriate processing given the current state.  */
/* Read a lookahead token if we need one and don't already have one.  */
/* pdtresume: */

  /* First try to decide what to do without reference to lookahead token.  */

  pdtn = pdtpact[pdtstate];
  if (pdtn == YYPACT_NINF)
    goto pdtdefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (pdtchar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      pdtchar = YYLEX;
    }

  if (pdtchar <= YYEOF)
    {
      pdtchar = pdttoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      pdttoken = YYTRANSLATE (pdtchar);
      YYDSYMPRINTF ("Next token is", pdttoken, &pdtlval, &pdtlloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  pdtn += pdttoken;
  if (pdtn < 0 || YYLAST < pdtn || pdtcheck[pdtn] != pdttoken)
    goto pdtdefault;
  pdtn = pdttable[pdtn];
  if (pdtn <= 0)
    {
      if (pdtn == 0 || pdtn == YYTABLE_NINF)
	goto pdterrlab;
      pdtn = -pdtn;
      goto pdtreduce;
    }

  if (pdtn == YYFINAL)
    YYACCEPT;

  /* Shift the lookahead token.  */
  YYDPRINTF ((stderr, "Shifting token %s, ", pdttname[pdttoken]));

  /* Discard the token being shifted unless it is eof.  */
  if (pdtchar != YYEOF)
    pdtchar = YYEMPTY;

  *++pdtvsp = pdtlval;


  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (pdterrstatus)
    pdterrstatus--;

  pdtstate = pdtn;
  goto pdtnewstate;


/*-----------------------------------------------------------.
| pdtdefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
pdtdefault:
  pdtn = pdtdefact[pdtstate];
  if (pdtn == 0)
    goto pdterrlab;
  goto pdtreduce;


/*-----------------------------.
| pdtreduce -- Do a reduction.  |
`-----------------------------*/
pdtreduce:
  /* pdtn is the number of a rule to reduce with.  */
  pdtlen = pdtr2[pdtn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  pdtval = pdtvsp[1-pdtlen];


  YY_REDUCE_PRINT (pdtn);
  switch (pdtn)
    {
        case 2:
#line 152 "pdt/parser.y"
    {
                 parseToIr("in");
                 parseToIr("out");
		 parseToIr("inout"); 
               }
    break;

  case 3:
#line 159 "pdt/parser.y"
    {
	       }
    break;

  case 4:
#line 163 "pdt/parser.y"
    {
	       }
    break;

  case 5:
#line 169 "pdt/parser.y"
    {
          }
    break;

  case 6:
#line 173 "pdt/parser.y"
    {
            inoutCodeR.append(pdtvsp[0].oline);
            inoutCodeR.append("\n");
          }
    break;

  case 7:
#line 183 "pdt/parser.y"
    {
                pdtval.text = new string();  
              }
    break;

  case 8:
#line 188 "pdt/parser.y"
    {
                pdtvsp[-1].text->append(pdtvsp[0].oline);
                pdtvsp[-1].text->append("\n");
              }
    break;

  case 9:
#line 195 "pdt/parser.y"
    {
           pdtval.oline = pdtvsp[0].oline;
         }
    break;

  case 10:
#line 201 "pdt/parser.y"
    {
           IrMap* map = new IrMap(pdtvsp[-3].ident,pdtvsp[-1].ident,pdtvsp[0].irNM_list,ir);
           ir->addMap(map);
         }
    break;

  case 11:
#line 207 "pdt/parser.y"
    {
           IrMap* map = new IrMap(pdtvsp[0].irNM_list,NULL);
           ir->setForAllMap(map);
         }
    break;

  case 12:
#line 213 "pdt/parser.y"
    {
	   inCodeR.append(pdtvsp[-1].text->c_str());
           inCodeR.append("\n");
	 }
    break;

  case 13:
#line 219 "pdt/parser.y"
    {
           outCodeR.append(pdtvsp[-1].text->c_str());
           outCodeR.append("\n");
         }
    break;

  case 14:
#line 225 "pdt/parser.y"
    {
           ir->omitInOut();
         }
    break;

  case 15:
#line 230 "pdt/parser.y"
    {
           ir->omitInOut(pdtvsp[0].ident); 
         }
    break;

  case 16:
#line 235 "pdt/parser.y"
    {
           ir->remapInOut(pdtvsp[0].ident);
         }
    break;

  case 17:
#line 242 "pdt/parser.y"
    {
                pdtval.irNM_list = new IrNameMapList();
              }
    break;

  case 18:
#line 247 "pdt/parser.y"
    {
                pdtval.irNM_list = pdtvsp[-1].irNM_list;
                if(pdtvsp[0].irNM) pdtvsp[-1].irNM_list->addNameMap(pdtvsp[0].irNM);
              }
    break;

  case 19:
#line 255 "pdt/parser.y"
    {
           pdtval.irNM = new IrNameMap(pdtvsp[-3].ident,pdtvsp[-1].ident);
	   if(pdtvsp[0].irNM_list) pdtval.irNM->addSubList(pdtvsp[0].irNM_list);
         }
    break;

  case 20:
#line 261 "pdt/parser.y"
    {
           pdtval.irNM = new IrNameMap(pdtvsp[-1].ident,pdtvsp[-1].ident);
           if(pdtvsp[0].irNM_list) pdtval.irNM->addSubList(pdtvsp[0].irNM_list);
         }
    break;

  case 21:
#line 269 "pdt/parser.y"
    {
                   pdtval.irNM_list = new IrNameMapList();
                 }
    break;

  case 22:
#line 274 "pdt/parser.y"
    {
                   pdtval.irNM_list = pdtvsp[-1].irNM_list;
                   if(pdtvsp[0].irNM) pdtvsp[-1].irNM_list->addNameMap(pdtvsp[0].irNM);
                 }
    break;

  case 23:
#line 282 "pdt/parser.y"
    {
              pdtval.irNM = new IrNameMap(pdtvsp[-2].ident,pdtvsp[0].ident);
            }
    break;


    }

/* Line 999 of yacc.c.  */
#line 1271 "y.tab.c"

  pdtvsp -= pdtlen;
  pdtssp -= pdtlen;


  YY_STACK_PRINT (pdtss, pdtssp);

  *++pdtvsp = pdtval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  pdtn = pdtr1[pdtn];

  pdtstate = pdtpgoto[pdtn - YYNTOKENS] + *pdtssp;
  if (0 <= pdtstate && pdtstate <= YYLAST && pdtcheck[pdtstate] == *pdtssp)
    pdtstate = pdttable[pdtstate];
  else
    pdtstate = pdtdefgoto[pdtn - YYNTOKENS];

  goto pdtnewstate;


/*------------------------------------.
| pdterrlab -- here on detecting error |
`------------------------------------*/
pdterrlab:
  /* If not already recovering from an error, report this error.  */
  if (!pdterrstatus)
    {
      ++pdtnerrs;
#if YYERROR_VERBOSE
      pdtn = pdtpact[pdtstate];

      if (YYPACT_NINF < pdtn && pdtn < YYLAST)
	{
	  YYSIZE_T pdtsize = 0;
	  int pdttype = YYTRANSLATE (pdtchar);
	  char *pdtmsg;
	  int pdtx, pdtcount;

	  pdtcount = 0;
	  /* Start YYX at -YYN if negative to avoid negative indexes in
	     YYCHECK.  */
	  for (pdtx = pdtn < 0 ? -pdtn : 0;
	       pdtx < (int) (sizeof (pdttname) / sizeof (char *)); pdtx++)
	    if (pdtcheck[pdtx + pdtn] == pdtx && pdtx != YYTERROR)
	      pdtsize += pdtstrlen (pdttname[pdtx]) + 15, pdtcount++;
	  pdtsize += pdtstrlen ("syntax error, unexpected ") + 1;
	  pdtsize += pdtstrlen (pdttname[pdttype]);
	  pdtmsg = (char *) YYSTACK_ALLOC (pdtsize);
	  if (pdtmsg != 0)
	    {
	      char *pdtp = pdtstpcpy (pdtmsg, "syntax error, unexpected ");
	      pdtp = pdtstpcpy (pdtp, pdttname[pdttype]);

	      if (pdtcount < 5)
		{
		  pdtcount = 0;
		  for (pdtx = pdtn < 0 ? -pdtn : 0;
		       pdtx < (int) (sizeof (pdttname) / sizeof (char *));
		       pdtx++)
		    if (pdtcheck[pdtx + pdtn] == pdtx && pdtx != YYTERROR)
		      {
			const char *pdtq = ! pdtcount ? ", expecting " : " or ";
			pdtp = pdtstpcpy (pdtp, pdtq);
			pdtp = pdtstpcpy (pdtp, pdttname[pdtx]);
			pdtcount++;
		      }
		}
	      pdterror (pdtmsg);
	      YYSTACK_FREE (pdtmsg);
	    }
	  else
	    pdterror ("syntax error; also virtual memory exhausted");
	}
      else
#endif /* YYERROR_VERBOSE */
	pdterror ("syntax error");
    }



  if (pdterrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      /* Return failure if at end of input.  */
      if (pdtchar == YYEOF)
        {
	  /* Pop the error token.  */
          YYPOPSTACK;
	  /* Pop the rest of the stack.  */
	  while (pdtss < pdtssp)
	    {
	      YYDSYMPRINTF ("Error: popping", pdtstos[*pdtssp], pdtvsp, pdtlsp);
	      pdtdestruct (pdtstos[*pdtssp], pdtvsp);
	      YYPOPSTACK;
	    }
	  YYABORT;
        }

      YYDSYMPRINTF ("Error: discarding", pdttoken, &pdtlval, &pdtlloc);
      pdtdestruct (pdttoken, &pdtlval);
      pdtchar = YYEMPTY;

    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto pdterrlab1;


/*----------------------------------------------------.
| pdterrlab1 -- error raised explicitly by an action.  |
`----------------------------------------------------*/
pdterrlab1:
  pdterrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      pdtn = pdtpact[pdtstate];
      if (pdtn != YYPACT_NINF)
	{
	  pdtn += YYTERROR;
	  if (0 <= pdtn && pdtn <= YYLAST && pdtcheck[pdtn] == YYTERROR)
	    {
	      pdtn = pdttable[pdtn];
	      if (0 < pdtn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (pdtssp == pdtss)
	YYABORT;

      YYDSYMPRINTF ("Error: popping", pdtstos[*pdtssp], pdtvsp, pdtlsp);
      pdtdestruct (pdtstos[pdtstate], pdtvsp);
      pdtvsp--;
      pdtstate = *--pdtssp;

      YY_STACK_PRINT (pdtss, pdtssp);
    }

  if (pdtn == YYFINAL)
    YYACCEPT;

  YYDPRINTF ((stderr, "Shifting error token, "));

  *++pdtvsp = pdtlval;


  pdtstate = pdtn;
  goto pdtnewstate;


/*-------------------------------------.
| pdtacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
pdtacceptlab:
  pdtresult = 0;
  goto pdtreturn;

/*-----------------------------------.
| pdtabortlab -- YYABORT comes here.  |
`-----------------------------------*/
pdtabortlab:
  pdtresult = 1;
  goto pdtreturn;

#ifndef pdtoverflow
/*----------------------------------------------.
| pdtoverflowlab -- parser overflow comes here.  |
`----------------------------------------------*/
pdtoverflowlab:
  pdterror ("parser stack overflow");
  pdtresult = 2;
  /* Fall through.  */
#endif

pdtreturn:
#ifndef pdtoverflow
  if (pdtss != pdtssa)
    YYSTACK_FREE (pdtss);
#endif
  return pdtresult;
}


#line 289 "pdt/parser.y"


int pdterror(char* s)
{
  extern int pdt_lineno;
  extern char* pdt_curfile;
  fprintf(stderr, "%s: %s at line %d\n" , pdt_curfile, s, pdt_lineno);
  return -1;
}


