/*
 * Copyright (c) 1994 Silicon Graphics, Inc.
 * 
 * Permission to use, copy, modify, distribute, and sell this software and
 * its documentation for any purpose is hereby granted without fee,
 * provided that (i) the above copyright notices and this permission
 * notice appear in all copies of the software and related documentation,
 * and (ii) the name of Silicon Graphics may not be used in any
 * advertising or publicity relating to the software without the specific,
 * prior written permission of Silicon Graphics.
 * 
 * THE SOFTWARE IS PROVIDED "AS-IS" AND WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS, IMPLIED OR OTHERWISE, INCLUDING WITHOUT LIMITATION, ANY
 * WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * 
 * IN NO EVENT SHALL SILICON GRAPHICS BE LIABLE FOR ANY SPECIAL,
 * INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY KIND, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
 * OR NOT ADVISED OF THE POSSIBILITY OF DAMAGE, AND ON ANY THEORY OF
 * LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THIS SOFTWARE.
 */


/*****************************************************************************
 * visPixelFormat, visGetGLXVisualInfo - tools for choosing OpenGL pixel
 *	formats
 *****************************************************************************/


#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "visinfo.h"

#define DEBUG 0

#define Emit(x)		*ConditionP++ = (x)
#define EmitKey(x)	*SortKeyP++ = (x)



/*
 * Tokens that are used by the parser and expression evaluator:
 */

#define VIS_OP_END		0	/* end of expression		*/
#define VIS_OP_ADD		1	/* C integer +			*/
#define VIS_OP_AND		2	/* C integer &&			*/
#define VIS_OP_DIV		3	/* C integer /			*/
#define VIS_OP_EQ		4	/* C integer ==			*/
#define VIS_OP_GE		5	/* C integer >=			*/
#define VIS_OP_GT		6	/* C integer >			*/
#define VIS_OP_LE		7	/* C integer <=			*/
#define VIS_OP_LT		8	/* C integer <			*/
#define VIS_OP_MOD		9	/* C integer %			*/
#define VIS_OP_MUL		10	/* C integer *			*/
#define VIS_OP_NE		11	/* C integer !=			*/
#define VIS_OP_NEGATE		12	/* C integer unary -		*/
#define VIS_OP_NOT		13	/* C integer unary !		*/
#define VIS_OP_OR		14	/* C integer ||			*/
#define VIS_OP_SUB		15	/* C integer -			*/

#define VIS_TOK_CONSTANT	16	/* integer constants		*/
#define VIS_TOK_ERROR		17	/* erroneous token		*/
#define VIS_TOK_LPAREN		18	/* (				*/
#define VIS_TOK_RPAREN		19	/* )				*/
#define VIS_TOK_SEPARATOR	20	/* ,				*/

#define VIS_SORT_MAX		21	/* sort largest value first	*/
#define VIS_SORT_MIN		22	/* sort smallest value first	*/

#define VIS_FIRST_VAR		23
#define VIS_VAR_A		23	/* alpha channel size		*/
#define VIS_VAR_ACCUM_A		24	/* accum alpha channel size	*/
#define VIS_VAR_ACCUM_B		25	/* accum blue channel size	*/
#define VIS_VAR_ACCUM_G		26	/* accum green channel size	*/
#define VIS_VAR_ACCUM_R		27	/* accum red channel size	*/
#define VIS_VAR_ACCUM_RGB	28	/* min(accum r, g, b)		*/
#define VIS_VAR_ACCUM_RGBA	29	/* min(accum r, g, b, a)	*/
#define VIS_VAR_AUX		30	/* number of aux color buffers	*/
#define VIS_VAR_B		31	/* blue channel size		*/
#define VIS_VAR_CI		32	/* color index buffer size	*/
#define VIS_VAR_DB		33	/* nonzero if double-buffered	*/
#define VIS_VAR_G		34	/* green channel size		*/
#define VIS_VAR_GLX		35	/* nonzero if supports OpenGL	*/
#define VIS_VAR_ID		36	/* visual (or pixel format) id	*/
#define VIS_VAR_LEVEL		37	/* buffer level			*/
#define VIS_VAR_MAIN		38	/* nonzero for main buffers	*/
#define VIS_VAR_MONO		39	/* nonzero if monoscopic	*/
#define VIS_VAR_MS		40	/* number of (multi)samples	*/
#define VIS_VAR_OVERLAY		41	/* nonzero for overlay buffers	*/
#define VIS_VAR_R		42	/* red channel size		*/
#define VIS_VAR_RGB		43	/* min(r, g, b)			*/
#define VIS_VAR_RGBA		44	/* min(r, g, b, a)		*/
#define VIS_VAR_S		45	/* stencil buffer size		*/
#define VIS_VAR_SB		46	/* nonzero if single-buffered	*/
#define VIS_VAR_STEREO		47	/* nonzero if stereoscopic	*/
#define VIS_VAR_UNDERLAY	48	/* nonzero for underlay buffers	*/
#define VIS_VAR_Z		49	/* depth (z) buffer size	*/
#define VIS_LAST_VAR		49


/*
 * Structures for parsing, sorting, and evaluating conditions:
 */

#define VIS_KEYSIZE		((VIS_LAST_VAR-VIS_FIRST_VAR+1)*2+1)
static int SortKeys[VIS_KEYSIZE];
static int* SortKeyP;

#define VIS_CONDSIZE		100
static int Condition[VIS_CONDSIZE];
static int* ConditionP;

#define VIS_VARSIZE		128
static char* CriteriaP;
static int Symbol;
static int Value;

#define VIS_STACKSIZE		100


/*
 * Parser status values:
 */

#define VIS_PRESENT		0
#define VIS_ABSENT		1
#define VIS_SYNTAX_ERROR	2


/*
 * Name-to-value mapping for converting variables to token values:
 */

typedef struct {
	char* name;
	int value;
	} NameMappingT;


/*
 * Pointer to X11 display (made global for use by qsort auxiliary routines):
 */

static Display* DisplayPointer;



static int AcceptableVisual(XVisualInfo* v);
static int CompareNames(const void* n1, const void* n2);
static int CompareVisuals(const void* left, const void* right);
static int FetchVariable(XVisualInfo* visual, int variable);
static void FilterVisuals(XVisualInfo** vp, int* n);
static void GetSymbol(void);
static int LookupName(char* name);
static int ParseArithExpr(void);
static int ParseArithFactor(void);
static int ParseArithPrimary(void);
static int ParseArithTerm(void);
static int ParseBoolFactor(void);
static int ParseBoolTerm(void);
static int ParseCriteria(void);
static int ParseCriterion(void);
static int ParseExpression(void);
static int ParseSortKey(void);



#if DEBUG
static char* SymbolToString(int symbol);
#endif



/******************************************************************************
 * AcceptableVisual
 *	Evaluate selection expression for a single XVisualInfo structure.
 *	Return nonzero if the visual meets all conditions.
 *****************************************************************************/

static int
AcceptableVisual(XVisualInfo* v) {
	int stack[VIS_STACKSIZE];
	int* sp;
	int* cp;

	/*
	 * Initialize evaluation stack.  Set pointer into the stack,
	 * rather than at the very beginning, to allow fetching operands
	 * without checking stack depth.
	 */
	sp = stack + 1;

	/*
	 * Process the RPN expression string:
	 */
	for (cp = Condition; *cp != VIS_OP_END; ) {
		int right = *--sp;
		int left = sp[-1];
		int result;

		switch (*cp++) {
			case VIS_OP_ADD:
				result = left + right;
				break;
			case VIS_OP_AND:
				result = left && right;
				break;
			case VIS_OP_DIV:
				if (right == 0)
					return 0;
				result = left / right;
				break;
			case VIS_OP_EQ:
				result = left == right;
				break;
			case VIS_OP_GE:
				result = left >= right;
				break;
			case VIS_OP_GT:
				result = left > right;
				break;
			case VIS_OP_LE:
				result = left <= right;
				break;
			case VIS_OP_LT:
				result = left < right;
				break;
			case VIS_OP_MOD:
				if (right == 0)
					return 0;
				result = left % right;
				break;
			case VIS_OP_MUL:
				result = left * right;
				break;
			case VIS_OP_NE:
				result = left != right;
				break;
			case VIS_OP_NEGATE:
				++sp;	/* put back unused operand */
				result = - right;
				break;
			case VIS_OP_NOT:
				++sp;	/* put back unused operand */
				result = ! right;
				break;
			case VIS_OP_OR:
				result = left || right;
				break;
			case VIS_OP_SUB:
				result = left - right;
				break;
			case VIS_TOK_CONSTANT:
				sp += 2;/* put back unused operands */
				result = *cp++;
				break;
			default:	/* must be a variable */
				sp += 2;/* put back unused operands */
				result = FetchVariable(v, cp[-1]);
				break;
			}
		
		sp[-1] = result;
		}

	return stack[1];
	}



/******************************************************************************
 * CompareNames
 *	Name comparison function used by bsearch() in LookupName().
 *****************************************************************************/

static int
CompareNames(const void* n1, const void* n2) {
	return strcmp(((NameMappingT*) n1)->name, ((NameMappingT*) n2)->name);
	}



/******************************************************************************
 * CompareVisuals
 *	Visual comparison function used by qsort() in visGetGLXVisualInfo().
 *****************************************************************************/

static int
CompareVisuals(const void* left, const void* right) {
	int varLeft, varRight;
	int difference;
	int k;

	for (k = 0; SortKeys[k] != VIS_OP_END; k += 2) {
		varLeft = FetchVariable(*(XVisualInfo**) left, SortKeys[k + 1]);
		varRight= FetchVariable(*(XVisualInfo**) right,SortKeys[k + 1]);
		difference = varLeft - varRight;
		if (SortKeys[k] == VIS_SORT_MAX)   /* sort largest first? */
			difference = -difference;
		if (difference)
			return difference;
		}

	return 0;
	}



/******************************************************************************
 * FetchVariable
 *	Fetch the value of a variable from an XVisualInfo structure
 *****************************************************************************/

static int
FetchVariable(XVisualInfo* visual, int variable) {
	int var;
	int i;

	switch (variable) {
		case VIS_VAR_A:
			glXGetConfig(DisplayPointer, visual, GLX_RGBA, &var);
			if (!var)
				break;
			glXGetConfig(DisplayPointer, visual, GLX_ALPHA_SIZE,
				&var);
			break;
		case VIS_VAR_ACCUM_A:
			glXGetConfig(DisplayPointer, visual,
				GLX_ACCUM_ALPHA_SIZE, &var);
			break;
		case VIS_VAR_ACCUM_B:
			glXGetConfig(DisplayPointer, visual,
				GLX_ACCUM_BLUE_SIZE, &var);
			break;
		case VIS_VAR_ACCUM_G:
			glXGetConfig(DisplayPointer, visual,
				GLX_ACCUM_GREEN_SIZE, &var);
			break;
		case VIS_VAR_ACCUM_R:
			glXGetConfig(DisplayPointer, visual, GLX_ACCUM_RED_SIZE,
				&var);
			break;
		case VIS_VAR_ACCUM_RGB:
			glXGetConfig(DisplayPointer, visual, GLX_ACCUM_RED_SIZE,
				&var);
			glXGetConfig(DisplayPointer, visual,
				GLX_ACCUM_GREEN_SIZE, &i);
			if (i < var)
				var = i;
			glXGetConfig(DisplayPointer, visual,
				GLX_ACCUM_BLUE_SIZE, &i);
			if (i < var)
				var = i;
			break;
		case VIS_VAR_ACCUM_RGBA:
			glXGetConfig(DisplayPointer, visual, GLX_ACCUM_RED_SIZE,
				&var);
			glXGetConfig(DisplayPointer, visual,
				GLX_ACCUM_GREEN_SIZE, &i);
			if (i < var)
				var = i;
			glXGetConfig(DisplayPointer, visual,
				GLX_ACCUM_BLUE_SIZE, &i);
			if (i < var)
				var = i;
			glXGetConfig(DisplayPointer, visual,
				GLX_ACCUM_ALPHA_SIZE, &i);
			if (i < var)
				var = i;
			break;
		case VIS_VAR_AUX:
			glXGetConfig(DisplayPointer, visual, GLX_AUX_BUFFERS,
				&var);
			break;
		case VIS_VAR_B:
			glXGetConfig(DisplayPointer, visual, GLX_RGBA, &var);
			if (!var)
				break;
			glXGetConfig(DisplayPointer, visual, GLX_BLUE_SIZE,
				&var);
			break;
		case VIS_VAR_CI:
			glXGetConfig(DisplayPointer, visual, GLX_RGBA, &var);
			if (var) {
				var = 0;
				break;
				}
			glXGetConfig(DisplayPointer, visual, GLX_BUFFER_SIZE,
				&var);
			break;
		case VIS_VAR_DB:
			glXGetConfig(DisplayPointer, visual, GLX_DOUBLEBUFFER,
				&var);
			break;
		case VIS_VAR_G:
			glXGetConfig(DisplayPointer, visual, GLX_RGBA, &var);
			if (!var)
				break;
			glXGetConfig(DisplayPointer, visual, GLX_GREEN_SIZE,
				&var);
			break;
		case VIS_VAR_GLX:
			glXGetConfig(DisplayPointer, visual, GLX_USE_GL,
				&var);
			break;
		case VIS_VAR_ID:
			var = (int)visual->visualid;
			break;
		case VIS_VAR_LEVEL:
			glXGetConfig(DisplayPointer, visual, GLX_LEVEL,
				&var);
			break;
		case VIS_VAR_MAIN:
			glXGetConfig(DisplayPointer, visual, GLX_LEVEL,
				&var);
			var = var == 0;
			break;
		case VIS_VAR_MONO:
			glXGetConfig(DisplayPointer, visual, GLX_STEREO,
				&var);
			var = !var;
			break;
		case VIS_VAR_MS:
#if defined(GL_SGIS_multisample) && defined(__sgi)
			glXGetConfig(DisplayPointer, visual, GLX_SAMPLES_SGIS,
				&var);
#else
			var = 0;
#endif
			break;
		case VIS_VAR_OVERLAY:
			glXGetConfig(DisplayPointer, visual, GLX_LEVEL,
				&var);
			var = var > 0;
			break;
		case VIS_VAR_R:
			glXGetConfig(DisplayPointer, visual, GLX_RGBA, &var);
			if (!var)
				break;
			glXGetConfig(DisplayPointer, visual, GLX_RED_SIZE,
				&var);
			break;
		case VIS_VAR_RGB:
			glXGetConfig(DisplayPointer, visual, GLX_RGBA, &var);
			if (!var)
				break;
			glXGetConfig(DisplayPointer, visual, GLX_RED_SIZE,
				&var);
			glXGetConfig(DisplayPointer, visual, GLX_GREEN_SIZE,
				&i);
			if (i < var)
				var = i;
			glXGetConfig(DisplayPointer, visual, GLX_BLUE_SIZE,
				&i);
			if (i < var)
				var = i;
			break;
		case VIS_VAR_RGBA:
			glXGetConfig(DisplayPointer, visual, GLX_RGBA, &var);
			if (!var)
				break;
			glXGetConfig(DisplayPointer, visual, GLX_RED_SIZE,
				&var);
			glXGetConfig(DisplayPointer, visual, GLX_GREEN_SIZE,
				&i);
			if (i < var)
				var = i;
			glXGetConfig(DisplayPointer, visual, GLX_BLUE_SIZE,
				&i);
			if (i < var)
				var = i;
			glXGetConfig(DisplayPointer, visual, GLX_ALPHA_SIZE,
				&i);
			if (i < var)
				var = i;
			break;
		case VIS_VAR_S:
			glXGetConfig(DisplayPointer, visual, GLX_STENCIL_SIZE,
				&var);
			break;
		case VIS_VAR_SB:
			glXGetConfig(DisplayPointer, visual, GLX_DOUBLEBUFFER,
				&var);
			var = !var;
			break;
		case VIS_VAR_STEREO:
			glXGetConfig(DisplayPointer, visual, GLX_STEREO,
				&var);
			break;
		case VIS_VAR_UNDERLAY:
			glXGetConfig(DisplayPointer, visual, GLX_LEVEL,
				&var);
			var = var < 0;
			break;
		case VIS_VAR_Z:
			glXGetConfig(DisplayPointer, visual, GLX_DEPTH_SIZE,
				&var);
			break;
		}

	return var;
	}



/******************************************************************************
 * FilterVisuals
 *	Remove unwanted visuals from consideration
 *****************************************************************************/

static void
FilterVisuals(XVisualInfo** vp, int* n) {
	int i;
	int j;

	/*
	 * Evaluate the conditional expression for each visual and
	 * eliminate pointers to visuals for which the expression is zero:
	 */
	j = 0;
	for (i = 0; i < *n; ++i) {
		if (AcceptableVisual(vp[i]))
			vp[j++] = vp[i];
		}
	*n = j;
	}



/******************************************************************************
 * GetSymbol
 *	Fetch next symbol from the input string
 *****************************************************************************/

static void
GetSymbol(void) {
	char var[VIS_VARSIZE];
	char* varP;
	char* p;
	char c;
	char nextC;

	p = CriteriaP;
	while (isspace(*p))
		++p;

	if (isalpha(*p)) {
		varP = var;
		while (isalnum(*p))
			*varP++ = tolower(*p++);
		*varP = '\0';
		CriteriaP = p;
		Symbol = LookupName(var);
		return;
		}

	if (isdigit(*p)) {
		Value = strtol(p, &CriteriaP, 0);
		Symbol = VIS_TOK_CONSTANT;
		return;
		}

	Symbol = VIS_TOK_ERROR;
	c = *p++;
	if (c)
		nextC = *p;
	switch (c) {
		case '|':
			if (nextC == '|') {
				++p;
				Symbol = VIS_OP_OR;
				}
			break;
		case '&':
			if (nextC == '&') {
				++p;
				Symbol = VIS_OP_AND;
				}
			break;
		case '<':
			if (nextC == '=') {
				++p;
				Symbol = VIS_OP_LE;
				}
			else
				Symbol = VIS_OP_LT;
			break;
		case '>':
			if (nextC == '=') {
				++p;
				Symbol = VIS_OP_GE;
				}
			else
				Symbol = VIS_OP_GT;
			break;
		case '=':
			if (nextC == '=') {
				++p;
				Symbol = VIS_OP_EQ;
				}
			break;
		case '!':
			if (nextC == '=') {
				++p;
				Symbol = VIS_OP_NE;
				}
			else
				Symbol = VIS_OP_NOT;
			break;
		case '+':
			Symbol = VIS_OP_ADD;
			break;
		case '-':
			Symbol = VIS_OP_SUB;
			break;
		case '*':
			Symbol = VIS_OP_MUL;
			break;
		case '/':
			Symbol = VIS_OP_DIV;
			break;
		case '%':
			Symbol = VIS_OP_MOD;
			break;
		case ',':
			Symbol = VIS_TOK_SEPARATOR;
			break;
		case '(':
			Symbol = VIS_TOK_LPAREN;
			break;
		case ')':
			Symbol = VIS_TOK_RPAREN;
			break;
		case '\0':
			Symbol = VIS_OP_END;
			--p;	/* push back '\0' */
			break;
		default:
			/* do nothing; Symbol already equals VIS_TOK_ERROR */
			;
		}
	CriteriaP = p;
	return;
	}



/******************************************************************************
 * LookupName
 *	finds the token corresponding to a character string, if any exists
 *****************************************************************************/

static int
LookupName(char* name) {
	static NameMappingT map[] = {	/* NOTE:  Must be sorted! */
		{"a", 		VIS_VAR_A},
		{"accuma", 	VIS_VAR_ACCUM_A},
		{"accumb", 	VIS_VAR_ACCUM_B},
		{"accumg", 	VIS_VAR_ACCUM_G},
		{"accumr", 	VIS_VAR_ACCUM_R},
		{"accumrgb", 	VIS_VAR_ACCUM_RGB},
		{"accumrgba", 	VIS_VAR_ACCUM_RGBA},
		{"aux", 	VIS_VAR_AUX},
		{"b", 		VIS_VAR_B},
		{"ci", 		VIS_VAR_CI},
		{"db", 		VIS_VAR_DB},
		{"g", 		VIS_VAR_G},
		{"id", 		VIS_VAR_ID},
		{"level", 	VIS_VAR_LEVEL},
		{"main",	VIS_VAR_MAIN},
		{"max", 	VIS_SORT_MAX},
		{"min", 	VIS_SORT_MIN},
		{"mono", 	VIS_VAR_MONO},
		{"ms", 		VIS_VAR_MS},
		{"overlay",	VIS_VAR_OVERLAY},
		{"r", 		VIS_VAR_R},
		{"rgb", 	VIS_VAR_RGB},
		{"rgba", 	VIS_VAR_RGBA},
		{"s", 		VIS_VAR_S},
		{"sb", 		VIS_VAR_SB},
		{"stereo", 	VIS_VAR_STEREO},
		{"underlay",	VIS_VAR_UNDERLAY},
		{"z", 		VIS_VAR_Z}
		};
	NameMappingT n;
	NameMappingT* m;

	n.name = name;
	m = bsearch(&n, map, sizeof(map)/sizeof(map[0]), sizeof(map[0]),
		CompareNames);
	return (m == NULL)? VIS_TOK_ERROR: m->value;
	}



/******************************************************************************
 * ParseArithExpr
 *	Syntax:	arithExpr -> arithTerm {('+'|'-') arithTerm}
 *****************************************************************************/

static int
ParseArithExpr(void) {
	int status;
	int op;

	status = ParseArithTerm();
	if (status != VIS_PRESENT)
		return status;

	for (;;) {
		if (Symbol == VIS_OP_ADD || Symbol == VIS_OP_SUB) {
			op = Symbol;
			GetSymbol();
			if (ParseArithTerm() != VIS_PRESENT)
				return VIS_SYNTAX_ERROR;
			Emit(op);
			}
		else
			return VIS_PRESENT;
		}
	}



/******************************************************************************
 * ParseArithFactor
 *	Syntax:	arithFactor -> ['+'|'-'|'!'] arithPrimary
 *****************************************************************************/

static int
ParseArithFactor(void) {
	int op;

	if (Symbol == VIS_OP_ADD || Symbol == VIS_OP_SUB
	  || Symbol == VIS_OP_NOT) {
		op = Symbol;
		GetSymbol();
		if (ParseArithPrimary() != VIS_PRESENT)
			return VIS_SYNTAX_ERROR;
		if (op == VIS_OP_SUB)
			Emit(VIS_OP_NEGATE);
		else if (op == VIS_OP_NOT)
			Emit(VIS_OP_NOT);
		return VIS_PRESENT;
		}

	return ParseArithPrimary();
	}



/******************************************************************************
 * ParseArithPrimary
 *	Syntax:	arithPrimary -> variable | constant | '(' expression ')'
 *****************************************************************************/

static int
ParseArithPrimary(void) {
	if (VIS_FIRST_VAR <= Symbol && Symbol <= VIS_LAST_VAR) {
		Emit(Symbol);
		GetSymbol();
		return VIS_PRESENT;
		}

	if (Symbol == VIS_TOK_CONSTANT) {
		Emit(VIS_TOK_CONSTANT);
		Emit(Value);
		GetSymbol();
		return VIS_PRESENT;
		}

	if (Symbol == VIS_TOK_LPAREN) {
		GetSymbol();
		if (ParseExpression() != VIS_PRESENT)
			return VIS_SYNTAX_ERROR;
		if (Symbol == VIS_TOK_RPAREN) {
			GetSymbol();
			return VIS_PRESENT;
			}
		else
			return VIS_SYNTAX_ERROR;
		}

	return VIS_ABSENT;
	}



/******************************************************************************
 * ParseArithTerm
 *	Syntax:	arithTerm -> arithFactor {('*'|'/'|'%') arithFactor}
 *****************************************************************************/

static int
ParseArithTerm(void) {
	int status;
	int op;

	status = ParseArithFactor();
	if (status != VIS_PRESENT)
		return status;

	for (;;) {
		if (Symbol == VIS_OP_MUL
		 || Symbol == VIS_OP_DIV
		 || Symbol == VIS_OP_MOD) {
			op = Symbol;
			GetSymbol();
			if (ParseArithFactor() != VIS_PRESENT)
				return VIS_SYNTAX_ERROR;
			Emit(op);
			}
		else
			return VIS_PRESENT;
		}
	}



/******************************************************************************
 * ParseBoolFactor
 *   Syntax:  boolFactor -> arithExpr [('<'|'>'|'<='|'>='|'=='|'!=') arithExpr]
 *****************************************************************************/

static int
ParseBoolFactor(void) {
	int status;
	int op;

	status = ParseArithExpr();
	if (status != VIS_PRESENT)
		return status;

	if (Symbol == VIS_OP_LT
	 || Symbol == VIS_OP_GT
	 || Symbol == VIS_OP_LE
	 || Symbol == VIS_OP_GE
	 || Symbol == VIS_OP_EQ
	 || Symbol == VIS_OP_NE) {
		op = Symbol;
		GetSymbol();
		if (ParseArithExpr() != VIS_PRESENT)
			return VIS_SYNTAX_ERROR;
		Emit(op);
		}

	return VIS_PRESENT;
	}



/******************************************************************************
 * ParseBoolTerm
 *	Syntax:	boolTerm -> boolFactor {'&&' boolFactor}
 *****************************************************************************/

static int
ParseBoolTerm(void) {
	int status;

	status = ParseBoolFactor();
	if (status != VIS_PRESENT)
		return status;

	for (;;) {
		if (Symbol == VIS_OP_AND) {
			GetSymbol();
			if (ParseBoolFactor() != VIS_PRESENT)
				return VIS_SYNTAX_ERROR;
			Emit(VIS_OP_AND);
			}
		else
			return VIS_PRESENT;
		}
	}



/******************************************************************************
 * ParseCriteria
 *	Syntax:  criteria -> criterion {',' criterion}
 *****************************************************************************/

static int
ParseCriteria(void) {
	int status;

	/* Consider only GLX-capable Visuals: */
	Emit(VIS_VAR_GLX);

	/* Process all the user-specified conditions and sort keys: */
	status = ParseCriterion();
	if (status != VIS_PRESENT)
		return status;

	for (;;) {
		if (Symbol == VIS_TOK_SEPARATOR) {
			GetSymbol();
			if (ParseCriterion() != VIS_PRESENT)
				return VIS_SYNTAX_ERROR;
			}
		else if (Symbol == VIS_OP_END)
			return VIS_PRESENT;
		else
			return VIS_SYNTAX_ERROR;
		}
	}



/******************************************************************************
 * ParseCriterion
 *	Syntax:  criterion -> sortKey | expression
 *****************************************************************************/

static int
ParseCriterion(void) {
	int status;

	status = ParseSortKey();
	if (status == VIS_ABSENT)
		status = ParseExpression();

	if (status == VIS_PRESENT)
		Emit(VIS_OP_AND);
	return status;
	}



/******************************************************************************
 * ParseExpression
 *	Syntax:  expression -> boolTerm {'||' boolTerm}
 *****************************************************************************/

static int
ParseExpression(void) {
	int status;

	status = ParseBoolTerm();
	if (status != VIS_PRESENT)
		return status;

	for (;;) {
		if (Symbol == VIS_OP_OR) {
			GetSymbol();
			if (ParseBoolTerm() != VIS_PRESENT)
				return VIS_SYNTAX_ERROR;
			Emit(VIS_OP_OR);
			}
		else
			return VIS_PRESENT;
		}
	}



/******************************************************************************
 * ParseSortKey
 *	Syntax:  sortKey -> ('max'|'min') variable
 *****************************************************************************/

static int
ParseSortKey(void) {
	if (Symbol == VIS_SORT_MAX || Symbol == VIS_SORT_MIN) {
		EmitKey(Symbol);
		GetSymbol();
		if (VIS_FIRST_VAR <= Symbol && Symbol <= VIS_LAST_VAR) {
			EmitKey(Symbol);
			/*
			 * When sorting, eliminate visuals with a zero value
			 * for the key.  This is hard to justify on grounds
			 * of orthogonality, but it seems to yield the right
			 * behavior.
			 */
			Emit(Symbol);
			GetSymbol();
			return VIS_PRESENT;
			}
		else
			return VIS_SYNTAX_ERROR;
		}

	return VIS_ABSENT;
	}



/******************************************************************************
 * visGetGLXVisualInfo
 *	Return an array of XVisualInfo structures satisfying the current
 *	visual selection expression, sorted according to the current sort
 *	keys, for the specified X11 display and screen.
 *
 * Returns NULL if malloc() fails, otherwise a list of XVisualInfo structures.
 * The memory referenced by the returned pointer should be freed with free()
 *	when it's no longer needed.
 *****************************************************************************/

XVisualInfo*
visGetGLXVisualInfo(Display* dpy, int screen, int* nVInfo) {
	XVisualInfo templat;
	XVisualInfo* v;
	XVisualInfo* newV;
	XVisualInfo** vp;
	int i;
	int n;

	*nVInfo = 0;		/* insurance against careless apps */
	DisplayPointer = dpy;	/* needed by FetchVariable routine */

	/* Get the list of raw XVisualInfo structures: */
	templat.screen = screen;
	if (!(v = XGetVisualInfo(dpy,VisualScreenMask,&templat,&n)))
		return NULL;

	/* Construct a list of pointers to the XVisualInfo structures: */
	vp = (XVisualInfo**) malloc(n * sizeof(XVisualInfo*));
	if (!vp) {
		XFree(v);
		return NULL;
		}
	for (i = 0; i < n; ++i)
		vp[i] = v + i;

	/* Delete pointers to visuals that don't meet the criteria: */
	FilterVisuals(vp, &n);

	/* Sort the visuals according to the sort keys: */
	qsort(vp, n, sizeof(XVisualInfo*), CompareVisuals);

	/* Pack the XVisualInfo structures into the output array: */
	newV = (XVisualInfo*) malloc(n * sizeof(XVisualInfo));
	if (!newV) {
		XFree(v);
		free(vp);
		return NULL;
		}
	for (i = 0; i < n; ++i)
		newV[i] = *vp[i];

	/* Clean up and return the new array of XVisualInfo structures: */
	XFree(v);
	free(vp);
	*nVInfo = n;
	return newV;
	}



/******************************************************************************
 * visPixelFormat
 *	Specifies sort keys and visual selection expression for use by a
 *	subsequent call to visGetXVisualInfo.
 *
 * Returns nonzero for success, zero for failure (syntax error in expression).
 *****************************************************************************/

int
visPixelFormat(char* criteria) {
	int status;

	CriteriaP = criteria;
	ConditionP = Condition;
	SortKeyP = SortKeys;
	GetSymbol();
	status = ParseCriteria();
	if (status != VIS_PRESENT) {
		/*
		 * An error occurred. Make things sane, in case
		 * visGetXVisualInfo is called without checking.
		 */
		ConditionP = Condition;
		Emit(VIS_TOK_CONSTANT);
		Emit(0);
		SortKeyP = SortKeys;
		}
	Emit(VIS_OP_END);

	/* Make the final sort in order of increasing visual ID: */
	EmitKey(VIS_SORT_MIN);
	EmitKey(VIS_VAR_ID);
	EmitKey(VIS_OP_END);

	return status == VIS_PRESENT;
	}



#if DEBUG
/******************************************************************************
 * SymbolToString
 *	Debugging routine to convert a symbol to a printable representation
 *****************************************************************************/

static char*
SymbolToString(int symbol) {
	static NameMappingT t[] = {
		{"VIS_ABSENT",		VIS_ABSENT},
		{"VIS_OP_ADD",		VIS_OP_ADD},
		{"VIS_OP_AND",		VIS_OP_AND},
		{"VIS_OP_DIV",		VIS_OP_DIV},
		{"VIS_OP_END",		VIS_OP_END},
		{"VIS_OP_EQ",		VIS_OP_EQ},
		{"VIS_OP_GE",		VIS_OP_GE},
		{"VIS_OP_GT",		VIS_OP_GT},
		{"VIS_OP_LE",		VIS_OP_LE},
		{"VIS_OP_LT",		VIS_OP_LT},
		{"VIS_OP_MOD",		VIS_OP_MOD},
		{"VIS_OP_MUL",		VIS_OP_MUL},
		{"VIS_OP_NE",		VIS_OP_NE},
		{"VIS_OP_NEGATE",	VIS_OP_NEGATE},
		{"VIS_OP_NOT",		VIS_OP_NOT},
		{"VIS_OP_OR",		VIS_OP_OR},
		{"VIS_OP_SUB",		VIS_OP_SUB},
		{"VIS_PRESENT",		VIS_PRESENT},
		{"VIS_SORT_MAX",	VIS_SORT_MAX},
		{"VIS_SORT_MIN",	VIS_SORT_MIN},
		{"VIS_SYNTAX_ERROR",	VIS_SYNTAX_ERROR},
		{"VIS_TOK_CONSTANT",	VIS_TOK_CONSTANT},
		{"VIS_TOK_ERROR",	VIS_TOK_ERROR},
		{"VIS_TOK_LPAREN",	VIS_TOK_LPAREN},
		{"VIS_TOK_RPAREN",	VIS_TOK_RPAREN},
		{"VIS_TOK_SEPARATOR",	VIS_TOK_SEPARATOR},
		{"VIS_VAR_A",		VIS_VAR_A},
		{"VIS_VAR_ACCUM_A",	VIS_VAR_ACCUM_A},
		{"VIS_VAR_ACCUM_B",	VIS_VAR_ACCUM_B},
		{"VIS_VAR_ACCUM_G",	VIS_VAR_ACCUM_G},
		{"VIS_VAR_ACCUM_R",	VIS_VAR_ACCUM_R},
		{"VIS_VAR_ACCUM_RGB",	VIS_VAR_ACCUM_RGB},
		{"VIS_VAR_ACCUM_RGBA",	VIS_VAR_ACCUM_RGBA},
		{"VIS_VAR_AUX",		VIS_VAR_AUX},
		{"VIS_VAR_B",		VIS_VAR_B},
		{"VIS_VAR_CI",		VIS_VAR_CI},
		{"VIS_VAR_DB",		VIS_VAR_DB},
		{"VIS_VAR_G",		VIS_VAR_G},
		{"VIS_VAR_GLX",		VIS_VAR_GLX},
		{"VIS_VAR_ID",		VIS_VAR_ID},
		{"VIS_VAR_LEVEL",	VIS_VAR_LEVEL},
		{"VIS_VAR_MAIN",	VIS_VAR_MAIN},
		{"VIS_VAR_MONO",	VIS_VAR_MONO},
		{"VIS_VAR_MS",		VIS_VAR_MS},
		{"VIS_VAR_OVERLAY",	VIS_VAR_OVERLAY},
		{"VIS_VAR_R",		VIS_VAR_R},
		{"VIS_VAR_RGB",		VIS_VAR_RGB},
		{"VIS_VAR_RGBA",	VIS_VAR_RGBA},
		{"VIS_VAR_S",		VIS_VAR_S},
		{"VIS_VAR_SB",		VIS_VAR_SB},
		{"VIS_VAR_STEREO",	VIS_VAR_STEREO},
		{"VIS_VAR_UNDERLAY",	VIS_VAR_UNDERLAY},
		{"VIS_VAR_Z",		VIS_VAR_Z},
		};
	int i;

	for (i = 0; i < sizeof(t) / sizeof(t[0]); ++i)
		if (t[i].value == symbol)
			return t[i].name;

	return "UNKNOWN";
	}
#endif
