#ifndef ExprLib_h
#define ExprLib_h

#include <expression/FieldManager.h>       // field manager definition 

#define ENABLE_UINTAH

#include <expression/Expression.h>      // basic expression support 
#include <expression/ExpressionTree.h>     // support for graphs 
#include <expression/ExpressionFactory.h>  // expression creation help 
#include <expression/TransportEquation.h>  // support for basic transport equations
#include <expression/Functions.h>          // some basic functions wrapped as expressions

#define EXPR_REPO_DATE "Thu Aug 28 15:24:33 2014 -0600"  // date of last commit for ExprLib
#define EXPR_REPO_HASH "139a0fc9fe71f5306355c4d1c1446a6eb9e31870"  // hash for ExprLib version
#endif // ExprLib_h
