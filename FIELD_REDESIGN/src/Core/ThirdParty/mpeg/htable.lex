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
/* A short program  to convert bit-values to decimal values.*/
%}

Delim		[ \t\n]
WhiteSpace	{Delim}+
Letter	  	[a-zA-Z]
Digit	 	[0-9]
HexDigit  	({Digit}|[a-fA-F])
OctalDigit	[0-7]
BinaryDigit	[0-1]
Id		{Letter}({Letter}|{Digit})*
DInteger	{Digit}+
HInteger	(0x|0X){HexDigit}+
BInteger	(0b|0B){BinaryDigit}+
OInteger	(0o|0O){OctalDigit}+

%%
{DInteger}	{printf("%d",strtol(yytext,NULL,10));}
{HInteger}	{printf("%d",strtol(yytext+2,NULL,16));}
{OInteger}	{printf("%d",strtol(yytext+2,NULL,8));}
{BInteger}	{printf("%d,%d",strlen(yytext)-2,strtol(yytext+2,NULL,2));}

.|\n		{printf("%s",yytext);} /*Everything's AOK */

%%

main(argc,argv) 
	int argc;
	char **argv;
{
  while (yylex() != 0);   /* Lex* */
}

