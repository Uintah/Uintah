BEGIN {quote=34}
{
    printf("XQColor_named_colors.insert(%c%s",quote,$4);
    if($5 != "")printf(" %s", $5);
    printf("%c, RGBColor(%g,%g,%g));\n",quote, $1/255.0, $2/255.0, $3/255.0);
}
