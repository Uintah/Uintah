
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char* argv[])
{
    char out[200];
    sprintf(out, "xxmc.4");
    FILE* ofile=fopen(out, "w");
    float data[2][2][2];
    data[0][0][0]=100;
    data[0][0][1]=101;
    data[0][1][0]=101;
    data[0][1][1]=100;
    data[1][0][0]=101;
    data[1][0][1]=101;
    data[1][1][0]=101;
    data[1][1][1]=100;
    fwrite(data, sizeof(float), 8, ofile);
    fclose(ofile);
    char buf[100];
    sprintf(buf, "%s.hdr", out);
    FILE* hdr=fopen(buf, "w");
    fprintf(hdr, "2 2 2\n");
    fprintf(hdr, "-1 -1 -1\n");
    fprintf(hdr, "1 1 1\n");
    fprintf(hdr, "100 101\n");
    fclose(hdr);
}
