/*                                                                             */
/*   (c) Copyright 1997,1998,1999,2000 Magic Earth Inc. - All Rights Reserved. */
/*          Proprietary Information of Magic Earth Inc.                        */
/*                                                                             */
 
#ifndef __GP_SHM__
#define __GP_SHM__


#ifdef __cplusplus
extern "C" {
#endif

/* point to the first memory location after a data structure.                  */
#define gp_shared_data(ptr,type)   ((unsigned char *)(ptr) + sizeof(struct type))
#define gp_shared_vol(ptr,type)    ((unsigned char *)(ptr) - sizeof(struct type))

/* shared memory constant definitions                                          */

#define SEG_NAME_MAX 50
#define SHMSEG_MAX  100

/* GP_volume.load values                                           */
#define GP_LOADING         2     /* still loading                  */
#define GP_LOADED          1     /* volume loaded                  */
#define GP_EXIT_MAIN_PROG  0     /* loader exited while loading    */
#define GP_LOAD_CANCEL     3     /* loader canceled while loading  */
#define GP_NOT_LOADED     -1     /* did not load                   */
#define GP_ATTACH_FAILED  -100   /* attach to shared memory failed */

enum GP_Units {
    GP_METERS                = 0,
    GP_DECIMETERS            = 1,
    GP_CENTIMETERS           = 2,
    GP_KILOMETERS            = 3,
    GP_FEET                  = 10,
    GP_DECIFEET              = 11,
    GP_MILES                 = 13,
    GP_SECONDS               = 30,
    GP_MILLISECONDS          = 35,
    GP_MICROSECONDS          = 37,
    GP_FEET_PER_SECOND       = 40,
    GP_METERS_PER_SECOND     = 50,
    GP_KILOMETERS_PER_SECOND = 53,
    GP_NONE                  = 254,
    GP_UNKNOWN               = 255
};

/* GP_volume.type values                                                     */
#define GP_NO_VOL               0    /* volume in shared memory has no data  */
#define GP_FULL_VOL             1    /* plain data                           */
#define GP_ENC_VOL              2    /* voxel-encoded data                   */
#define GP_ENCGRD_VOL           4    /* voxel-gradient-encoded data          */

#define GP_LEVELS               256  /* 8-bit data                           */

/* GP_volume.magic value                                                     */
#define GP_VOL_VERSION_2_0         0xABC2            /* unique magic number  */
#define GP_VOL_CURRENT_VERSION     GP_VOL_VERSION_2_0

typedef struct GP_volume {
    int          magic;                        /* magic number                   */
    int          volType;                      /* volume type flag               */
    unsigned int size;                         /* not correct for large volumes  */
    int          volLoad;                      /* loading status flag            */
    unsigned     GpSpacer_0;
    char         pathname[300];                /* volume pathname                */
    int          ndata;                        /* number of data bits per voxel  */

    int          GpSpacer_1;
    int          GpSpacer_2;

    int          nbits;                        /* number of bits per voxel     */
    int          xsize, ysize, zsize;          /* volume dimensions (pixels)   */
    int          sliceInterp;                  /* slice interpolation factor   */

    float        voffset, xoffset, yoffset, zoffset;  /* world coord offset    */

    float        vstep, xstep, ystep, zstep;   /* step factors (units/voxel)   */

    char         vunit[16];                    /* attribute physical units     */
    char         xunit[16], yunit[16], zunit[16]; /* coordinate physical units */

    char         vlabel[16];                   /* attribute label              */
    char         xlabel[16], ylabel[16], zlabel[16];   /* coordinate labels    */

    unsigned int histogram[GP_LEVELS];         /* histogram                    */

    unsigned int GpSpacer_3[GP_LEVELS];
    int          GpSpacer_4;
    unsigned     GpSpacer_5;
    char         GpSpacer_6[226-160-SEG_NAME_MAX];

    /* Survey units and World coord units as 0xAAxxyyzz, where:                */
    /*    xx = x-dimension units flag                                          */
    /*    yy = y-dimension units flag                                          */
    /*    zz = z-dimension units flag                                          */
    int         volSurveyUnits;
    int         volWorldUnits;

    /* Three reference points for conversion of spatial coordinates       */
    /* worldXref[0][i] = World X                                          */
    /* worldXref[1][i] = World Y                                          */
    /* worldXref[2][i] = Survey Y                                         */
    /* worldXref[3][i] = Survey X                                         */
    double      worldXref[4][3];

    int         GpSpacer_7;
    int         GpSpacer_8;
    float       GpSpacer_9[12];

    int         orig_x, orig_y, orig_z;        /* original volume dimensions     */

    int         GpSpacer_10;
    short       GpSpacer_11;

    char        segName[SEG_NAME_MAX];         /* Segment name, as xxx/yyy.vol   */
    int         procGrp;                       /* Process group                  */

    char        GpSpacer_12[256];
}   GP_hdr;

#ifdef __cplusplus
}
#endif
#endif /* __GP_SHM__ */
