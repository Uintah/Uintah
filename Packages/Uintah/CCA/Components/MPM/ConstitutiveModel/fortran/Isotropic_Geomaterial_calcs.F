      SUBROUTINE ISOTROPIC_GEOMATERIAL_CALC( NBLK, NINSV, DT, PROP,
     $                                   SIGARG, D, SVARG, USM)

      CHARACTER*27 IAM
      PARAMETER (IAM='Isotropic_Geomaterial_calcs')

      CALL FATERR(IAM,'This model is only available to licensees')
      CALL FATERR(IAM,'See Jim Guilkey for more information')

      END
