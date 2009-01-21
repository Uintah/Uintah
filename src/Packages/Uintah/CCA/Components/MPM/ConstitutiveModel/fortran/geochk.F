      SUBROUTINE GEOCHK ( UI, GC, DC)

      CHARACTER*6 IAM
      PARAMETER( IAM = 'GEOCHK' )

      CALL FATERR(IAM,'This model is only available to licensees')
      CALL FATERR(IAM,'See Jim Guilkey for more information')

      END
