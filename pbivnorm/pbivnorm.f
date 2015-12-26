* Selected portion of code taken from:
*    http://www.math.wsu.edu/faculty/genz/software/mvtdstpack.f
* to compute bivariate normal and Student's t distribution functions.
*
* Author:
*          Alan Genz
*          Department of Mathematics
*          Washington State University
*          Pullman, WA 99164-3113
*          Email : alangenz@wsu.edu
*
* except for some auxiliary functions whose authors are indicated
* in the respective code below.
*
*     "pbivnorm" subroutine, for calculating vectorized bivariate
*     normals, added by Brenton Kenkel, 2011-02-21.  based on a similar
*     subroutine by Adelchi Azzalini in the 'mnormt' package

      SUBROUTINE PBIVNORM(PROB, LOWER, UPPERA, UPPERB, INFIN, 
     +     CORREL, LENGTH)
      DOUBLE PRECISION PROB(*), LOWER(*), UPPERA(*), UPPERB(*),
     +     CORREL(*)
      DOUBLE PRECISION THIS_UPPER(2), THIS_CORREL, MVBVN
      INTEGER INFIN(*), LENGTH
      DO I = 1, LENGTH, 1
         THIS_UPPER(1) = UPPERA(I)
         THIS_UPPER(2) = UPPERB(I)
         THIS_CORREL = CORREL(I)
         PROB(I) = MVBVN(LOWER, THIS_UPPER, INFIN, THIS_CORREL)
      END DO
      RETURN
      END

************************************************************************      
*
      DOUBLE PRECISION FUNCTION MVBVN( LOWER, UPPER, INFIN, CORREL )
*
*     A function for computing bivariate normal probabilities.
*
*  Parameters
*
*     LOWER  REAL, array of lower integration limits.
*     UPPER  REAL, array of upper integration limits.
*     INFIN  INTEGER, array of integration limits flags:
*            if INFIN(I) = 0, Ith limits are (-infinity, UPPER(I)];
*            if INFIN(I) = 1, Ith limits are [LOWER(I), infinity);
*            if INFIN(I) = 2, Ith limits are [LOWER(I), UPPER(I)].
*     CORREL REAL, correlation coefficient.
*
      DOUBLE PRECISION LOWER(*), UPPER(*), CORREL, MVBVU
      INTEGER INFIN(*)
      IF ( INFIN(1) .EQ. 2  .AND. INFIN(2) .EQ. 2 ) THEN
         MVBVN =  MVBVU ( LOWER(1), LOWER(2), CORREL )
     +           - MVBVU ( UPPER(1), LOWER(2), CORREL )
     +           - MVBVU ( LOWER(1), UPPER(2), CORREL )
     +           + MVBVU ( UPPER(1), UPPER(2), CORREL )
      ELSE IF ( INFIN(1) .EQ. 2  .AND. INFIN(2) .EQ. 1 ) THEN
         MVBVN =  MVBVU ( LOWER(1), LOWER(2), CORREL )
     +           - MVBVU ( UPPER(1), LOWER(2), CORREL )
      ELSE IF ( INFIN(1) .EQ. 1  .AND. INFIN(2) .EQ. 2 ) THEN
         MVBVN =  MVBVU ( LOWER(1), LOWER(2), CORREL )
     +           - MVBVU ( LOWER(1), UPPER(2), CORREL )
      ELSE IF ( INFIN(1) .EQ. 2  .AND. INFIN(2) .EQ. 0 ) THEN
         MVBVN =  MVBVU ( -UPPER(1), -UPPER(2), CORREL )
     +           - MVBVU ( -LOWER(1), -UPPER(2), CORREL )
      ELSE IF ( INFIN(1) .EQ. 0  .AND. INFIN(2) .EQ. 2 ) THEN
         MVBVN =  MVBVU ( -UPPER(1), -UPPER(2), CORREL )
     +           - MVBVU ( -UPPER(1), -LOWER(2), CORREL )
      ELSE IF ( INFIN(1) .EQ. 1  .AND. INFIN(2) .EQ. 0 ) THEN
         MVBVN =  MVBVU ( LOWER(1), -UPPER(2), -CORREL )
      ELSE IF ( INFIN(1) .EQ. 0  .AND. INFIN(2) .EQ. 1 ) THEN
         MVBVN =  MVBVU ( -UPPER(1), LOWER(2), -CORREL )
      ELSE IF ( INFIN(1) .EQ. 1  .AND. INFIN(2) .EQ. 1 ) THEN
         MVBVN =  MVBVU ( LOWER(1), LOWER(2), CORREL )
      ELSE IF ( INFIN(1) .EQ. 0  .AND. INFIN(2) .EQ. 0 ) THEN
         MVBVN =  MVBVU ( -UPPER(1), -UPPER(2), CORREL )
      ELSE
         MVBVN = 1
      END IF
      END 
      DOUBLE PRECISION FUNCTION MVBVU( SH, SK, R )
*
*     A function for computing bivariate normal probabilities;
*       developed using 
*         Drezner, Z. and Wesolowsky, G. O. (1989),
*         On the Computation of the Bivariate Normal Integral,
*         J. Stat. Comput. Simul.. 35 pp. 101-107.
*       with extensive modications for double precisions by    
*         Alan Genz and Yihong Ge
*         Department of Mathematics
*         Washington State University
*         Pullman, WA 99164-3113
*         Email : alangenz@wsu.edu
*
* BVN - calculate the probability that X is larger than SH and Y is
*       larger than SK.
*
* Parameters
*
*   SH  REAL, integration limit
*   SK  REAL, integration limit
*   R   REAL, correlation coefficient
*   LG  INTEGER, number of Gauss Rule Points and Weights
*
      DOUBLE PRECISION BVN, SH, SK, R, ZERO, TWOPI 
      INTEGER I, LG, NG
      PARAMETER ( ZERO = 0, TWOPI = 6.283185307179586D0 ) 
      DOUBLE PRECISION X(10,3), W(10,3), AS, A, B, C, D, RS, XS
      DOUBLE PRECISION MVPHI, SN, ASR, H, K, BS, HS, HK
      SAVE X, W
*     Gauss Legendre Points and Weights, N =  6
      DATA ( W(I,1), X(I,1), I = 1, 3 ) /
     *  0.1713244923791705D+00,-0.9324695142031522D+00,
     *  0.3607615730481384D+00,-0.6612093864662647D+00,
     *  0.4679139345726904D+00,-0.2386191860831970D+00/
*     Gauss Legendre Points and Weights, N = 12
      DATA ( W(I,2), X(I,2), I = 1, 6 ) /
     *  0.4717533638651177D-01,-0.9815606342467191D+00,
     *  0.1069393259953183D+00,-0.9041172563704750D+00,
     *  0.1600783285433464D+00,-0.7699026741943050D+00,
     *  0.2031674267230659D+00,-0.5873179542866171D+00,
     *  0.2334925365383547D+00,-0.3678314989981802D+00,
     *  0.2491470458134029D+00,-0.1252334085114692D+00/
*     Gauss Legendre Points and Weights, N = 20
      DATA ( W(I,3), X(I,3), I = 1, 10 ) /
     *  0.1761400713915212D-01,-0.9931285991850949D+00,
     *  0.4060142980038694D-01,-0.9639719272779138D+00,
     *  0.6267204833410906D-01,-0.9122344282513259D+00,
     *  0.8327674157670475D-01,-0.8391169718222188D+00,
     *  0.1019301198172404D+00,-0.7463319064601508D+00,
     *  0.1181945319615184D+00,-0.6360536807265150D+00,
     *  0.1316886384491766D+00,-0.5108670019508271D+00,
     *  0.1420961093183821D+00,-0.3737060887154196D+00,
     *  0.1491729864726037D+00,-0.2277858511416451D+00,
     *  0.1527533871307259D+00,-0.7652652113349733D-01/
      IF ( ABS(R) .LT. 0.3 ) THEN
         NG = 1
         LG = 3
      ELSE IF ( ABS(R) .LT. 0.75 ) THEN
         NG = 2
         LG = 6
      ELSE 
         NG = 3
         LG = 10
      ENDIF
      H = SH
      K = SK 
      HK = H*K
      BVN = 0
      IF ( ABS(R) .LT. 0.925 ) THEN
         HS = ( H*H + K*K )/2
         ASR = ASIN(R)
         DO I = 1, LG
            SN = SIN(ASR*( X(I,NG)+1 )/2)
            BVN = BVN + W(I,NG)*EXP( ( SN*HK - HS )/( 1 - SN*SN ) )
            SN = SIN(ASR*(-X(I,NG)+1 )/2)
            BVN = BVN + W(I,NG)*EXP( ( SN*HK - HS )/( 1 - SN*SN ) )
         END DO
         BVN = BVN*ASR/(2*TWOPI) + MVPHI(-H)*MVPHI(-K) 
      ELSE
         IF ( R .LT. 0 ) THEN
            K = -K
            HK = -HK
         ENDIF
         IF ( ABS(R) .LT. 1 ) THEN
            AS = ( 1 - R )*( 1 + R )
            A = SQRT(AS)
            BS = ( H - K )**2
            C = ( 4 - HK )/8 
            D = ( 12 - HK )/16
            BVN = A*EXP( -(BS/AS + HK)/2 )
     +             *( 1 - C*(BS - AS)*(1 - D*BS/5)/3 + C*D*AS*AS/5 )
            IF ( HK .GT. -160 ) THEN
               B = SQRT(BS)
               BVN = BVN - EXP(-HK/2)*SQRT(TWOPI)*MVPHI(-B/A)*B
     +                    *( 1 - C*BS*( 1 - D*BS/5 )/3 ) 
            ENDIF
            A = A/2
            DO I = 1, LG
               XS = ( A*(X(I,NG)+1) )**2
               RS = SQRT( 1 - XS )
               BVN = BVN + A*W(I,NG)*
     +              ( EXP( -BS/(2*XS) - HK/(1+RS) )/RS 
     +              - EXP( -(BS/XS+HK)/2 )*( 1 + C*XS*( 1 + D*XS ) ) )
               XS = AS*(-X(I,NG)+1)**2/4
               RS = SQRT( 1 - XS )
               BVN = BVN + A*W(I,NG)*EXP( -(BS/XS + HK)/2 )
     +                    *( EXP( -HK*(1-RS)/(2*(1+RS)) )/RS 
     +                       - ( 1 + C*XS*( 1 + D*XS ) ) )
            END DO
            BVN = -BVN/TWOPI
         ENDIF
         IF ( R .GT. 0 ) BVN =  BVN + MVPHI( -MAX( H, K ) )
         IF ( R .LT. 0 ) BVN = -BVN + MAX( ZERO, MVPHI(-H) - MVPHI(-K) )     
      ENDIF
      MVBVU = BVN
      END
*
      DOUBLE PRECISION FUNCTION MVSTDT( NU, T )
*
*     Student t Distribution Function
*
*                       T
*         TSTDNT = C   I  ( 1 + y*y/NU )**( -(NU+1)/2 ) dy
*                   NU -INF
*
      INTEGER NU, J
      DOUBLE PRECISION MVPHI, T, CSTHE, SNTHE, POLYN, TT, TS, RN, PI
      PARAMETER ( PI = 3.141592653589793D0 )
      IF ( NU .LT. 1 ) THEN
         MVSTDT = MVPHI( T )
      ELSE IF ( NU .EQ. 1 ) THEN
         MVSTDT = ( 1 + 2*ATAN( T )/PI )/2
      ELSE IF ( NU .EQ. 2) THEN
         MVSTDT = ( 1 + T/SQRT( 2 + T*T ))/2
      ELSE 
         TT = T*T
         CSTHE = NU/( NU + TT )
         POLYN = 1
         DO J = NU - 2, 2, -2
            POLYN = 1 + ( J - 1 )*CSTHE*POLYN/J
         END DO
         IF ( MOD( NU, 2 ) .EQ. 1 ) THEN
            RN = NU
            TS = T/SQRT(RN)
            MVSTDT = ( 1 + 2*( ATAN( TS ) + TS*CSTHE*POLYN )/PI )/2
         ELSE
            SNTHE = T/SQRT( NU + TT )
            MVSTDT = ( 1 + SNTHE*POLYN )/2
         END IF
         IF ( MVSTDT .LT. 0 ) MVSTDT = 0
      ENDIF
      END
*
      DOUBLE PRECISION FUNCTION MVBVT( NU, LOWER, UPPER, INFIN, CORREL )      
*
*     A function for computing bivariate normal and t probabilities.
*
*  Parameters
*
*     NU     INTEGER degrees of freedom parameter; NU < 1 gives normal case.
*     LOWER  REAL, array of lower integration limits.
*     UPPER  REAL, array of upper integration limits.
*     INFIN  INTEGER, array of integration limits flags:
*            if INFIN(I) = 0, Ith limits are (-infinity, UPPER(I)];
*            if INFIN(I) = 1, Ith limits are [LOWER(I), infinity);
*            if INFIN(I) = 2, Ith limits are [LOWER(I), UPPER(I)].
*     CORREL REAL, correlation coefficient.
*
      DOUBLE PRECISION LOWER(*), UPPER(*), CORREL, MVBVN, MVBVTL
      INTEGER NU, INFIN(*)
      IF ( NU .LT. 1 ) THEN
            MVBVT =  MVBVN ( LOWER, UPPER, INFIN, CORREL )
      ELSE
         IF ( INFIN(1) .EQ. 2  .AND. INFIN(2) .EQ. 2 ) THEN
            MVBVT =  MVBVTL ( NU, UPPER(1), UPPER(2), CORREL )
     +           - MVBVTL ( NU, UPPER(1), LOWER(2), CORREL )
     +           - MVBVTL ( NU, LOWER(1), UPPER(2), CORREL )
     +           + MVBVTL ( NU, LOWER(1), LOWER(2), CORREL )
         ELSE IF ( INFIN(1) .EQ. 2  .AND. INFIN(2) .EQ. 1 ) THEN
            MVBVT =  MVBVTL ( NU, -LOWER(1), -LOWER(2), CORREL )
     +           - MVBVTL ( NU, -UPPER(1), -LOWER(2), CORREL )
         ELSE IF ( INFIN(1) .EQ. 1  .AND. INFIN(2) .EQ. 2 ) THEN
            MVBVT =  MVBVTL ( NU, -LOWER(1), -LOWER(2), CORREL )
     +           - MVBVTL ( NU, -LOWER(1), -UPPER(2), CORREL )
         ELSE IF ( INFIN(1) .EQ. 2  .AND. INFIN(2) .EQ. 0 ) THEN
            MVBVT =  MVBVTL ( NU, UPPER(1), UPPER(2), CORREL )
     +           - MVBVTL ( NU, LOWER(1), UPPER(2), CORREL )
         ELSE IF ( INFIN(1) .EQ. 0  .AND. INFIN(2) .EQ. 2 ) THEN
            MVBVT =  MVBVTL ( NU, UPPER(1), UPPER(2), CORREL )
     +           - MVBVTL ( NU, UPPER(1), LOWER(2), CORREL )
         ELSE IF ( INFIN(1) .EQ. 1  .AND. INFIN(2) .EQ. 0 ) THEN
            MVBVT =  MVBVTL ( NU, -LOWER(1), UPPER(2), -CORREL )
         ELSE IF ( INFIN(1) .EQ. 0  .AND. INFIN(2) .EQ. 1 ) THEN
            MVBVT =  MVBVTL ( NU, UPPER(1), -LOWER(2), -CORREL )
         ELSE IF ( INFIN(1) .EQ. 1  .AND. INFIN(2) .EQ. 1 ) THEN
            MVBVT =  MVBVTL ( NU, -LOWER(1), -LOWER(2), CORREL )
         ELSE IF ( INFIN(1) .EQ. 0  .AND. INFIN(2) .EQ. 0 ) THEN
            MVBVT =  MVBVTL ( NU, UPPER(1), UPPER(2), CORREL )
         ELSE
            MVBVT = 1
         END IF
      END IF
      END
*
      DOUBLE PRECISION FUNCTION MVBVTC( NU, L, U, INFIN, RHO )      
*
*     A function for computing complementary bivariate normal and t 
*       probabilities.
*
*  Parameters
*
*     NU     INTEGER degrees of freedom parameter.
*     L      REAL, array of lower integration limits.
*     U      REAL, array of upper integration limits.
*     INFIN  INTEGER, array of integration limits flags:
*            if INFIN(1) INFIN(2),        then MVBVTC computes
*                 0         0              P( X>U(1), Y>U(2) )
*                 1         0              P( X<L(1), Y>U(2) )
*                 0         1              P( X>U(1), Y<L(2) )
*                 1         1              P( X<L(1), Y<L(2) )
*                 2         0      P( X>U(1), Y>U(2) ) + P( X<L(1), Y>U(2) )
*                 2         1      P( X>U(1), Y<L(2) ) + P( X<L(1), Y<L(2) )
*                 0         2      P( X>U(1), Y>U(2) ) + P( X>U(1), Y<L(2) )
*                 1         2      P( X<L(1), Y>U(2) ) + P( X<L(1), Y<L(2) )
*                 2         2      P( X>U(1), Y<L(2) ) + P( X<L(1), Y<L(2) )
*                               +  P( X>U(1), Y>U(2) ) + P( X<L(1), Y>U(2) )
*
*     RHO    REAL, correlation coefficient.
*
      DOUBLE PRECISION L(*), U(*), LW(2), UP(2), B, RHO, MVBVT
      INTEGER I, NU, INFIN(*), INF(2)
*
      DO I = 1, 2
         IF ( MOD( INFIN(I), 2 ) .EQ. 0 ) THEN
            INF(I) = 1
            LW(I) = U(I) 
         ELSE
            INF(I) = 0
            UP(I) = L(I) 
         END IF
      END DO
      B = MVBVT( NU, LW, UP, INF, RHO )
      DO I = 1, 2
         IF ( INFIN(I) .EQ. 2 ) THEN
            INF(I) = 0
            UP(I) = L(I) 
            B = B + MVBVT( NU, LW, UP, INF, RHO )
         END IF
      END DO
      IF ( INFIN(1) .EQ. 2 .AND. INFIN(2) .EQ. 2 ) THEN
         INF(1) = 1
         LW(1) = U(1) 
         B = B + MVBVT( NU, LW, UP, INF, RHO )
      END IF
      MVBVTC = B
      END
*
      double precision function mvbvtl( nu, dh, dk, r )
*
*     a function for computing bivariate t probabilities.
*
*       Alan Genz
*       Department of Mathematics
*       Washington State University
*       Pullman, Wa 99164-3113
*       Email : alangenz@wsu.edu
*
*    this function is based on the method described by 
*        Dunnett, C.W. and M. Sobel, (1954),
*        A bivariate generalization of Student's t-distribution
*        with tables for certain special cases,
*        Biometrika 41, pp. 153-169.
*
* mvbvtl - calculate the probability that x < dh and y < dk. 
*
* parameters
*
*   nu number of degrees of freedom
*   dh 1st lower integration limit
*   dk 2nd lower integration limit
*   r   correlation coefficient
*
      integer nu, j, hs, ks
      double precision dh, dk, r
      double precision tpi, pi, ors, hrk, krh, bvt, snu 
      double precision gmph, gmpk, xnkh, xnhk, qhrk, hkn, hpk, hkrn
      double precision btnckh, btnchk, btpdkh, btpdhk, one
      parameter ( pi = 3.14159265358979323844d0, tpi = 2*pi, one = 1 )
      snu = sqrt( dble(nu) )
      ors = 1 - r*r  
      hrk = dh - r*dk  
      krh = dk - r*dh  
      if ( abs(hrk) + ors .gt. 0 ) then
         xnhk = hrk**2/( hrk**2 + ors*( nu + dk**2 ) ) 
         xnkh = krh**2/( krh**2 + ors*( nu + dh**2 ) ) 
      else
         xnhk = 0
         xnkh = 0  
      end if
      hs = sign( one, dh - r*dk )  
      ks = sign( one, dk - r*dh ) 
      if ( mod( nu, 2 ) .eq. 0 ) then
         bvt = atan2( sqrt(ors), -r )/tpi 
         gmph = dh/sqrt( 16*( nu + dh**2 ) )  
         gmpk = dk/sqrt( 16*( nu + dk**2 ) )  
         btnckh = 2*atan2( sqrt( xnkh ), sqrt( 1 - xnkh ) )/pi  
         btpdkh = 2*sqrt( xnkh*( 1 - xnkh ) )/pi 
         btnchk = 2*atan2( sqrt( xnhk ), sqrt( 1 - xnhk ) )/pi  
         btpdhk = 2*sqrt( xnhk*( 1 - xnhk ) )/pi 
         do j = 1, nu/2
            bvt = bvt + gmph*( 1 + ks*btnckh ) 
            bvt = bvt + gmpk*( 1 + hs*btnchk ) 
            btnckh = btnckh + btpdkh  
            btpdkh = 2*j*btpdkh*( 1 - xnkh )/( 2*j + 1 )  
            btnchk = btnchk + btpdhk  
            btpdhk = 2*j*btpdhk*( 1 - xnhk )/( 2*j + 1 )  
            gmph = gmph*( 2*j - 1 )/( 2*j*( 1 + dh**2/nu ) ) 
            gmpk = gmpk*( 2*j - 1 )/( 2*j*( 1 + dk**2/nu ) ) 
         end do
      else
         qhrk = sqrt( dh**2 + dk**2 - 2*r*dh*dk + nu*ors )  
         hkrn = dh*dk + r*nu  
         hkn = dh*dk - nu  
         hpk = dh + dk 
         bvt = atan2(-snu*(hkn*qhrk+hpk*hkrn),hkn*hkrn-nu*hpk*qhrk)/tpi  
         if ( bvt .lt. -1d-15 ) bvt = bvt + 1
         gmph = dh/( tpi*snu*( 1 + dh**2/nu ) )  
         gmpk = dk/( tpi*snu*( 1 + dk**2/nu ) )  
         btnckh = sqrt( xnkh )  
         btpdkh = btnckh 
         btnchk = sqrt( xnhk )  
         btpdhk = btnchk  
         do j = 1, ( nu - 1 )/2
            bvt = bvt + gmph*( 1 + ks*btnckh ) 
            bvt = bvt + gmpk*( 1 + hs*btnchk ) 
            btpdkh = ( 2*j - 1 )*btpdkh*( 1 - xnkh )/( 2*j )  
            btnckh = btnckh + btpdkh  
            btpdhk = ( 2*j - 1 )*btpdhk*( 1 - xnhk )/( 2*j )  
            btnchk = btnchk + btpdhk  
            gmph = 2*j*gmph/( ( 2*j + 1 )*( 1 + dh**2/nu ) ) 
            gmpk = 2*j*gmpk/( ( 2*j + 1 )*( 1 + dk**2/nu ) ) 
         end do
      end if
      mvbvtl = bvt 
*
*     end mvbvtl
*
      end
*
*
      DOUBLE PRECISION FUNCTION MVPHI(Z)
*     
*     Normal distribution probabilities accurate to 1d-15.
*     Reference: J.L. Schonfelder, Math Comp 32(1978), pp 1232-1240. 
*     
      INTEGER I, IM
      DOUBLE PRECISION A(0:43), BM, B, BP, P, RTWO, T, XA, Z
      PARAMETER( RTWO = 1.414213562373095048801688724209D0, IM = 24 )
      SAVE A
      DATA ( A(I), I = 0, 43 )/
     &    6.10143081923200417926465815756D-1,
     &   -4.34841272712577471828182820888D-1,
     &    1.76351193643605501125840298123D-1,
     &   -6.0710795609249414860051215825D-2,
     &    1.7712068995694114486147141191D-2,
     &   -4.321119385567293818599864968D-3, 
     &    8.54216676887098678819832055D-4, 
     &   -1.27155090609162742628893940D-4,
     &    1.1248167243671189468847072D-5, 3.13063885421820972630152D-7,      
     &   -2.70988068537762022009086D-7, 3.0737622701407688440959D-8,
     &    2.515620384817622937314D-9, -1.028929921320319127590D-9,
     &    2.9944052119949939363D-11, 2.6051789687266936290D-11,
     &   -2.634839924171969386D-12, -6.43404509890636443D-13,
     &    1.12457401801663447D-13, 1.7281533389986098D-14, 
     &   -4.264101694942375D-15, -5.45371977880191D-16,
     &    1.58697607761671D-16, 2.0899837844334D-17, 
     &   -5.900526869409D-18, -9.41893387554D-19, 2.14977356470D-19, 
     &    4.6660985008D-20, -7.243011862D-21, -2.387966824D-21, 
     &    1.91177535D-22, 1.20482568D-22, -6.72377D-25, -5.747997D-24,
     &   -4.28493D-25, 2.44856D-25, 4.3793D-26, -8.151D-27, -3.089D-27, 
     &    9.3D-29, 1.74D-28, 1.6D-29, -8.0D-30, -2.0D-30 /       
*     
      XA = ABS(Z)/RTWO
      IF ( XA .GT. 100 ) THEN
         P = 0
      ELSE
         T = ( 8*XA - 30 ) / ( 4*XA + 15 )
         BM = 0
         B  = 0
         DO I = IM, 0, -1 
            BP = B
            B  = BM
            BM = T*B - BP  + A(I)
         END DO
         P = EXP( -XA*XA )*( BM - BP )/4
      END IF
      IF ( Z .GT. 0 ) P = 1 - P
      MVPHI = P
      END
*
