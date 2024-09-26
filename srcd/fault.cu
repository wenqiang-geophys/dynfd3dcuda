#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "params.h"
#include "macdrp.h"

//#define DEBUG

extern __device__ real_t norm3(real_t *A);
extern __device__ real_t dot_product(real_t *A, real_t *B);
extern __device__ void matmul3x1(real_t A[][3], real_t B[3], real_t C[3]);
extern __device__ void matmul3x3(real_t A[][3], real_t B[][3], real_t C[][3]);
extern __device__ void invert3x3(real_t m[][3]);
extern __device__ real_t Fr_func(const real_t r, const real_t R);
extern __device__ real_t Gt_func(const real_t t, const real_t T);

__global__
void fault_deriv_cu(Wave W, Fault F, realptr_t M,
    int FlagX, int FlagY, int FlagZ)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int j1 = j + 3;
  int k1 = k + 3;
  int i;
  //int ni = par.ni;
  int nj = par.nj;
  int nk = par.nk;

  int nx = par.nx;
  int ny = par.ny;
  int nz = par.nz;

  //real_t DH = par.DH;
  real_t rDH = par.rDH;
  //real_t DT = par.DT;
  // OUTPUT
  int stride = nx * ny * nz;
  int nyz = ny * nz;
  int nyz2 = nyz * 2;

  real_t *w_Vx  = W.W + 0 * stride;
  real_t *w_Vy  = W.W + 1 * stride;
  real_t *w_Vz  = W.W + 2 * stride;
  real_t *w_Txx = W.W + 3 * stride;
  real_t *w_Tyy = W.W + 4 * stride;
  real_t *w_Tzz = W.W + 5 * stride;
  real_t *w_Txy = W.W + 6 * stride;
  real_t *w_Txz = W.W + 7 * stride;
  real_t *w_Tyz = W.W + 8 * stride;

  real_t *w_hVx  = W.hW + 0 * stride;
  real_t *w_hVy  = W.hW + 1 * stride;
  real_t *w_hVz  = W.hW + 2 * stride;
  real_t *w_hTxx = W.hW + 3 * stride;
  real_t *w_hTyy = W.hW + 4 * stride;
  real_t *w_hTzz = W.hW + 5 * stride;
  real_t *w_hTxy = W.hW + 6 * stride;
  real_t *w_hTxz = W.hW + 7 * stride;
  real_t *w_hTyz = W.hW + 8 * stride;

  // INPUT
  real_t *XIX = M + 0 * stride;
  real_t *XIY = M + 1 * stride;
  real_t *XIZ = M + 2 * stride;
  real_t *ETX = M + 3 * stride;
  real_t *ETY = M + 4 * stride;
  real_t *ETZ = M + 5 * stride;
  real_t *ZTX = M + 6 * stride;
  real_t *ZTY = M + 7 * stride;
  real_t *ZTZ = M + 8 * stride;
  real_t *JAC = M + 9 * stride;
  real_t *LAM = M + 10 * stride;
  real_t *MIU = M + 11 * stride;
  real_t *RHO = M + 12 * stride;

  // Split nodes
  //stride = ny * nz * 2; // y vary first

  // INPUT
  real_t *f_Vx  = F.W + 0 * nyz2;
  real_t *f_Vy  = F.W + 1 * nyz2;
  real_t *f_Vz  = F.W + 2 * nyz2;
  real_t *f_T21 = F.W + 3 * nyz2;
  real_t *f_T22 = F.W + 4 * nyz2;
  real_t *f_T23 = F.W + 5 * nyz2;
  real_t *f_T31 = F.W + 6 * nyz2;
  real_t *f_T32 = F.W + 7 * nyz2;
  real_t *f_T33 = F.W + 8 * nyz2;

  real_t *f_hVx  = F.hW + 0 * nyz2;
  real_t *f_hVy  = F.hW + 1 * nyz2;
  real_t *f_hVz  = F.hW + 2 * nyz2;
  real_t *f_hT21 = F.hW + 3 * nyz2;
  real_t *f_hT22 = F.hW + 4 * nyz2;
  real_t *f_hT23 = F.hW + 5 * nyz2;
  real_t *f_hT31 = F.hW + 6 * nyz2;
  real_t *f_hT32 = F.hW + 7 * nyz2;
  real_t *f_hT33 = F.hW + 8 * nyz2;

  //int istep = it % 8;
  //int sign1 = irk % 2;
  //int FlagX = Flags[istep][0];
  //int FlagY = Flags[istep][1];
  //int FlagZ = Flags[istep][2];

  //if(sign1) { FlagX *= -1; FlagY *= -1; FlagZ *= -1; }

  //if(j==100 && k == 100)
  //printf("@fault driv Flag = %2d %2d %2d\n", FlagX, FlagY, FlagZ);

  real_t xix, xiy, xiz;
  real_t etx, ety, etz;
  real_t ztx, zty, ztz;

  real_t mu, lam, lam2mu;
  real_t rrhojac;// jac, rrho;

  real_t DxVx[8],DxVy[8],DxVz[8];
  real_t DyVx[8],DyVy[8],DyVz[8];
  real_t DzVx[8],DzVy[8],DzVz[8];

  real_t vecT11[7], vecT12[7], vecT13[7];
  real_t vecT21[7], vecT22[7], vecT23[7];
  real_t vecT31[7], vecT32[7], vecT33[7];
  real_t DxTx[3],DyTy[3],DzTz[3];

  real_t vec_3[3], vec_5[5];
  real_t mat1[3][3], mat2[3][3], mat3[3][3];
  real_t vec1[3], vec2[3], vec3[3];
  real_t vecg1[3], vecg2[3], vecg3[3];

  real_t matMin2Plus1[3][3];
  real_t matMin2Plus2[3][3];
  real_t matMin2Plus3[3][3];
  real_t matMin2Plus4[3][3];
  real_t matMin2Plus5[3][3];
  real_t matPlus2Min1[3][3];
  real_t matPlus2Min2[3][3];
  real_t matPlus2Min3[3][3];
  real_t matPlus2Min4[3][3];
  real_t matPlus2Min5[3][3];
  real_t dxV1[3], dyV1[3], dzV1[3];
  real_t dxV2[3], dyV2[3], dzV2[3];
  real_t out1[3], out2[3], out3[3], out4[3], out5[3];
  real_t matMin2Plus1f[3][3], matMin2Plus2f[3][3], matMin2Plus3f[3][3];
  real_t matPlus2Min1f[3][3], matPlus2Min2f[3][3], matPlus2Min3f[3][3];
  real_t matVx2Vz1[3][3], matVy2Vz1[3][3];
  real_t matVx2Vz2[3][3], matVy2Vz2[3][3];

  int ii, jj, mm, l, n;
  //int km;

  int pos, pos_m, slice, segment;
  int pos0, pos1, pos2;
  int idx;

  int i0 = nx/2;
//#ifdef FreeSurface
//  if (j >= 30 && j < nj-31 && k >= 30 && k < nk ) { // non united
//#else
//  if (j >= 30 && j < nj-31 && k >= 30 && k < nk-31 ) { // non united
//#endif
  if (j < nj && k < nk ) { 
    // non united
    if(F.united[j + k * nj]) return;
    //  km = NZ -(thisid[2]*nk+k-3);
    //int km = nz - k; // nk2-3, nk2-2, nk2-1 ==> (3, 2, 1)
    int km = nk - k; // nk2-3, nk2-2, nk2-1 ==> (3, 2, 1)
    // update velocity at surrounding points
    for (i = i0-3; i <= i0+3; i++) {
      //n = i - i0; // -3, -2, -1, +1, +2, +3, not at fault plane
      n = i0 - i; // +3, +2, +1, -1, -2, -3, not at fault plane
      if(n==0) continue; // skip Split nodes

      //pos_m = (i * ny * nz + j * nz + k) * MSIZE;
      //pos   = (i * ny * nz + j * nz + k) * WSIZE;
      //slice = ny * nz * WSIZE; segment = nz * WSIZE;

      //xix = M[pos_m + 0]; xiy = M[pos_m + 1]; xiz = M[pos_m + 2];
      //etx = M[pos_m + 3]; ety = M[pos_m + 4]; etz = M[pos_m + 5];
      //ztx = M[pos_m + 6]; zty = M[pos_m + 7]; ztz = M[pos_m + 8];
      //lam = M[pos_m + 10];
      //mu  = M[pos_m + 11];
      //rrho = 1.0f/M[pos_m + 12];
      //lam2mu = lam + 2.0f*mu;


      for (l = -3; l <= 3; l++){

        //pos_m = ((i+l)*ny*nz + j1*nz + k1)*MSIZE;
        //pos   = ((i+l)*ny*nz + j1*nz + k1)*WSIZE;
        //xix = M[pos_m + 0]; xiy = M[pos_m + 1]; xiz = M[pos_m + 2];
        //etx = M[pos_m + 3]; ety = M[pos_m + 4]; etz = M[pos_m + 5];
        //ztx = M[pos_m + 6]; zty = M[pos_m + 7]; ztz = M[pos_m + 8];
        //jac = M[pos_m + 9];
        //xix = XIX[pos]; xiy = XIY[pos]; xiz = XIZ[pos];
        //etx = ETX[pos]; ety = ETY[pos]; etz = ETZ[pos];
        //ztx = ZTX[pos]; zty = ZTY[pos]; ztz = ZTZ[pos];
        //jac = JAC[pos];
        // Txx 3    Tyy4    Tzz5    Txy6    Txz7     Tyz8
        // Txx Txy Txz (3 6 7)
        // Txy Tyy Tyz (6 4 8)
        // Txz Tyz Tzz (7 8 5)
        //vecT11[l+3] = jac*(xix*W.W[pos + 3] + xiy*W.W[pos + 6] + xiz*W.W[pos + 7]);
        //vecT12[l+3] = jac*(xix*W.W[pos + 6] + xiy*W.W[pos + 4] + xiz*W.W[pos + 8]);
        //vecT13[l+3] = jac*(xix*W.W[pos + 7] + xiy*W.W[pos + 8] + xiz*W.W[pos + 5]);
        //pos_m = (i*ny*nz + (j1+l)*nz + k1)*MSIZE;
        //pos   = (i*ny*nz + (j1+l)*nz + k1)*WSIZE;
        //pos_m = (i*ny*nz + j1*nz + (k1+l))*MSIZE;
        //pos   = (i*ny*nz + j1*nz + (k1+l))*WSIZE;

        pos = j1 + k1 * ny + (i+l) * ny * nz;
        vecT11[l+3] = JAC[pos]*(XIX[pos]*w_Txx[pos] + XIY[pos]*w_Txy[pos] + XIZ[pos]*w_Txz[pos]);
        vecT12[l+3] = JAC[pos]*(XIX[pos]*w_Txy[pos] + XIY[pos]*w_Tyy[pos] + XIZ[pos]*w_Tyz[pos]);
        vecT13[l+3] = JAC[pos]*(XIX[pos]*w_Txz[pos] + XIY[pos]*w_Tyz[pos] + XIZ[pos]*w_Tzz[pos]);

        // bug fixed, xi, et, zt
        pos = (j1+l) + k1 * ny + i * ny * nz;
        vecT21[l+3] = JAC[pos]*(ETX[pos]*w_Txx[pos] + ETY[pos]*w_Txy[pos] + ETZ[pos]*w_Txz[pos]);
        vecT22[l+3] = JAC[pos]*(ETX[pos]*w_Txy[pos] + ETY[pos]*w_Tyy[pos] + ETZ[pos]*w_Tyz[pos]);
        vecT23[l+3] = JAC[pos]*(ETX[pos]*w_Txz[pos] + ETY[pos]*w_Tyz[pos] + ETZ[pos]*w_Tzz[pos]);

        pos = j1 + (k1+l) * ny + i * ny * nz;
        vecT31[l+3] = JAC[pos]*(ZTX[pos]*w_Txx[pos] + ZTY[pos]*w_Txy[pos] + ZTZ[pos]*w_Txz[pos]);
        vecT32[l+3] = JAC[pos]*(ZTX[pos]*w_Txy[pos] + ZTY[pos]*w_Tyy[pos] + ZTZ[pos]*w_Tyz[pos]);
        vecT33[l+3] = JAC[pos]*(ZTX[pos]*w_Txz[pos] + ZTY[pos]*w_Tyz[pos] + ZTZ[pos]*w_Tzz[pos]);
      }

      // Traction Low
      //pos = 1*ny*nz + j1*nz + k1;
      //pos = j1 + k1 * ny + 1 * ny * nz;
      pos = j1 + k1 * ny + 3 * ny * nz;
      vecT11[n+3] = F.T11[pos];// F->T11[1][j][k];
      vecT12[n+3] = F.T12[pos];// F->T12[1][j][k];
      vecT13[n+3] = F.T13[pos];// F->T13[1][j][k];

//#ifdef TractionLow
      // reduce order
      if(abs(n)==1){
        DxTx[0] = vec_L22(vecT11,3,FlagX)*rDH;
        DxTx[1] = vec_L22(vecT12,3,FlagX)*rDH;
        DxTx[2] = vec_L22(vecT13,3,FlagX)*rDH;
      }else if(abs(n)==2){
        DxTx[0] = vec_L24(vecT11,3,FlagX)*rDH;
        DxTx[1] = vec_L24(vecT12,3,FlagX)*rDH;
        DxTx[2] = vec_L24(vecT13,3,FlagX)*rDH;
      }else if(abs(n)==3){
        DxTx[0] = vec_L(vecT11,3,FlagX)*rDH;
        DxTx[1] = vec_L(vecT12,3,FlagX)*rDH;
        DxTx[2] = vec_L(vecT13,3,FlagX)*rDH;
      }
//#endif
#ifdef TractionImg
      if (n==2) { // i0-2
        vecT11[6] = 2.0*vecT11[5] - vecT11[4];
        vecT12[6] = 2.0*vecT12[5] - vecT12[4];
        vecT13[6] = 2.0*vecT13[5] - vecT13[4];
      }
      if (n==1) { // i0-1
        vecT11[5] = 2.0*vecT11[4] - vecT11[3];
        vecT12[5] = 2.0*vecT12[4] - vecT12[3];
        vecT13[5] = 2.0*vecT13[4] - vecT13[3];
        vecT11[6] = 2.0*vecT11[4] - vecT11[2];
        vecT12[6] = 2.0*vecT12[4] - vecT12[2];
        vecT13[6] = 2.0*vecT13[4] - vecT13[2];
      }
      if (n==-1) { // i0+1
        vecT11[0] = 2.0*vecT11[2] - vecT11[4];
        vecT12[0] = 2.0*vecT12[2] - vecT12[4];
        vecT13[0] = 2.0*vecT13[2] - vecT13[4];
        vecT11[1] = 2.0*vecT11[2] - vecT11[3];
        vecT12[1] = 2.0*vecT12[2] - vecT12[3];
        vecT13[1] = 2.0*vecT13[2] - vecT13[3];
      }
      if (n==-2) { // i0+2
        vecT11[0] = 2.0*vecT11[1] - vecT11[2];
        vecT12[0] = 2.0*vecT12[1] - vecT12[2];
        vecT13[0] = 2.0*vecT13[1] - vecT13[2];
      }

      DxTx[0] = vec_L(vecT11,3,FlagX)*rDH;
      DxTx[1] = vec_L(vecT12,3,FlagX)*rDH;
      DxTx[2] = vec_L(vecT13,3,FlagX)*rDH;
#endif
      if(par.freenode && km<=3){
        //extendvect(vecT31, km+2, 0.0);
        //extendvect(vecT32, km+2, 0.0);
        //extendvect(vecT33, km+2, 0.0);
        vecT31[km+2] = 0.0;
        vecT32[km+2] = 0.0;
        vecT33[km+2] = 0.0;
        for (l = km+3; l<7; l++){
          vecT31[l] = -vecT31[2*(km+2)-l];
          vecT32[l] = -vecT32[2*(km+2)-l];
          vecT33[l] = -vecT33[2*(km+2)-l];
        }
      }

      DyTy[0] = vec_L(vecT21,3,FlagY)*rDH;
      DyTy[1] = vec_L(vecT22,3,FlagY)*rDH;
      DyTy[2] = vec_L(vecT23,3,FlagY)*rDH;
      DzTz[0] = vec_L(vecT31,3,FlagZ)*rDH;
      DzTz[1] = vec_L(vecT32,3,FlagZ)*rDH;
      DzTz[2] = vec_L(vecT33,3,FlagZ)*rDH;

      //pos = (i*ny*nz + j1*nz + k1)*WSIZE;
      //pos_m = (i*ny*nz + j1*nz + k1)*MSIZE;
      pos = j1 + k1 * ny + i * ny * nz;

      //rrho = 1.0 / RHO[pos];
      //jac = JAC[pos];
      //rrhojac = rrho/jac;
      rrhojac = 1.0 / (RHO[pos] * JAC[pos]);
      w_hVx[pos] = (DxTx[0]+DyTy[0]+DzTz[0])*rrhojac;
      w_hVy[pos] = (DxTx[1]+DyTy[1]+DzTz[1])*rrhojac;
      w_hVz[pos] = (DxTx[2]+DyTy[2]+DzTz[2])*rrhojac;

    } // end of loop i

    // update velocity at the fault plane
    // 0 for minus side on the fault
    // 1 for plus  side on the fault
    for (mm = 0; mm < 2; mm++){
      //km = NZ -(thisid[2]*nk+k-3);
      //rrhojac = F->rrhojac_f[mm][j][k];

      // one side derive
      //pos0 = 0*ny*nz + j1*nz + k1;
      //pos1 = 1*ny*nz + j1*nz + k1;
      //pos2 = 2*ny*nz + j1*nz + k1;
      //pos0 = j1 + k1 * ny + 0 * ny * nz;
      //pos1 = j1 + k1 * ny + 1 * ny * nz;
      //pos2 = j1 + k1 * ny + 2 * ny * nz;
//#ifdef TractionLow
      pos0 = j1 + k1 * ny + (3-1) * ny * nz;
      pos1 = j1 + k1 * ny + (3  ) * ny * nz;
      pos2 = j1 + k1 * ny + (3+1) * ny * nz;
      if(mm==0){
        DxTx[0] = (F.T11[pos1] - F.T11[pos0])*rDH;
        DxTx[1] = (F.T12[pos1] - F.T12[pos0])*rDH;
        DxTx[2] = (F.T13[pos1] - F.T13[pos0])*rDH;
      }else{
        DxTx[0] = (F.T11[pos2] - F.T11[pos1])*rDH;
        DxTx[1] = (F.T12[pos2] - F.T12[pos1])*rDH;
        DxTx[2] = (F.T13[pos2] - F.T13[pos1])*rDH;
      }
//#endif
#ifdef TractionImg
      real_t a0p,a0m;
      if(FlagX==FWD){
        a0p = a_0pF;
        a0m = a_0mF;
      }else{
        a0p = a_0pB;
        a0m = a_0mB;
      }
      if(mm==0){ // "-" side
        DxTx[0] = rDH*(
            a0m*F.T11[j1+k1*ny+3*ny*nz] -
            a_1*F.T11[j1+k1*ny+2*ny*nz] -
            a_2*F.T11[j1+k1*ny+1*ny*nz] -
            a_3*F.T11[j1+k1*ny+0*ny*nz] );
        DxTx[1] = rDH*(
            a0m*F.T12[j1+k1*ny+3*ny*nz] -
            a_1*F.T12[j1+k1*ny+2*ny*nz] -
            a_2*F.T12[j1+k1*ny+1*ny*nz] -
            a_3*F.T12[j1+k1*ny+0*ny*nz] );
        DxTx[2] = rDH*(
            a0m*F.T13[j1+k1*ny+3*ny*nz] -
            a_1*F.T13[j1+k1*ny+2*ny*nz] -
            a_2*F.T13[j1+k1*ny+1*ny*nz] -
            a_3*F.T13[j1+k1*ny+0*ny*nz] );
      }else{ // "+" side
        DxTx[0] = rDH*(
            a0p*F.T11[j1+k1*ny+3*ny*nz] +
            a_1*F.T11[j1+k1*ny+4*ny*nz] +
            a_2*F.T11[j1+k1*ny+5*ny*nz] +
            a_3*F.T11[j1+k1*ny+6*ny*nz] );
        DxTx[1] = rDH*(
            a0p*F.T12[j1+k1*ny+3*ny*nz] +
            a_1*F.T12[j1+k1*ny+4*ny*nz] +
            a_2*F.T12[j1+k1*ny+5*ny*nz] +
            a_3*F.T12[j1+k1*ny+6*ny*nz] );
        DxTx[2] = rDH*(
            a0p*F.T13[j1+k1*ny+3*ny*nz] +
            a_1*F.T13[j1+k1*ny+4*ny*nz] +
            a_2*F.T13[j1+k1*ny+5*ny*nz] +
            a_3*F.T13[j1+k1*ny+6*ny*nz] );
      }
#endif

      for (l = -3; l <= 3; l++){
        //pos = (mm*ny*nz + (j1+l)*nz + k1)*FSIZE;
        pos = (j1+l) + k1 * ny + mm * ny * nz;
        vecT21[l+3] = f_T21[pos];//.W[pos + 3];
        vecT22[l+3] = f_T22[pos];//.W[pos + 4];
        vecT23[l+3] = f_T23[pos];//.W[pos + 5];
        //pos = (mm*ny*nz + j1*nz + (k1+l))*FSIZE;
        pos = j1 + (k1+l) * ny + mm * ny * nz;
        vecT31[l+3] = f_T31[pos];//.W[pos + 6];
        vecT32[l+3] = f_T32[pos];//.W[pos + 7];
        vecT33[l+3] = f_T33[pos];//.W[pos + 8];
      }

      if(par.freenode && km<=3){
        vecT31[km+2] = 0.0;
        vecT32[km+2] = 0.0;
        vecT33[km+2] = 0.0;
        for (l = km+3; l<7; l++){
          vecT31[l] = -vecT31[2*(km+2)-l];
          vecT32[l] = -vecT32[2*(km+2)-l];
          vecT33[l] = -vecT33[2*(km+2)-l];
        }

        DzTz[0] = vec_L(vecT31,3,FlagZ)*rDH;
        DzTz[1] = vec_L(vecT32,3,FlagZ)*rDH;
        DzTz[2] = vec_L(vecT33,3,FlagZ)*rDH;

      }else{
        if(F.rup_index_z[j + k * nj] % 7){
          DzTz[0] = vec_L22(vecT31,3,FlagZ)*rDH;
          DzTz[1] = vec_L22(vecT32,3,FlagZ)*rDH;
          DzTz[2] = vec_L22(vecT33,3,FlagZ)*rDH;
        }else{
          DzTz[0] = vec_L(vecT31,3,FlagZ)*rDH;
          DzTz[1] = vec_L(vecT32,3,FlagZ)*rDH;
          DzTz[2] = vec_L(vecT33,3,FlagZ)*rDH;
        }
      }

      if(F.rup_index_y[j + k * nj] % 7){
        DyTy[0] = vec_L22(vecT21,3,FlagY)*rDH;
        DyTy[1] = vec_L22(vecT22,3,FlagY)*rDH;
        DyTy[2] = vec_L22(vecT23,3,FlagY)*rDH;
      }else{
        DyTy[0] = vec_L(vecT21,3,FlagY)*rDH;
        DyTy[1] = vec_L(vecT22,3,FlagY)*rDH;
        DyTy[2] = vec_L(vecT23,3,FlagY)*rDH;
      }

      //pos = (mm*ny*nz + j1*nz + k1)*FSIZE;
      //pos_m = (i0*ny*nz + j1*nz + k1)*MSIZE;
      //rrho = 1.0 / M[pos_m + 12];
      //jac = M[pos_m + 9];
      //rrhojac = rrho/jac;
      //F.hW[pos + 0] = (DxTx[0]+DyTy[0]+DzTz[0])*rrhojac;
      //F.hW[pos + 1] = (DxTx[1]+DyTy[1]+DzTz[1])*rrhojac;
      //F.hW[pos + 2] = (DxTx[2]+DyTy[2]+DzTz[2])*rrhojac;

      pos_m = j1 + k1 * ny + i0 * ny * nz;
      //rrhojac = 1.0 / (RHO[pos_m] * JAC[pos_m]);
      pos = j1 + k1 * ny + mm * ny * nz; // mm = 0, 1
      rrhojac = 1.0 / (F.rho_f[pos] * JAC[pos_m]);

      pos = j1 + k1 * ny + mm * ny * nz; // mm = 0, 1
      f_hVx[pos] = (DxTx[0]+DyTy[0]+DzTz[0])*rrhojac;
      f_hVy[pos] = (DxTx[1]+DyTy[1]+DzTz[1])*rrhojac;
      f_hVz[pos] = (DxTx[2]+DyTy[2]+DzTz[2])*rrhojac;
//#ifdef DEBUG
//      if(j==ny/2-15 && k==nz/2-15)
//        printf("@fault_deriv F.hW(mm=%d) = %e %e %e\n"
//            "DxTx = %g %g %g, DyTy = %g %g %g, DzTz = %g %g %g\n",
//            mm,
//            F.hW[pos + 0],
//            F.hW[pos + 1],
//            F.hW[pos + 2],
//            DxTx[0], DxTx[1], DxTx[2],
//            DyTy[0], DyTy[1], DyTy[2],
//            DzTz[0], DzTz[1], DzTz[2]
//            );
//#endif
    } // end of loop mm  update fault plane

    //
    //     Update Stress (in Zhang's Thesis, 2014)
    // ---V-----V-----V-----0-----0-----V-----V-----V---  (grid point)
    //    G     F     E     D-    D+    C     B     A     (grid name in thesis)
    //    i0-3  i0-2  i0-1  i0-0  i0+0  i0+1  i0+2  i0+3  (3D grid index)
    //    0     1     2     3     4     5     6     7     (vec grid index)
    //    -3    -2    -1    -0    +0    1     2     3     (offset from fault)
    //
    // Split nodes {{{
    slice = nz*FSIZE; //segment = FSIZE;
    if(F.rup_index_y[j + k * nj] % 7){
      //pos = (0*ny*nz + j1*nz + k1)*FSIZE;
      //pos = (1*ny*nz + j1*nz + k1)*FSIZE;
      pos = j1 + k1 * ny + 0 * ny * nz;
      DyVx[3] = L22(f_Vx, pos, 1, FlagY)*rDH;
      DyVy[3] = L22(f_Vy, pos, 1, FlagY)*rDH;
      DyVz[3] = L22(f_Vz, pos, 1, FlagY)*rDH;
      pos = j1 + k1 * ny + 1 * ny * nz;
      DyVx[4] = L22(f_Vx, pos, 1, FlagY)*rDH;
      DyVy[4] = L22(f_Vy, pos, 1, FlagY)*rDH;
      DyVz[4] = L22(f_Vz, pos, 1, FlagY)*rDH;
    }else{
      pos = j1 + k1 * ny + 0 * ny * nz;
      DyVx[3] = L(f_Vx, pos, 1, FlagY)*rDH;
      DyVy[3] = L(f_Vy, pos, 1, FlagY)*rDH;
      DyVz[3] = L(f_Vz, pos, 1, FlagY)*rDH;
      pos = j1 + k1 * ny + 1 * ny * nz;
      DyVx[4] = L(f_Vx, pos, 1, FlagY)*rDH;
      DyVy[4] = L(f_Vy, pos, 1, FlagY)*rDH;
      DyVz[4] = L(f_Vz, pos, 1, FlagY)*rDH;
      //pos = (0*ny*nz + j1*nz + k1)*FSIZE;
      //DyVx[3] = L(F.W, (pos + 0), slice, FlagY)*rDH;
      //DyVy[3] = L(F.W, (pos + 1), slice, FlagY)*rDH;
      //DyVz[3] = L(F.W, (pos + 2), slice, FlagY)*rDH;
      //pos = (1*ny*nz + j1*nz + k1)*FSIZE;
      //DyVx[4] = L(F.W, (pos + 0), slice, FlagY)*rDH;
      //DyVy[4] = L(F.W, (pos + 1), slice, FlagY)*rDH;
      //DyVz[4] = L(F.W, (pos + 2), slice, FlagY)*rDH;
    }

    //km = NZ -(thisid[2]*nk+k-3);
    //km = nz -(k-3);

    if(FlagX == FWD){
      // calculate forward on the plus side directly
      //pos = ((i0+1)*ny*nz + j1*nz + k1)*WSIZE;
      //pos1 = (1*ny*nz + j1*nz + k1)*FSIZE;
      //DxVx[4] = ( W.W[pos + 0] - F.W[pos1 + 0] )*rDH;
      //DxVy[4] = ( W.W[pos + 1] - F.W[pos1 + 1] )*rDH;
      //DxVz[4] = ( W.W[pos + 2] - F.W[pos1 + 2] )*rDH; // plus side
      pos = j1 + k1 * ny + (i0+1) * ny * nz; 
      pos1 = j1 + k1 * ny + 1 * ny * nz;
      DxVx[4] = ( w_Vx[pos] - f_Vx[pos1] )*rDH;
      DxVy[4] = ( w_Vy[pos] - f_Vy[pos1] )*rDH;
      DxVz[4] = ( w_Vz[pos] - f_Vz[pos1] )*rDH; // plus side

      if(par.freenode && km==1){
        // get DxV[3], DzV[3], DzV[4] {{{
        for (ii = 0; ii < 3; ii++){
          for (jj = 0; jj < 3; jj++){
            matVx2Vz1    [ii][jj] = F.matVx2Vz1    [j1*9 + ii*3 + jj];
            matVx2Vz2    [ii][jj] = F.matVx2Vz2    [j1*9 + ii*3 + jj];
            matVy2Vz1    [ii][jj] = F.matVy2Vz1    [j1*9 + ii*3 + jj];
            matVy2Vz2    [ii][jj] = F.matVy2Vz2    [j1*9 + ii*3 + jj];
            matPlus2Min1f[ii][jj] = F.matPlus2Min1f[j1*9 + ii*3 + jj];
            matPlus2Min2f[ii][jj] = F.matPlus2Min2f[j1*9 + ii*3 + jj];
            matPlus2Min3f[ii][jj] = F.matPlus2Min3f[j1*9 + ii*3 + jj];
          }
        }

      //  if(thisid[1]*nj+j-3==120){
      //    printf("matVx2Vz1\n");
      //    print_mat3x3(matVx2Vz1);
      //    printf("matVy2Vz1\n");
      //    print_mat3x3(matVy2Vz1);
      //    printf("matVx2Vz2\n");
      //    print_mat3x3(matVx2Vz2);
      //    printf("matVy2Vz2\n");
      //    print_mat3x3(matVy2Vz2);
      //    printf("matPlus2Min1f\n");
      //    print_mat3x3(matPlus2Min1f);
      //    printf("matPlus2Min2f\n");
      //    print_mat3x3(matPlus2Min2f);
      //    printf("matPlus2Min3f\n");
      //    print_mat3x3(matPlus2Min3f);
      //  }

                           dyV1[0] = DyVx[3];
                           dyV1[1] = DyVy[3];
                           dyV1[2] = DyVz[3];

        dxV2[0] = DxVx[4]; dyV2[0] = DyVx[4];
        dxV2[1] = DxVy[4]; dyV2[1] = DyVy[4];
        dxV2[2] = DxVz[4]; dyV2[2] = DyVz[4];

        matmul3x1(matPlus2Min1f, dxV2, out1);
        matmul3x1(matPlus2Min2f, dyV2, out2);
        matmul3x1(matPlus2Min3f, dyV1, out3);

        DxVx[3] = out1[0] + out2[0] - out3[0];
        DxVy[3] = out1[1] + out2[1] - out3[1];
        DxVz[3] = out1[2] + out2[2] - out3[2];

        dxV1[0] = DxVx[3];
        dxV1[1] = DxVy[3];
        dxV1[2] = DxVz[3];

        matmul3x1(matVx2Vz1, dxV1, out1);
        matmul3x1(matVy2Vz1, dyV1, out2);

        DzVx[3] = out1[0] + out2[0];
        DzVy[3] = out1[1] + out2[1];
        DzVz[3] = out1[2] + out2[2];

        matmul3x1(matVx2Vz2, dxV2, out1);
        matmul3x1(matVy2Vz2, dyV2, out2);

        DzVx[4] = out1[0] + out2[0];
        DzVy[4] = out1[1] + out2[1];
        DzVz[4] = out1[2] + out2[2];

        // }}}
      }// end of par.freenode and km == 1

      if(par.freenode && (km==2 || km==3) ){
        if(km==2){
          // get DzV[3], DzV[4] {{{
          //segment = FSIZE;
          //pos = (0*ny*nz + j1*nz + k1)*FSIZE;
          //DzVx[3] = L22(F.W, (pos + 0), segment, FlagZ)*rDH;
          //DzVy[3] = L22(F.W, (pos + 1), segment, FlagZ)*rDH;
          //DzVz[3] = L22(F.W, (pos + 2), segment, FlagZ)*rDH;
          //pos = (1*ny*nz + j1*nz + k1)*FSIZE;
          //DzVx[4] = L22(F.W, (pos + 0), segment, FlagZ)*rDH;
          //DzVy[4] = L22(F.W, (pos + 1), segment, FlagZ)*rDH;
          //DzVz[4] = L22(F.W, (pos + 2), segment, FlagZ)*rDH;
          segment = ny;
          pos = j1 + k1 * ny + 0 * ny * nz;
          DzVx[3] = L22(f_Vx, pos, segment, FlagZ)*rDH;
          DzVy[3] = L22(f_Vy, pos, segment, FlagZ)*rDH;
          DzVz[3] = L22(f_Vz, pos, segment, FlagZ)*rDH;
          pos = j1 + k1 * ny + 1 * ny * nz;
          DzVx[4] = L22(f_Vx, pos, segment, FlagZ)*rDH;
          DzVy[4] = L22(f_Vy, pos, segment, FlagZ)*rDH;
          DzVz[4] = L22(f_Vz, pos, segment, FlagZ)*rDH;
          //}}}
        }else{ // km==3
          // get DzV[3], DzV[4] {{{
          //segment = FSIZE;
          //pos = (0*ny*nz + j1*nz + k1)*FSIZE;
          //DzVx[3] = L24(F.W, (pos + 0), segment, FlagZ)*rDH;
          //DzVy[3] = L24(F.W, (pos + 1), segment, FlagZ)*rDH;
          //DzVz[3] = L24(F.W, (pos + 2), segment, FlagZ)*rDH;
          //pos = (1*ny*nz + j1*nz + k1)*FSIZE;
          //DzVx[4] = L24(F.W, (pos + 0), segment, FlagZ)*rDH;
          //DzVy[4] = L24(F.W, (pos + 1), segment, FlagZ)*rDH;
          //DzVz[4] = L24(F.W, (pos + 2), segment, FlagZ)*rDH;
          segment = ny;
          pos = j1 + k1 * ny + 0 * ny * nz;
          DzVx[3] = L24(f_Vx, pos, segment, FlagZ)*rDH;
          DzVy[3] = L24(f_Vy, pos, segment, FlagZ)*rDH;
          DzVz[3] = L24(f_Vz, pos, segment, FlagZ)*rDH;
          pos = j1 + k1 * ny + 1 * ny * nz;
          DzVx[4] = L24(f_Vx, pos, segment, FlagZ)*rDH;
          DzVy[4] = L24(f_Vy, pos, segment, FlagZ)*rDH;
          DzVz[4] = L24(f_Vz, pos, segment, FlagZ)*rDH;
          //DzVx[3] = L24z(F->Vx_f,0,j,k,FlagZ);
          //DzVy[3] = L24z(F->Vy_f,0,j,k,FlagZ);
          //DzVz[3] = L24z(F->Vz_f,0,j,k,FlagZ);
          //DzVx[4] = L24z(F->Vx_f,1,j,k,FlagZ);
          //DzVy[4] = L24z(F->Vy_f,1,j,k,FlagZ);
          //DzVz[4] = L24z(F->Vz_f,1,j,k,FlagZ);
          //}}}
        }

        //idx = (j1*nz + k1)*3*3;
        idx = (j1 + k1 * ny)*3*3;
        // get DxV[3] {{{
        for (ii = 0; ii < 3; ii++){
          for (jj = 0; jj < 3; jj++){
            matPlus2Min1[ii][jj] = F.matPlus2Min1[idx + ii*3 + jj];
            matPlus2Min2[ii][jj] = F.matPlus2Min2[idx + ii*3 + jj];
            matPlus2Min3[ii][jj] = F.matPlus2Min3[idx + ii*3 + jj];
            matPlus2Min4[ii][jj] = F.matPlus2Min4[idx + ii*3 + jj];
            matPlus2Min5[ii][jj] = F.matPlus2Min5[idx + ii*3 + jj];
          }
        }

                           dyV1[0] = DyVx[3]; dzV1[0] = DzVx[3];
                           dyV1[1] = DyVy[3]; dzV1[1] = DzVy[3];
                           dyV1[2] = DyVz[3]; dzV1[2] = DzVz[3];

        dxV2[0] = DxVx[4]; dyV2[0] = DyVx[4]; dzV2[0] = DzVx[4];
        dxV2[1] = DxVy[4]; dyV2[1] = DyVy[4]; dzV2[1] = DzVy[4];
        dxV2[2] = DxVz[4]; dyV2[2] = DyVz[4]; dzV2[2] = DzVz[4];

        matmul3x1(matPlus2Min1, dxV2, out1);
        matmul3x1(matPlus2Min2, dyV2, out2);
        matmul3x1(matPlus2Min3, dzV2, out3);
        matmul3x1(matPlus2Min4, dyV1, out4);
        matmul3x1(matPlus2Min5, dzV1, out5);

        DxVx[3] = out1[0] + out2[0] + out3[0] - out4[0] - out5[0];
        DxVy[3] = out1[1] + out2[1] + out3[1] - out4[1] - out5[1];
        DxVz[3] = out1[2] + out2[2] + out3[2] - out4[2] - out5[2];
        //}}}
      } // par.freenode and km==2 or km==3

      if(!(par.freenode && km<=3)){
        // get DzV[3], DzV[4], DxV[3] {{{
        //DzVx[3] = Lz(F->Vx_f,0,j,k,FlagZ);
        segment = ny;
        if(F.rup_index_z[j + k * nj] % 7){
          //pos = (0*ny*nz + j1*nz + k1)*FSIZE; // minus
          //DzVx[3] = L22(F.W, (pos + 0), segment, FlagZ)*rDH;
          //DzVy[3] = L22(F.W, (pos + 1), segment, FlagZ)*rDH;
          //DzVz[3] = L22(F.W, (pos + 2), segment, FlagZ)*rDH;
          //pos = (1*ny*nz + j1*nz + k1)*FSIZE; // plus
          //DzVx[4] = L22(F.W, (pos + 0), segment, FlagZ)*rDH;
          //DzVy[4] = L22(F.W, (pos + 1), segment, FlagZ)*rDH;
          //DzVz[4] = L22(F.W, (pos + 2), segment, FlagZ)*rDH;
          pos = j1 + k1 * ny + 0 * ny * nz; // minus
          DzVx[3] = L22(f_Vx, pos, segment, FlagZ)*rDH;
          DzVy[3] = L22(f_Vy, pos, segment, FlagZ)*rDH;
          DzVz[3] = L22(f_Vz, pos, segment, FlagZ)*rDH;
          pos = j1 + k1 * ny + 1 * ny * nz; // plus
          DzVx[4] = L22(f_Vx, pos, segment, FlagZ)*rDH;
          DzVy[4] = L22(f_Vy, pos, segment, FlagZ)*rDH;
          DzVz[4] = L22(f_Vz, pos, segment, FlagZ)*rDH;
        }else{
          pos = j1 + k1 * ny + 0 * ny * nz; // minus
          DzVx[3] = L(f_Vx, pos, segment, FlagZ)*rDH;
          DzVy[3] = L(f_Vy, pos, segment, FlagZ)*rDH;
          DzVz[3] = L(f_Vz, pos, segment, FlagZ)*rDH;
          pos = j1 + k1 * ny + 1 * ny * nz; // plus
          DzVx[4] = L(f_Vx, pos, segment, FlagZ)*rDH;
          DzVy[4] = L(f_Vy, pos, segment, FlagZ)*rDH;
          DzVz[4] = L(f_Vz, pos, segment, FlagZ)*rDH;
        }

        //idx = (j1*nz + k1)*3*3;
        idx = (j1 + k1 * ny)*3*3;
        for (ii = 0; ii < 3; ii++){
          for (jj = 0; jj < 3; jj++){
            matPlus2Min1[ii][jj] = F.matPlus2Min1[idx + ii*3 + jj];
            matPlus2Min2[ii][jj] = F.matPlus2Min2[idx + ii*3 + jj];
            matPlus2Min3[ii][jj] = F.matPlus2Min3[idx + ii*3 + jj];
            matPlus2Min4[ii][jj] = F.matPlus2Min4[idx + ii*3 + jj];
            matPlus2Min5[ii][jj] = F.matPlus2Min5[idx + ii*3 + jj];
          }
        }

                           dyV1[0] = DyVx[3]; dzV1[0] = DzVx[3];
                           dyV1[1] = DyVy[3]; dzV1[1] = DzVy[3];
                           dyV1[2] = DyVz[3]; dzV1[2] = DzVz[3];

        dxV2[0] = DxVx[4]; dyV2[0] = DyVx[4]; dzV2[0] = DzVx[4];
        dxV2[1] = DxVy[4]; dyV2[1] = DyVy[4]; dzV2[1] = DzVy[4];
        dxV2[2] = DxVz[4]; dyV2[2] = DyVz[4]; dzV2[2] = DzVz[4];

        matmul3x1(matPlus2Min1, dxV2, out1);
        matmul3x1(matPlus2Min2, dyV2, out2);
        matmul3x1(matPlus2Min3, dzV2, out3);
        matmul3x1(matPlus2Min4, dyV1, out4);
        matmul3x1(matPlus2Min5, dzV1, out5);

        DxVx[3] = out1[0] + out2[0] + out3[0] - out4[0] - out5[0];
        DxVy[3] = out1[1] + out2[1] + out3[1] - out4[1] - out5[1];
        DxVz[3] = out1[2] + out2[2] + out3[2] - out4[2] - out5[2];
        //}}}
      } // !(par.freenode and km<=3)
    }else{
      // calculate backward on the minus side directly
      //pos = ((i0-1)*ny*nz + j1*nz + k1)*WSIZE;
      //pos0 = (0*ny*nz + j1*nz + k1)*FSIZE;
      //DxVx[3] = ( F.W[pos0 + 0] - W.W[pos + 0] )*rDH;
      //DxVy[3] = ( F.W[pos0 + 1] - W.W[pos + 1] )*rDH;
      //DxVz[3] = ( F.W[pos0 + 2] - W.W[pos + 2] )*rDH; // minus side
      pos = j1 + k1 * ny + (i0-1) * ny * nz;
      pos0 = j1 + k1 * ny + 0 * ny * nz;
      DxVx[3] = ( f_Vx[pos0] - w_Vx[pos] )*rDH;
      DxVy[3] = ( f_Vy[pos0] - w_Vy[pos] )*rDH;
      DxVz[3] = ( f_Vz[pos0] - w_Vz[pos] )*rDH; // minus side

      if(par.freenode && km==1){
        // get DxV[4], DzV[3], DzV[4] {{{
        for (ii = 0; ii < 3; ii++){
          for (jj = 0; jj < 3; jj++){
            matVx2Vz1    [ii][jj] = F.matVx2Vz1    [j1*9 + ii*3 + jj];
            matVx2Vz2    [ii][jj] = F.matVx2Vz2    [j1*9 + ii*3 + jj];
            matVy2Vz1    [ii][jj] = F.matVy2Vz1    [j1*9 + ii*3 + jj];
            matVy2Vz2    [ii][jj] = F.matVy2Vz2    [j1*9 + ii*3 + jj];
            matMin2Plus1f[ii][jj] = F.matMin2Plus1f[j1*9 + ii*3 + jj];
            matMin2Plus2f[ii][jj] = F.matMin2Plus2f[j1*9 + ii*3 + jj];
            matMin2Plus3f[ii][jj] = F.matMin2Plus3f[j1*9 + ii*3 + jj];
          }
        }

//        if(thisid[1]*nj+j-3==120){
//          printf("matVx2Vz1\n");
//          print_mat3x3(matVx2Vz1);
//          printf("matVy2Vz1\n");
//          print_mat3x3(matVy2Vz1);
//          printf("matVx2Vz2\n");
//          print_mat3x3(matVx2Vz2);
//          printf("matVy2Vz2\n");
//          print_mat3x3(matVy2Vz2);
//          printf("matMin2Plus1f\n");
//          print_mat3x3(matMin2Plus1f);
//          printf("matMin2Plus2f\n");
//          print_mat3x3(matMin2Plus2f);
//          printf("matMin2Plus3f\n");
//          print_mat3x3(matMin2Plus3f);
//        }
//
        dxV1[0] = DxVx[3]; dyV1[0] = DyVx[3];
        dxV1[1] = DxVy[3]; dyV1[1] = DyVy[3];
        dxV1[2] = DxVz[3]; dyV1[2] = DyVz[3];

                           dyV2[0] = DyVx[4];
                           dyV2[1] = DyVy[4];
                           dyV2[2] = DyVz[4];

        matmul3x1(matMin2Plus1f, dxV1, out1);
        matmul3x1(matMin2Plus2f, dyV1, out2);
        matmul3x1(matMin2Plus3f, dyV2, out3);

        DxVx[4] = out1[0] + out2[0] - out3[0];
        DxVy[4] = out1[1] + out2[1] - out3[1];
        DxVz[4] = out1[2] + out2[2] - out3[2];

        dxV2[0] = DxVx[4];
        dxV2[1] = DxVy[4];
        dxV2[2] = DxVz[4];

        matmul3x1(matVx2Vz1, dxV1, out1);
        matmul3x1(matVy2Vz1, dyV1, out2);

        DzVx[3] = out1[0] + out2[0];
        DzVy[3] = out1[1] + out2[1];
        DzVz[3] = out1[2] + out2[2];

        matmul3x1(matVx2Vz2, dxV2, out1);
        matmul3x1(matVy2Vz2, dyV2, out2);

        DzVx[4] = out1[0] + out2[0];
        DzVy[4] = out1[1] + out2[1];
        DzVz[4] = out1[2] + out2[2];
        //}}}
      } // end of par.freenode and km==1


      if( par.freenode && (km==2 || km==3) ){
        if(km==2){
          // get DzV[3], DzV[4] {{{
          //segment = FSIZE;
          //pos = (0*ny*nz + j1*nz + k1) * FSIZE;
          //DzVx[3] = L22(F.W, (pos + 0), segment, FlagZ)*rDH;
          //DzVy[3] = L22(F.W, (pos + 1), segment, FlagZ)*rDH;
          //DzVz[3] = L22(F.W, (pos + 2), segment, FlagZ)*rDH;
          //pos = (1*ny*nz + j1*nz + k1) * FSIZE;
          //DzVx[4] = L22(F.W, (pos + 0), segment, FlagZ)*rDH;
          //DzVy[4] = L22(F.W, (pos + 1), segment, FlagZ)*rDH;
          //DzVz[4] = L22(F.W, (pos + 2), segment, FlagZ)*rDH;
          segment = ny;
          pos = j1 + k1 * ny + 0 * ny * nz;
          DzVx[3] = L22(f_Vx, pos, segment, FlagZ)*rDH;
          DzVy[3] = L22(f_Vy, pos, segment, FlagZ)*rDH;
          DzVz[3] = L22(f_Vz, pos, segment, FlagZ)*rDH;
          pos = j1 + k1 * ny + 1 * ny * nz;
          DzVx[4] = L22(f_Vx, pos, segment, FlagZ)*rDH;
          DzVy[4] = L22(f_Vy, pos, segment, FlagZ)*rDH;
          DzVz[4] = L22(f_Vz, pos, segment, FlagZ)*rDH;
          //DzVx[3] = L22z(F->Vx_f,0,j,k,FlagZ);
          //DzVy[3] = L22z(F->Vy_f,0,j,k,FlagZ);
          //DzVz[3] = L22z(F->Vz_f,0,j,k,FlagZ);
          //DzVx[4] = L22z(F->Vx_f,1,j,k,FlagZ);
          //DzVy[4] = L22z(F->Vy_f,1,j,k,FlagZ);
          //DzVz[4] = L22z(F->Vz_f,1,j,k,FlagZ);
          //}}}
        }else{ // km==3
          // get DzV[3], DzV[4] {{{
          segment = ny;
          pos = j1 + k1 * ny + 0 * ny * nz;
          DzVx[3] = L24(f_Vx, pos, segment, FlagZ)*rDH;
          DzVy[3] = L24(f_Vy, pos, segment, FlagZ)*rDH;
          DzVz[3] = L24(f_Vz, pos, segment, FlagZ)*rDH;
          pos = j1 + k1 * ny + 1 * ny * nz;
          DzVx[4] = L24(f_Vx, pos, segment, FlagZ)*rDH;
          DzVy[4] = L24(f_Vy, pos, segment, FlagZ)*rDH;
          DzVz[4] = L24(f_Vz, pos, segment, FlagZ)*rDH;
          //segment = FSIZE;
          //pos = (0*ny*nz + j1*nz + k1) * FSIZE;
          //DzVx[3] = L24(F.W, (pos + 0), segment, FlagZ)*rDH;
          //DzVy[3] = L24(F.W, (pos + 1), segment, FlagZ)*rDH;
          //DzVz[3] = L24(F.W, (pos + 2), segment, FlagZ)*rDH;
          //pos = (1*ny*nz + j1*nz + k1) * FSIZE;
          //DzVx[4] = L24(F.W, (pos + 0), segment, FlagZ)*rDH;
          //DzVy[4] = L24(F.W, (pos + 1), segment, FlagZ)*rDH;
          //DzVz[4] = L24(F.W, (pos + 2), segment, FlagZ)*rDH;
          //}}}
        }

        // get DxV[4] {{{
        //idx = (j1*nz + k1)*3*3;
        idx = (j1 + k1 * ny)*3*3;
        for (ii = 0; ii < 3; ii++){
          for (jj = 0; jj < 3; jj++){
            matMin2Plus1[ii][jj] = F.matMin2Plus1[idx + ii*3 + jj];
            matMin2Plus2[ii][jj] = F.matMin2Plus2[idx + ii*3 + jj];
            matMin2Plus3[ii][jj] = F.matMin2Plus3[idx + ii*3 + jj];
            matMin2Plus4[ii][jj] = F.matMin2Plus4[idx + ii*3 + jj];
            matMin2Plus5[ii][jj] = F.matMin2Plus5[idx + ii*3 + jj];
          }
        }

        dxV1[0] = DxVx[3]; dyV1[0] = DyVx[3]; dzV1[0] = DzVx[3];
        dxV1[1] = DxVy[3]; dyV1[1] = DyVy[3]; dzV1[1] = DzVy[3];
        dxV1[2] = DxVz[3]; dyV1[2] = DyVz[3]; dzV1[2] = DzVz[3];

                           dyV2[0] = DyVx[4]; dzV2[0] = DzVx[4];
                           dyV2[1] = DyVy[4]; dzV2[1] = DzVy[4];
                           dyV2[2] = DyVz[4]; dzV2[2] = DzVz[4];

        matmul3x1(matMin2Plus1, dxV1, out1);
        matmul3x1(matMin2Plus2, dyV1, out2);
        matmul3x1(matMin2Plus3, dzV1, out3);
        matmul3x1(matMin2Plus4, dyV2, out4);
        matmul3x1(matMin2Plus5, dzV2, out5);

        DxVx[4] = out1[0] + out2[0] + out3[0] - out4[0] - out5[0];
        DxVy[4] = out1[1] + out2[1] + out3[1] - out4[1] - out5[1];
        DxVz[4] = out1[2] + out2[2] + out3[2] - out4[2] - out5[2];
        //}}}
      } // par.freenode and (km== 2 or 3)

      if(!(par.freenode && km<=3)){
      //if(1){
        // get DzV[3], DzV[4], DxV[4] {{{
        //DzVx[3] = Lz(F->Vx_f,0,j,k,FlagZ);
        //pos0 = (0*ny*nz + j1*nz + k1)*FSIZE;
        //pos1 = (1*ny*nz + j1*nz + k1)*FSIZE;
        pos0 = j1 + k1 * ny + 0 * ny * nz;
        pos1 = j1 + k1 * ny + 1 * ny * nz;
        segment = ny;
        if(F.rup_index_z[j + k * nj] % 7){
          DzVx[3] = L22(f_Vx, pos0, segment, FlagZ)*rDH;
          DzVy[3] = L22(f_Vy, pos0, segment, FlagZ)*rDH;
          DzVz[3] = L22(f_Vz, pos0, segment, FlagZ)*rDH;
          DzVx[4] = L22(f_Vx, pos1, segment, FlagZ)*rDH;
          DzVy[4] = L22(f_Vy, pos1, segment, FlagZ)*rDH;
          DzVz[4] = L22(f_Vz, pos1, segment, FlagZ)*rDH;
        }else{
          DzVx[3] = L(f_Vx, pos0, segment, FlagZ)*rDH;
          DzVy[3] = L(f_Vy, pos0, segment, FlagZ)*rDH;
          DzVz[3] = L(f_Vz, pos0, segment, FlagZ)*rDH;
          DzVx[4] = L(f_Vx, pos1, segment, FlagZ)*rDH;
          DzVy[4] = L(f_Vy, pos1, segment, FlagZ)*rDH;
          DzVz[4] = L(f_Vz, pos1, segment, FlagZ)*rDH;
          //DzVx[3] = L(F.W, (pos0 + 0), segment, FlagZ)*rDH;
          //DzVy[3] = L(F.W, (pos0 + 1), segment, FlagZ)*rDH;
          //DzVz[3] = L(F.W, (pos0 + 2), segment, FlagZ)*rDH;
          //DzVx[4] = L(F.W, (pos1 + 0), segment, FlagZ)*rDH;
          //DzVy[4] = L(F.W, (pos1 + 1), segment, FlagZ)*rDH;
          //DzVz[4] = L(F.W, (pos1 + 2), segment, FlagZ)*rDH;
        }

        //idx = (j1*nz + k1)*3*3;
        idx = (j1 + k1 * ny)*3*3;
        for (ii = 0; ii < 3; ii++){
          for (jj = 0; jj < 3; jj++){
            matMin2Plus1[ii][jj] = F.matMin2Plus1[idx + ii*3 + jj];
            matMin2Plus2[ii][jj] = F.matMin2Plus2[idx + ii*3 + jj];
            matMin2Plus3[ii][jj] = F.matMin2Plus3[idx + ii*3 + jj];
            matMin2Plus4[ii][jj] = F.matMin2Plus4[idx + ii*3 + jj];
            matMin2Plus5[ii][jj] = F.matMin2Plus5[idx + ii*3 + jj];
          }
        }

        dxV1[0] = DxVx[3]; dyV1[0] = DyVx[3]; dzV1[0] = DzVx[3];
        dxV1[1] = DxVy[3]; dyV1[1] = DyVy[3]; dzV1[1] = DzVy[3];
        dxV1[2] = DxVz[3]; dyV1[2] = DyVz[3]; dzV1[2] = DzVz[3];

                           dyV2[0] = DyVx[4]; dzV2[0] = DzVx[4];
                           dyV2[1] = DyVy[4]; dzV2[1] = DzVy[4];
                           dyV2[2] = DyVz[4]; dzV2[2] = DzVz[4];

        matmul3x1(matMin2Plus1, dxV1, out1);
        matmul3x1(matMin2Plus2, dyV1, out2);
        matmul3x1(matMin2Plus3, dzV1, out3);
        matmul3x1(matMin2Plus4, dyV2, out4);
        matmul3x1(matMin2Plus5, dzV2, out5);

        DxVx[4] = out1[0] + out2[0] + out3[0] - out4[0] - out5[0];
        DxVy[4] = out1[1] + out2[1] + out3[1] - out4[1] - out5[1];
        DxVz[4] = out1[2] + out2[2] + out3[2] - out4[2] - out5[2];
        //}}}
      } // end of !(par.freenode and km<=3)
    } // end if of FlagX

    // calculate F->hT2, F->hT3 on the Minus side {{{
    //############################### Minus ####################################
    vec1[0] = DxVx[3]; vec1[1] = DxVy[3]; vec1[2] = DxVz[3];
    vec2[0] = DyVx[3]; vec2[1] = DyVy[3]; vec2[2] = DyVz[3];
    vec3[0] = DzVx[3]; vec3[1] = DzVy[3]; vec3[2] = DzVz[3];
    ////////////////////////////////////////////////////////////////////////////
    //idx = (j1*nz + k1)*3*3;
    idx = (j1 + k1 * ny)*3*3;
    for (ii = 0; ii < 3; ii++){
      for (jj = 0; jj < 3; jj++){
        mat1[ii][jj] = F.D21_1[idx + ii*3 + jj];
        mat2[ii][jj] = F.D22_1[idx + ii*3 + jj];
        mat3[ii][jj] = F.D23_1[idx + ii*3 + jj];
      }
    }

    matmul3x1(mat1, vec1, vecg1);
    matmul3x1(mat2, vec2, vecg2);
    matmul3x1(mat3, vec3, vecg3);

    //pos0 = (0*ny*nz + j1*nz + k1)*FSIZE;
    pos0 = j1 + k1 * ny + 0 * ny * nz;
    //F->hT21[0][j][k] = vecg1[0] + vecg2[0] + vecg3[0];
    //F.hW[pos0 + 3] = vecg1[0] + vecg2[0] + vecg3[0];
    //F.hW[pos0 + 4] = vecg1[1] + vecg2[1] + vecg3[1];
    //F.hW[pos0 + 5] = vecg1[2] + vecg2[2] + vecg3[2];
    f_hT21[pos0] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT22[pos0] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT23[pos0] = vecg1[2] + vecg2[2] + vecg3[2];

    ////////////////////////////////////////////////////////////////////////////
    //idx = (j1*nz + k1)*3*3;
    idx = (j1 + k1 * ny)*3*3;
    for (ii = 0; ii < 3; ii++){
      for (jj = 0; jj < 3; jj++){
        mat1[ii][jj] = F.D31_1[idx + ii*3 + jj];
        mat2[ii][jj] = F.D32_1[idx + ii*3 + jj];
        mat3[ii][jj] = F.D33_1[idx + ii*3 + jj];
      }
    }

    matmul3x1(mat1, vec1, vecg1);
    matmul3x1(mat2, vec2, vecg2);
    matmul3x1(mat3, vec3, vecg3);

    //pos0 = (0*ny*nz + j1*nz + k1)*FSIZE;
    pos0 = j1 + k1 * ny + 0 * ny * nz;
    //F->hT31[0][j][k] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT31[pos0] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT32[pos0] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT33[pos0] = vecg1[2] + vecg2[2] + vecg3[2];
    // }}}

    // calculate F->hT2, F->hT3 on the Plus side {{{
    //############################### Plus  ####################################
    vec1[0] = DxVx[4]; vec1[1] = DxVy[4]; vec1[2] = DxVz[4];
    vec2[0] = DyVx[4]; vec2[1] = DyVy[4]; vec2[2] = DyVz[4];
    vec3[0] = DzVx[4]; vec3[1] = DzVy[4]; vec3[2] = DzVz[4];
    ////////////////////////////////////////////////////////////////////////////
    //idx = (j1*nz + k1)*3*3;
    idx = (j1 + k1 * ny)*3*3;
    for (ii = 0; ii < 3; ii++){
      for (jj = 0; jj < 3; jj++){
        mat1[ii][jj] = F.D21_2[idx + ii*3 + jj];
        mat2[ii][jj] = F.D22_2[idx + ii*3 + jj];
        mat3[ii][jj] = F.D23_2[idx + ii*3 + jj];
      }
    }

    matmul3x1(mat1, vec1, vecg1);
    matmul3x1(mat2, vec2, vecg2);
    matmul3x1(mat3, vec3, vecg3);

    //pos1 = (1*ny*nz + j1*nz + k1)*FSIZE;
    pos1 = j1 + k1 * ny + 1 * ny * nz;
    f_hT21[pos1] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT22[pos1] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT23[pos1] = vecg1[2] + vecg2[2] + vecg3[2];
    //F.hW[pos1 + 3] = vecg1[0] + vecg2[0] + vecg3[0];
    //F.hW[pos1 + 4] = vecg1[1] + vecg2[1] + vecg3[1];
    //F.hW[pos1 + 5] = vecg1[2] + vecg2[2] + vecg3[2];

    ////////////////////////////////////////////////////////////////////////////
    //idx = (j1*nz + k1)*3*3;
    idx = (j1 + k1 * ny)*3*3;
    for (ii = 0; ii < 3; ii++){
      for (jj = 0; jj < 3; jj++){
        mat1[ii][jj] = F.D31_2[idx + ii*3 + jj];
        mat2[ii][jj] = F.D32_2[idx + ii*3 + jj];
        mat3[ii][jj] = F.D33_2[idx + ii*3 + jj];
      }
    }

    matmul3x1(mat1, vec1, vecg1);
    matmul3x1(mat2, vec2, vecg2);
    matmul3x1(mat3, vec3, vecg3);

    //pos1 = (1*ny*nz + j1*nz + k1)*FSIZE;
    pos1 = j1 + k1 * ny + 1 * ny * nz;
    f_hT31[pos1] = vecg1[0] + vecg2[0] + vecg3[0];
    f_hT32[pos1] = vecg1[1] + vecg2[1] + vecg3[1];
    f_hT33[pos1] = vecg1[2] + vecg2[2] + vecg3[2];
    //F.hW[pos1 + 6] = vecg1[0] + vecg2[0] + vecg3[0];
    //F.hW[pos1 + 7] = vecg1[1] + vecg2[1] + vecg3[1];
    //F.hW[pos1 + 8] = vecg1[2] + vecg2[2] + vecg3[2];
    // Split nodes  }}}

    ////////////////////////////////////////////////////////////////////////////
    //    now we begin to update surrounding points
    ////////////////////////////////////////////////////////////////////////////

    //n = 0; l = -3; get DxV[0] {{{
    if(FlagX == FWD){
      //pos = ((i0-4)*ny*nz + j1*nz + k1)*WSIZE; vec_5[0] = W.W[pos + 0];
      //pos = ((i0-3)*ny*nz + j1*nz + k1)*WSIZE; vec_5[1] = W.W[pos + 0];
      //pos = ((i0-2)*ny*nz + j1*nz + k1)*WSIZE; vec_5[2] = W.W[pos + 0];
      //pos = ((i0-1)*ny*nz + j1*nz + k1)*WSIZE; vec_5[3] = W.W[pos + 0];
      //pos = (    0 *ny*nz + j1*nz + k1)*FSIZE; vec_5[4] = F.W[pos + 0];
      pos = j1 + k1 * ny + (i0-4)*ny*nz; vec_5[0] = w_Vx[pos];
      pos = j1 + k1 * ny + (i0-3)*ny*nz; vec_5[1] = w_Vx[pos];
      pos = j1 + k1 * ny + (i0-2)*ny*nz; vec_5[2] = w_Vx[pos];
      pos = j1 + k1 * ny + (i0-1)*ny*nz; vec_5[3] = w_Vx[pos];
      pos = j1 + k1 * ny + (   0)*ny*nz; vec_5[4] = f_Vx[pos];
      DxVx[0] = vec_LF(vec_5,1)*rDH;

      pos = j1 + k1 * ny + (i0-4)*ny*nz; vec_5[0] = w_Vy[pos];
      pos = j1 + k1 * ny + (i0-3)*ny*nz; vec_5[1] = w_Vy[pos];
      pos = j1 + k1 * ny + (i0-2)*ny*nz; vec_5[2] = w_Vy[pos];
      pos = j1 + k1 * ny + (i0-1)*ny*nz; vec_5[3] = w_Vy[pos];
      pos = j1 + k1 * ny + (   0)*ny*nz; vec_5[4] = f_Vy[pos];
      DxVy[0] = vec_LF(vec_5,1)*rDH;

      pos = j1 + k1 * ny + (i0-4)*ny*nz; vec_5[0] = w_Vz[pos];
      pos = j1 + k1 * ny + (i0-3)*ny*nz; vec_5[1] = w_Vz[pos];
      pos = j1 + k1 * ny + (i0-2)*ny*nz; vec_5[2] = w_Vz[pos];
      pos = j1 + k1 * ny + (i0-1)*ny*nz; vec_5[3] = w_Vz[pos];
      pos = j1 + k1 * ny + (   0)*ny*nz; vec_5[4] = f_Vz[pos];
      DxVz[0] = vec_LF(vec_5,1)*rDH;
      //pos = ((i0-4)*ny*nz + j1*nz + k1)*WSIZE; vec_5[0] = W.W[pos + 1];
      //pos = ((i0-3)*ny*nz + j1*nz + k1)*WSIZE; vec_5[1] = W.W[pos + 1];
      //pos = ((i0-2)*ny*nz + j1*nz + k1)*WSIZE; vec_5[2] = W.W[pos + 1];
      //pos = ((i0-1)*ny*nz + j1*nz + k1)*WSIZE; vec_5[3] = W.W[pos + 1];
      //pos = (    0 *ny*nz + j1*nz + k1)*FSIZE; vec_5[4] = F.W[pos + 1];
      //DxVy[0] = vec_LF(vec_5,1)*rDH;

      //pos = ((i0-4)*ny*nz + j1*nz + k1)*WSIZE; vec_5[0] = W.W[pos + 2];
      //pos = ((i0-3)*ny*nz + j1*nz + k1)*WSIZE; vec_5[1] = W.W[pos + 2];
      //pos = ((i0-2)*ny*nz + j1*nz + k1)*WSIZE; vec_5[2] = W.W[pos + 2];
      //pos = ((i0-1)*ny*nz + j1*nz + k1)*WSIZE; vec_5[3] = W.W[pos + 2];
      //pos = (    0 *ny*nz + j1*nz + k1)*FSIZE; vec_5[4] = F.W[pos + 2];
      //DxVz[0] = vec_LF(vec_5,1)*rDH;
    }else{
      //DxVx[0] = LxB(W->Vx,i0-3,j,k);
      //pos = ((i0-3)*ny*nz + j1*nz + k1)*WSIZE;
      //slice = ny*nz*WSIZE;
      //DxVx[0] = LB(W.W, (pos + 0), slice)*rDH;
      //DxVy[0] = LB(W.W, (pos + 1), slice)*rDH;
      //DxVz[0] = LB(W.W, (pos + 2), slice)*rDH;
      pos = j1 + k1 * ny + (i0-3) * ny * nz;
      slice = ny*nz;
      DxVx[0] = LB(w_Vx, pos, slice)*rDH;
      DxVy[0] = LB(w_Vy, pos, slice)*rDH;
      DxVz[0] = LB(w_Vz, pos, slice)*rDH;
    } // }}}

    //n = 1; l = -2; get DxV[1] {{{
    if(FlagX == FWD){
      //pos = ((i0-2)*ny*nz + j1*nz + k1)*WSIZE; vec_3[0] = W.W[pos + 0];
      //pos = ((i0-1)*ny*nz + j1*nz + k1)*WSIZE; vec_3[1] = W.W[pos + 0];
      //pos = (    0 *ny*nz + j1*nz + k1)*FSIZE; vec_3[2] = F.W[pos + 0];
      //DxVx[1] = vec_L24F(vec_3,0)*rDH;

      //pos = ((i0-2)*ny*nz + j1*nz + k1)*WSIZE; vec_3[0] = W.W[pos + 1];
      //pos = ((i0-1)*ny*nz + j1*nz + k1)*WSIZE; vec_3[1] = W.W[pos + 1];
      //pos = (    0 *ny*nz + j1*nz + k1)*FSIZE; vec_3[2] = F.W[pos + 1];
      //DxVy[1] = vec_L24F(vec_3,0)*rDH;

      //pos = ((i0-2)*ny*nz + j1*nz + k1)*WSIZE; vec_3[0] = W.W[pos + 2];
      //pos = ((i0-1)*ny*nz + j1*nz + k1)*WSIZE; vec_3[1] = W.W[pos + 2];
      //pos = (    0 *ny*nz + j1*nz + k1)*FSIZE; vec_3[2] = F.W[pos + 2];
      //DxVz[1] = vec_L24F(vec_3,0)*rDH;
      pos = j1 + k1 * ny + (i0-2) * ny * nz; vec_3[0] = w_Vx[pos];
      pos = j1 + k1 * ny + (i0-1) * ny * nz; vec_3[1] = w_Vx[pos];
      pos = j1 + k1 * ny + (   0) * ny * nz; vec_3[2] = f_Vx[pos];
      DxVx[1] = vec_L24F(vec_3,0)*rDH;

      pos = j1 + k1 * ny + (i0-2) * ny * nz; vec_3[0] = w_Vy[pos];
      pos = j1 + k1 * ny + (i0-1) * ny * nz; vec_3[1] = w_Vy[pos];
      pos = j1 + k1 * ny + (   0) * ny * nz; vec_3[2] = f_Vy[pos];
      DxVy[1] = vec_L24F(vec_3,0)*rDH;

      pos = j1 + k1 * ny + (i0-2) * ny * nz; vec_3[0] = w_Vz[pos];
      pos = j1 + k1 * ny + (i0-1) * ny * nz; vec_3[1] = w_Vz[pos];
      pos = j1 + k1 * ny + (   0) * ny * nz; vec_3[2] = f_Vz[pos];
      DxVz[1] = vec_L24F(vec_3,0)*rDH;
    }else{
      //pos = ((i0-2)*ny*nz + j1*nz + k1)*WSIZE;
      //slice = ny*nz*WSIZE;
      //DxVx[1] = L24B(W.W, (pos + 0), slice)*rDH;
      //DxVy[1] = L24B(W.W, (pos + 1), slice)*rDH;
      //DxVz[1] = L24B(W.W, (pos + 2), slice)*rDH;
      pos = j1 + k1 * ny + (i0-2) * ny * nz;
      slice = ny*nz;
      DxVx[1] = L24B(w_Vx, pos, slice)*rDH;
      DxVy[1] = L24B(w_Vy, pos, slice)*rDH;
      DxVz[1] = L24B(w_Vz, pos, slice)*rDH;
    } //}}}

    //n = 2; l = -1; get DxV[2] {{{
    if(FlagX == FWD){
      //pos = (0*ny*nz + j1*nz + k1)*FSIZE;
      //pos1 = ((i0-1)*ny*nz + j1*nz + k1)*WSIZE;
      //DxVx[2] = (F.W[pos + 0] - W.W[pos1 + 0])*rDH; // forward
      //DxVy[2] = (F.W[pos + 1] - W.W[pos1 + 1])*rDH; // forward
      //DxVz[2] = (F.W[pos + 2] - W.W[pos1 + 2])*rDH; // forward
      pos = j1 + k1 * ny + 0 * ny * nz;
      pos1 = j1 + k1 * ny + (i0-1) * ny * nz;
      DxVx[2] = (f_Vx[pos] - w_Vx[pos1])*rDH; // forward
      DxVy[2] = (f_Vy[pos] - w_Vy[pos1])*rDH; // forward
      DxVz[2] = (f_Vz[pos] - w_Vz[pos1])*rDH; // forward
    }else{
      pos = j1 + k1 * ny + (i0-1) * ny * nz;
      slice = ny*nz;
      DxVx[2] = L22B(w_Vx, pos, slice)*rDH;
      DxVy[2] = L22B(w_Vy, pos, slice)*rDH;
      DxVz[2] = L22B(w_Vz, pos, slice)*rDH;
      //pos = ((i0-1)*ny*nz + j1*nz + k1)*WSIZE;
      //slice = ny*nz*WSIZE;
      //DxVx[2] = L22B(W.W, (pos + 0), slice)*rDH;
      //DxVy[2] = L22B(W.W, (pos + 1), slice)*rDH;
      //DxVz[2] = L22B(W.W, (pos + 2), slice)*rDH;
    } //}}}

    // n = 5; l = +1; get DxV[5] {{{
    if(FlagX == FWD){
      //pos = ((i0+1)*ny*nz + j1*nz + k1)*WSIZE;
      //slice = ny*nz*WSIZE;
      //DxVx[5] = L22F(W.W, (pos + 0), slice)*rDH;;
      //DxVy[5] = L22F(W.W, (pos + 1), slice)*rDH;;
      //DxVz[5] = L22F(W.W, (pos + 2), slice)*rDH;;
      pos = j1 + k1 * ny + (i0+1) * ny * nz;
      slice = ny*nz;
      DxVx[5] = L22F(w_Vx, pos, slice)*rDH;;
      DxVy[5] = L22F(w_Vy, pos, slice)*rDH;;
      DxVz[5] = L22F(w_Vz, pos, slice)*rDH;;
    }else{
      pos = j1 + k1 * ny + 1 * ny * nz;
      pos1 = j1 + k1 * ny + (i0+1) * ny * nz;
      DxVx[5] = (w_Vx[pos1] - f_Vx[pos])*rDH; // backward
      DxVy[5] = (w_Vy[pos1] - f_Vy[pos])*rDH; // backward
      DxVz[5] = (w_Vz[pos1] - f_Vz[pos])*rDH; // backward
      //pos = (1*ny*nz + j1*nz + k1)*FSIZE;
      //pos1 = ((i0+1)*ny*nz + j1*nz + k1)*WSIZE;
      //DxVx[5] = (W.W[pos1 + 0] - F.W[pos + 0])*rDH; // backward
      //DxVy[5] = (W.W[pos1 + 1] - F.W[pos + 1])*rDH; // backward
      //DxVz[5] = (W.W[pos1 + 2] - F.W[pos + 2])*rDH; // backward
    } //}}}

    //n = 6; l = +2; get DxV[6] {{{
    if(FlagX == FWD){
      //pos = ((i0+2)*ny*nz + j1*nz + k1)*WSIZE;
      //slice = ny*nz*WSIZE;
      //DxVx[6] = L24F(W.W, (pos + 0), slice)*rDH;
      //DxVy[6] = L24F(W.W, (pos + 1), slice)*rDH;
      //DxVz[6] = L24F(W.W, (pos + 2), slice)*rDH;
      pos = j1 + k1 * ny + (i0+2) * ny * nz;
      slice = ny*nz;
      DxVx[6] = L24F(w_Vx, pos, slice)*rDH;
      DxVy[6] = L24F(w_Vy, pos, slice)*rDH;
      DxVz[6] = L24F(w_Vz, pos, slice)*rDH;
    }else{
      pos = j1 + k1 * ny + (   1) * ny * nz; vec_3[0] = f_Vx[pos];
      pos = j1 + k1 * ny + (i0+1) * ny * nz; vec_3[1] = w_Vx[pos];
      pos = j1 + k1 * ny + (i0+2) * ny * nz; vec_3[2] = w_Vx[pos];
      DxVx[6] = vec_L24B(vec_3,2)*rDH;

      pos = j1 + k1 * ny + (   1) * ny * nz; vec_3[0] = f_Vy[pos];
      pos = j1 + k1 * ny + (i0+1) * ny * nz; vec_3[1] = w_Vy[pos];
      pos = j1 + k1 * ny + (i0+2) * ny * nz; vec_3[2] = w_Vy[pos];
      DxVy[6] = vec_L24B(vec_3,2)*rDH;

      pos = j1 + k1 * ny + (   1) * ny * nz; vec_3[0] = f_Vz[pos];
      pos = j1 + k1 * ny + (i0+1) * ny * nz; vec_3[1] = w_Vz[pos];
      pos = j1 + k1 * ny + (i0+2) * ny * nz; vec_3[2] = w_Vz[pos];
      DxVz[6] = vec_L24B(vec_3,2)*rDH;
      //pos = (    1 *ny*nz + j1*nz + k1)*FSIZE; vec_3[0] = F.W[pos + 0];
      //pos = ((i0+1)*ny*nz + j1*nz + k1)*WSIZE; vec_3[1] = W.W[pos + 0];
      //pos = ((i0+2)*ny*nz + j1*nz + k1)*WSIZE; vec_3[2] = W.W[pos + 0];
      //DxVx[6] = vec_L24B(vec_3,2)*rDH;

      //pos = (    1 *ny*nz + j1*nz + k1)*FSIZE; vec_3[0] = F.W[pos + 1];
      //pos = ((i0+1)*ny*nz + j1*nz + k1)*WSIZE; vec_3[1] = W.W[pos + 1];
      //pos = ((i0+2)*ny*nz + j1*nz + k1)*WSIZE; vec_3[2] = W.W[pos + 1];
      //DxVy[6] = vec_L24B(vec_3,2)*rDH;

      //pos = (    1 *ny*nz + j1*nz + k1)*FSIZE; vec_3[0] = F.W[pos + 2];
      //pos = ((i0+1)*ny*nz + j1*nz + k1)*WSIZE; vec_3[1] = W.W[pos + 2];
      //pos = ((i0+2)*ny*nz + j1*nz + k1)*WSIZE; vec_3[2] = W.W[pos + 2];
      //DxVz[6] = vec_L24B(vec_3,2)*rDH;
    } // }}}

    //n = 7; l = +3; get DxV[7] {{{
    if(FlagX == FWD){
      //pos = ((i0+3)*ny*nz + j1*nz + k1)*WSIZE;
      //slice = ny*nz*WSIZE;
      //DxVx[7] = LF(W.W, (pos + 0), slice)*rDH;
      //DxVy[7] = LF(W.W, (pos + 1), slice)*rDH;
      //DxVz[7] = LF(W.W, (pos + 2), slice)*rDH;
      pos = j1 + k1 * ny + (i0+3) * ny * nz;
      slice = ny*nz;
      DxVx[7] = LF(w_Vx, pos, slice)*rDH;
      DxVy[7] = LF(w_Vy, pos, slice)*rDH;
      DxVz[7] = LF(w_Vz, pos, slice)*rDH;
    }else{
      pos = j1 + k1 * ny + (   1) * ny * nz; vec_5[0] = f_Vx[pos];
      pos = j1 + k1 * ny + (i0+1) * ny * nz; vec_5[1] = w_Vx[pos];
      pos = j1 + k1 * ny + (i0+2) * ny * nz; vec_5[2] = w_Vx[pos];
      pos = j1 + k1 * ny + (i0+3) * ny * nz; vec_5[3] = w_Vx[pos];
      pos = j1 + k1 * ny + (i0+4) * ny * nz; vec_5[4] = w_Vx[pos];
      DxVx[7] = vec_LB(vec_5,3)*rDH;

      pos = j1 + k1 * ny + (   1) * ny * nz; vec_5[0] = f_Vy[pos];
      pos = j1 + k1 * ny + (i0+1) * ny * nz; vec_5[1] = w_Vy[pos];
      pos = j1 + k1 * ny + (i0+2) * ny * nz; vec_5[2] = w_Vy[pos];
      pos = j1 + k1 * ny + (i0+3) * ny * nz; vec_5[3] = w_Vy[pos];
      pos = j1 + k1 * ny + (i0+4) * ny * nz; vec_5[4] = w_Vy[pos];
      DxVy[7] = vec_LB(vec_5,3)*rDH;

      // fix bug here, typo
      pos = j1 + k1 * ny + (   1) * ny * nz; vec_5[0] = f_Vz[pos];
      pos = j1 + k1 * ny + (i0+1) * ny * nz; vec_5[1] = w_Vz[pos];
      pos = j1 + k1 * ny + (i0+2) * ny * nz; vec_5[2] = w_Vz[pos];
      pos = j1 + k1 * ny + (i0+3) * ny * nz; vec_5[3] = w_Vz[pos];
      pos = j1 + k1 * ny + (i0+4) * ny * nz; vec_5[4] = w_Vz[pos];
      DxVz[7] = vec_LB(vec_5,3)*rDH;

      //pos = (    1 *ny*nz + j1*nz + k1)*FSIZE; vec_5[0] = F.W[pos + 0];
      //pos = ((i0+1)*ny*nz + j1*nz + k1)*WSIZE; vec_5[1] = W.W[pos + 0];
      //pos = ((i0+2)*ny*nz + j1*nz + k1)*WSIZE; vec_5[2] = W.W[pos + 0];
      //pos = ((i0+3)*ny*nz + j1*nz + k1)*WSIZE; vec_5[3] = W.W[pos + 0];
      //pos = ((i0+4)*ny*nz + j1*nz + k1)*WSIZE; vec_5[4] = W.W[pos + 0];
      //DxVx[7] = vec_LB(vec_5,3)*rDH;

      //pos = (    1 *ny*nz + j1*nz + k1)*FSIZE; vec_5[0] = F.W[pos + 1];
      //pos = ((i0+1)*ny*nz + j1*nz + k1)*WSIZE; vec_5[1] = W.W[pos + 1];
      //pos = ((i0+2)*ny*nz + j1*nz + k1)*WSIZE; vec_5[2] = W.W[pos + 1];
      //pos = ((i0+3)*ny*nz + j1*nz + k1)*WSIZE; vec_5[3] = W.W[pos + 1];
      //pos = ((i0+4)*ny*nz + j1*nz + k1)*WSIZE; vec_5[4] = W.W[pos + 1];
      //DxVy[7] = vec_LB(vec_5,3)*rDH;

      //pos = (    1 *ny*nz + j1*nz + k1)*FSIZE; vec_5[0] = F.W[pos + 2];
      //pos = ((i0+1)*ny*nz + j1*nz + k1)*WSIZE; vec_5[1] = W.W[pos + 2];
      //pos = ((i0+2)*ny*nz + j1*nz + k1)*WSIZE; vec_5[2] = W.W[pos + 2];
      //pos = ((i0+3)*ny*nz + j1*nz + k1)*WSIZE; vec_5[3] = W.W[pos + 2];
      //pos = ((i0+4)*ny*nz + j1*nz + k1)*WSIZE; vec_5[4] = W.W[pos + 2];
      //DxVz[7] = vec_LB(vec_5,3)*rDH;
    } //}}}

    // update surrounding points {{{
    for (n = 0; n < 8 ; n++){
      // n =  0  1  2  3  4  5  6  7
      // l = -3 -2 -1 -0 +0 +1 +2 +3
      if(n < 4){
        l = n-3;
      }else{
        l = n-4;
      }
      if(l==0) continue; // do not update i0
      i = i0+l;

      //pos = (i*ny*nz + j1*nz + k1)*WSIZE;
      pos = j1 + k1 * ny + i * ny * nz;
      //slice = nz*WSIZE;
      //DyVx[n] = L(W.W, (pos + 0), slice, FlagY)*rDH;
      //DyVy[n] = L(W.W, (pos + 1), slice, FlagY)*rDH;
      //DyVz[n] = L(W.W, (pos + 2), slice, FlagY)*rDH;
      DyVx[n] = L(w_Vx, pos, 1, FlagY)*rDH;
      DyVy[n] = L(w_Vy, pos, 1, FlagY)*rDH;
      DyVz[n] = L(w_Vz, pos, 1, FlagY)*rDH;

      // get Dz by Dx, Dy or directly
      //idx = (i*ny + j1)*9;
      idx = (j1 + i * ny)*9;

      if(par.freenode && km==1){
        DzVx[n] = W.matVx2Vz[idx + 0*3 + 0] * DxVx[n]
                + W.matVx2Vz[idx + 0*3 + 1] * DxVy[n]
                + W.matVx2Vz[idx + 0*3 + 2] * DxVz[n]
                + W.matVy2Vz[idx + 0*3 + 0] * DyVx[n]
                + W.matVy2Vz[idx + 0*3 + 1] * DyVy[n]
                + W.matVy2Vz[idx + 0*3 + 2] * DyVz[n];

        DzVy[n] = W.matVx2Vz[idx + 1*3 + 0] * DxVx[n]
                + W.matVx2Vz[idx + 1*3 + 1] * DxVy[n]
                + W.matVx2Vz[idx + 1*3 + 2] * DxVz[n]
                + W.matVy2Vz[idx + 1*3 + 0] * DyVx[n]
                + W.matVy2Vz[idx + 1*3 + 1] * DyVy[n]
                + W.matVy2Vz[idx + 1*3 + 2] * DyVz[n];

        DzVz[n] = W.matVx2Vz[idx + 2*3 + 0] * DxVx[n]
                + W.matVx2Vz[idx + 2*3 + 1] * DxVy[n]
                + W.matVx2Vz[idx + 2*3 + 2] * DxVz[n]
                + W.matVy2Vz[idx + 2*3 + 0] * DyVx[n]
                + W.matVy2Vz[idx + 2*3 + 1] * DyVy[n]
                + W.matVy2Vz[idx + 2*3 + 2] * DyVz[n] ;
      }else if(par.freenode && km==2){
        //pos = (i*ny*nz + j1*nz + k1)*WSIZE;
        //segment = WSIZE;
        //DzVx[n] = L22(W.W, (pos + 0), segment, FlagZ)*rDH;
        //DzVy[n] = L22(W.W, (pos + 1), segment, FlagZ)*rDH;
        //DzVz[n] = L22(W.W, (pos + 2), segment, FlagZ)*rDH;
        pos = j1 + k1 * ny + i * ny * nz;
        DzVx[n] = L22(w_Vx, pos, ny, FlagZ)*rDH;
        DzVy[n] = L22(w_Vy, pos, ny, FlagZ)*rDH;
        DzVz[n] = L22(w_Vz, pos, ny, FlagZ)*rDH;
      }else if(par.freenode && km==3){
        pos = j1 + k1 * ny + i * ny * nz;
        DzVx[n] = L24(w_Vx, pos, ny, FlagZ)*rDH;
        DzVy[n] = L24(w_Vy, pos, ny, FlagZ)*rDH;
        DzVz[n] = L24(w_Vz, pos, ny, FlagZ)*rDH;
      }else{
        pos = j1 + k1 * ny + i * ny * nz;
        DzVx[n] = L(w_Vx, pos, ny, FlagZ)*rDH;
        DzVy[n] = L(w_Vy, pos, ny, FlagZ)*rDH;
        DzVz[n] = L(w_Vz, pos, ny, FlagZ)*rDH;
        //pos = (i*ny*nz + j1*nz + k1)*WSIZE;
        //segment = WSIZE;
        //DzVx[n] = L(W.W, (pos + 0), segment, FlagZ)*rDH;
        //DzVy[n] = L(W.W, (pos + 1), segment, FlagZ)*rDH;
        //DzVz[n] = L(W.W, (pos + 2), segment, FlagZ)*rDH;
        //DzVx[n] = Lz(W->Vx,i,j,k,FlagZ);
        //DzVy[n] = Lz(W->Vy,i,j,k,FlagZ);
        //DzVz[n] = Lz(W->Vz,i,j,k,FlagZ);
      }

      pos = j1 + k1 * ny + i * ny * nz;
      lam = LAM[pos]; mu = MIU[pos];
      lam2mu  = lam + 2.0f*mu;
      xix = XIX[pos]; xiy = XIY[pos]; xiz = XIZ[pos];
      etx = ETX[pos]; ety = ETY[pos]; etz = ETZ[pos];
      ztx = ZTX[pos]; zty = ZTY[pos]; ztz = ZTZ[pos];
      //pos_m = (i*ny*nz + j1*nz + k1)*MSIZE;
      //pos   = (i*ny*nz + j1*nz + k1)*WSIZE;
      //lam     = M[pos_m + 10];
      //mu      = M[pos_m + 11];
      //lam2mu  = lam + 2.0f*mu;
      //xix = M[pos_m + 0]; xiy = M[pos_m + 1]; xiz = M[pos_m + 2];
      //etx = M[pos_m + 3]; ety = M[pos_m + 4]; etz = M[pos_m + 5];
      //ztx = M[pos_m + 6]; zty = M[pos_m + 7]; ztz = M[pos_m + 8];
      //xix = M->xi_x[i][j][k]; xiy = M->xi_y[i][j][k]; xiz = M->xi_z[i][j][k];
      //etx = M->et_x[i][j][k]; ety = M->et_y[i][j][k]; etz = M->et_z[i][j][k];
      //ztx = M->zt_x[i][j][k]; zty = M->zt_y[i][j][k]; ztz = M->zt_z[i][j][k];

      w_hTxx[pos] = (
          lam2mu*xix*DxVx[n] + lam*xiy*DxVy[n] + lam*xiz*DxVz[n] +
          lam2mu*etx*DyVx[n] + lam*ety*DyVy[n] + lam*etz*DyVz[n] +
          lam2mu*ztx*DzVx[n] + lam*zty*DzVy[n] + lam*ztz*DzVz[n] );

      w_hTyy[pos] = (
          lam*xix*DxVx[n] + lam2mu*xiy*DxVy[n] + lam*xiz*DxVz[n] +
          lam*etx*DyVx[n] + lam2mu*ety*DyVy[n] + lam*etz*DyVz[n] +
          lam*ztx*DzVx[n] + lam2mu*zty*DzVy[n] + lam*ztz*DzVz[n] );

      w_hTzz[pos] = (
          lam*xix*DxVx[n] + lam*xiy*DxVy[n] + lam2mu*xiz*DxVz[n] +
          lam*etx*DyVx[n] + lam*ety*DyVy[n] + lam2mu*etz*DyVz[n] +
          lam*ztx*DzVx[n] + lam*zty*DzVy[n] + lam2mu*ztz*DzVz[n] );

      w_hTxy[pos] = (
          xiy*DxVx[n] + xix*DxVy[n] +
          ety*DyVx[n] + etx*DyVy[n] +
          zty*DzVx[n] + ztx*DzVy[n] ) * mu;

      w_hTxz[pos] = (
          xiz*DxVx[n] + xix*DxVz[n] +
          etz*DyVx[n] + etx*DyVz[n] +
          ztz*DzVx[n] + ztx*DzVz[n] ) * mu;

      w_hTyz[pos] = (
          xiz*DxVy[n] + xiy*DxVz[n] +
          etz*DyVy[n] + ety*DyVz[n] +
          ztz*DzVy[n] + zty*DzVz[n] ) * mu;

      //W.hW[pos + 3] = (
      //                   lam2mu*xix*DxVx[n] + lam*xiy*DxVy[n] + lam*xiz*DxVz[n]
      //                  +lam2mu*etx*DyVx[n] + lam*ety*DyVy[n] + lam*etz*DyVz[n]
      //                  +lam2mu*ztx*DzVx[n] + lam*zty*DzVy[n] + lam*ztz*DzVz[n]
      //                   ); // Txx
      //W.hW[pos + 4] = (
      //                   lam*xix*DxVx[n] + lam2mu*xiy*DxVy[n] + lam*xiz*DxVz[n]
      //                  +lam*etx*DyVx[n] + lam2mu*ety*DyVy[n] + lam*etz*DyVz[n]
      //                  +lam*ztx*DzVx[n] + lam2mu*zty*DzVy[n] + lam*ztz*DzVz[n]
      //                  ); // Tyy
      //W.hW[pos + 5] = (
      //                   lam*xix*DxVx[n] + lam*xiy*DxVy[n] + lam2mu*xiz*DxVz[n]
      //                  +lam*etx*DyVx[n] + lam*ety*DyVy[n] + lam2mu*etz*DyVz[n]
      //                  +lam*ztx*DzVx[n] + lam*zty*DzVy[n] + lam2mu*ztz*DzVz[n]
      //                  ); // Tzz

      //W.hW[pos + 6] = mu*(
      //                       xiy*DxVx[n] + xix*DxVy[n]
      //                      +ety*DyVx[n] + etx*DyVy[n]
      //                      +zty*DzVx[n] + ztx*DzVy[n]
      //                       ); // Txy
      //W.hW[pos + 7] = mu*(
      //                       xiz*DxVx[n] + xix*DxVz[n]
      //                      +etz*DyVx[n] + etx*DyVz[n]
      //                      +ztz*DzVx[n] + ztx*DzVz[n]
      //                       ); // Txz
      //W.hW[pos + 8] = mu*(
      //                       xiz*DxVy[n] + xiy*DxVz[n]
      //                      +etz*DyVy[n] + ety*DyVz[n]
      //                      +ztz*DzVy[n] + zty*DzVz[n]
      //                       ); // Tyz
    } // end loop of n   }}}

#ifdef DEBUG
    //idx = (j1*nz + k1)*3*3;
    idx = (j1 + k1 * ny)*3*3;
    if(j==100 && k==100){
      printf("matPlus2Min1 = %e %e %e %e %e %e %e %e %e\n",
          F.matPlus2Min1[idx + 0],
          F.matPlus2Min1[idx + 1],
          F.matPlus2Min1[idx + 2],
          F.matPlus2Min1[idx + 3],
          F.matPlus2Min1[idx + 4],
          F.matPlus2Min1[idx + 5],
          F.matPlus2Min1[idx + 6],
          F.matPlus2Min1[idx + 7],
          F.matPlus2Min1[idx + 8]);
      printf("matPlus2Min2 = %e %e %e %e %e %e %e %e %e\n",
          F.matPlus2Min2[idx + 0],
          F.matPlus2Min2[idx + 1],
          F.matPlus2Min2[idx + 2],
          F.matPlus2Min2[idx + 3],
          F.matPlus2Min2[idx + 4],
          F.matPlus2Min2[idx + 5],
          F.matPlus2Min2[idx + 6],
          F.matPlus2Min2[idx + 7],
          F.matPlus2Min2[idx + 8]);
      printf("matPlus2Min3 = %e %e %e %e %e %e %e %e %e\n",
          F.matPlus2Min3[idx + 0],
          F.matPlus2Min3[idx + 1],
          F.matPlus2Min3[idx + 2],
          F.matPlus2Min3[idx + 3],
          F.matPlus2Min3[idx + 4],
          F.matPlus2Min3[idx + 5],
          F.matPlus2Min3[idx + 6],
          F.matPlus2Min3[idx + 7],
          F.matPlus2Min3[idx + 8]);
      printf("matPlus2Min4 = %e %e %e %e %e %e %e %e %e\n",
          F.matPlus2Min4[idx + 0],
          F.matPlus2Min4[idx + 1],
          F.matPlus2Min4[idx + 2],
          F.matPlus2Min4[idx + 3],
          F.matPlus2Min4[idx + 4],
          F.matPlus2Min4[idx + 5],
          F.matPlus2Min4[idx + 6],
          F.matPlus2Min4[idx + 7],
          F.matPlus2Min4[idx + 8]);
      printf("matPlus2Min5 = %e %e %e %e %e %e %e %e %e\n",
          F.matPlus2Min5[idx + 0],
          F.matPlus2Min5[idx + 1],
          F.matPlus2Min5[idx + 2],
          F.matPlus2Min5[idx + 3],
          F.matPlus2Min5[idx + 4],
          F.matPlus2Min5[idx + 5],
          F.matPlus2Min5[idx + 6],
          F.matPlus2Min5[idx + 7],
          F.matPlus2Min5[idx + 8]);
      printf("matMin2Plus1 = %e %e %e %e %e %e %e %e %e\n",
          F.matMin2Plus1[idx + 0],
          F.matMin2Plus1[idx + 1],
          F.matMin2Plus1[idx + 2],
          F.matMin2Plus1[idx + 3],
          F.matMin2Plus1[idx + 4],
          F.matMin2Plus1[idx + 5],
          F.matMin2Plus1[idx + 6],
          F.matMin2Plus1[idx + 7],
          F.matMin2Plus1[idx + 8]);
      printf("matMin2Plus2 = %e %e %e %e %e %e %e %e %e\n",
          F.matMin2Plus2[idx + 0],
          F.matMin2Plus2[idx + 1],
          F.matMin2Plus2[idx + 2],
          F.matMin2Plus2[idx + 3],
          F.matMin2Plus2[idx + 4],
          F.matMin2Plus2[idx + 5],
          F.matMin2Plus2[idx + 6],
          F.matMin2Plus2[idx + 7],
          F.matMin2Plus2[idx + 8]);
      printf("matMin2Plus3 = %e %e %e %e %e %e %e %e %e\n",
          F.matMin2Plus3[idx + 0],
          F.matMin2Plus3[idx + 1],
          F.matMin2Plus3[idx + 2],
          F.matMin2Plus3[idx + 3],
          F.matMin2Plus3[idx + 4],
          F.matMin2Plus3[idx + 5],
          F.matMin2Plus3[idx + 6],
          F.matMin2Plus3[idx + 7],
          F.matMin2Plus3[idx + 8]);
      printf("matMin2Plus4 = %e %e %e %e %e %e %e %e %e\n",
          F.matMin2Plus4[idx + 0],
          F.matMin2Plus4[idx + 1],
          F.matMin2Plus4[idx + 2],
          F.matMin2Plus4[idx + 3],
          F.matMin2Plus4[idx + 4],
          F.matMin2Plus4[idx + 5],
          F.matMin2Plus4[idx + 6],
          F.matMin2Plus4[idx + 7],
          F.matMin2Plus4[idx + 8]);
      printf("matMin2Plus5 = %e %e %e %e %e %e %e %e %e\n",
          F.matMin2Plus5[idx + 0],
          F.matMin2Plus5[idx + 1],
          F.matMin2Plus5[idx + 2],
          F.matMin2Plus5[idx + 3],
          F.matMin2Plus5[idx + 4],
          F.matMin2Plus5[idx + 5],
          F.matMin2Plus5[idx + 6],
          F.matMin2Plus5[idx + 7],
          F.matMin2Plus5[idx + 8]);
      printf("D21_1 = %e %e %e %e %e %e %e %e %e\n",
          F.D21_1[idx + 0],
          F.D21_1[idx + 1],
          F.D21_1[idx + 2],
          F.D21_1[idx + 3],
          F.D21_1[idx + 4],
          F.D21_1[idx + 5],
          F.D21_1[idx + 6],
          F.D21_1[idx + 7],
          F.D21_1[idx + 8]);
      printf("D22_1 = %e %e %e %e %e %e %e %e %e\n",
          F.D22_1[idx + 0],
          F.D22_1[idx + 1],
          F.D22_1[idx + 2],
          F.D22_1[idx + 3],
          F.D22_1[idx + 4],
          F.D22_1[idx + 5],
          F.D22_1[idx + 6],
          F.D22_1[idx + 7],
          F.D22_1[idx + 8]);
      printf("D23_1 = %e %e %e %e %e %e %e %e %e\n",
          F.D23_1[idx + 0],
          F.D23_1[idx + 1],
          F.D23_1[idx + 2],
          F.D23_1[idx + 3],
          F.D23_1[idx + 4],
          F.D23_1[idx + 5],
          F.D23_1[idx + 6],
          F.D23_1[idx + 7],
          F.D23_1[idx + 8]);
      printf("D31_1 = %e %e %e %e %e %e %e %e %e\n",
          F.D31_1[idx + 0],
          F.D31_1[idx + 1],
          F.D31_1[idx + 2],
          F.D31_1[idx + 3],
          F.D31_1[idx + 4],
          F.D31_1[idx + 5],
          F.D31_1[idx + 6],
          F.D31_1[idx + 7],
          F.D31_1[idx + 8]);
      printf("D32_1 = %e %e %e %e %e %e %e %e %e\n",
          F.D32_1[idx + 0],
          F.D32_1[idx + 1],
          F.D32_1[idx + 2],
          F.D32_1[idx + 3],
          F.D32_1[idx + 4],
          F.D32_1[idx + 5],
          F.D32_1[idx + 6],
          F.D32_1[idx + 7],
          F.D32_1[idx + 8]);
      printf("D33_1 = %e %e %e %e %e %e %e %e %e\n",
          F.D33_1[idx + 0],
          F.D33_1[idx + 1],
          F.D33_1[idx + 2],
          F.D33_1[idx + 3],
          F.D33_1[idx + 4],
          F.D33_1[idx + 5],
          F.D33_1[idx + 6],
          F.D33_1[idx + 7],
          F.D33_1[idx + 8]);
      printf("D21_2 = %e %e %e %e %e %e %e %e %e\n",
          F.D21_2[idx + 0],
          F.D21_2[idx + 1],
          F.D21_2[idx + 2],
          F.D21_2[idx + 3],
          F.D21_2[idx + 4],
          F.D21_2[idx + 5],
          F.D21_2[idx + 6],
          F.D21_2[idx + 7],
          F.D21_2[idx + 8]);
      printf("D22_2 = %e %e %e %e %e %e %e %e %e\n",
          F.D22_2[idx + 0],
          F.D22_2[idx + 1],
          F.D22_2[idx + 2],
          F.D22_2[idx + 3],
          F.D22_2[idx + 4],
          F.D22_2[idx + 5],
          F.D22_2[idx + 6],
          F.D22_2[idx + 7],
          F.D22_2[idx + 8]);
      printf("D23_2 = %e %e %e %e %e %e %e %e %e\n",
          F.D23_2[idx + 0],
          F.D23_2[idx + 1],
          F.D23_2[idx + 2],
          F.D23_2[idx + 3],
          F.D23_2[idx + 4],
          F.D23_2[idx + 5],
          F.D23_2[idx + 6],
          F.D23_2[idx + 7],
          F.D23_2[idx + 8]);
      printf("D31_2 = %e %e %e %e %e %e %e %e %e\n",
          F.D31_2[idx + 0],
          F.D31_2[idx + 1],
          F.D31_2[idx + 2],
          F.D31_2[idx + 3],
          F.D31_2[idx + 4],
          F.D31_2[idx + 5],
          F.D31_2[idx + 6],
          F.D31_2[idx + 7],
          F.D31_2[idx + 8]);
      printf("D32_2 = %e %e %e %e %e %e %e %e %e\n",
          F.D32_2[idx + 0],
          F.D32_2[idx + 1],
          F.D32_2[idx + 2],
          F.D32_2[idx + 3],
          F.D32_2[idx + 4],
          F.D32_2[idx + 5],
          F.D32_2[idx + 6],
          F.D32_2[idx + 7],
          F.D32_2[idx + 8]);
      printf("D33_2 = %e %e %e %e %e %e %e %e %e\n",
          F.D33_2[idx + 0],
          F.D33_2[idx + 1],
          F.D33_2[idx + 2],
          F.D33_2[idx + 3],
          F.D33_2[idx + 4],
          F.D33_2[idx + 5],
          F.D33_2[idx + 6],
          F.D33_2[idx + 7],
          F.D33_2[idx + 8]);
    }
#endif

   // end ---------------------------------------//
  } // end j k
  return;
}

void fault_deriv(Wave W, Fault F, realptr_t M,
    int FlagX, int FlagY, int FlagZ)
{
  dim3 block(16, 8, 1);
  dim3 grid(
      (hostParams.nj + block.x - 1) / block.x,
      (hostParams.nk + block.y - 1) / block.y,
      1);
  fault_deriv_cu <<<grid, block>>> (W, F, M, FlagX, FlagY, FlagZ);
  return;
}
