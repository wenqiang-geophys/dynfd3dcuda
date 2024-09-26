#ifdef useNetCDF
#include <stdio.h>
#include <stdlib.h>
#include "params.h"
#include "common.h"
#include "io.h"

#ifdef DoublePrecision
#define nc_get_vara_real_t nc_get_vara_double
#define nc_put_vara_real_t nc_put_vara_double
#else
#define nc_get_vara_real_t nc_get_vara_float
#define nc_put_vara_real_t nc_put_vara_float
#endif

void nc_def_fault(Fault F, ncFile *nc)
{
  int err;

  int nj = hostParams.nj;
  int nk = hostParams.nk;

  char filename[1000];
  sprintf(filename, "%s/fault_mpi%02d%02d%02d.nc", OUT,
      thisid[0], thisid[1], thisid[2]);

  err = nc_create(filename, NC_CLOBBER, &(nc->ncid)); handle_err(err);

  // define dimensions
  err = nc_def_dim(nc->ncid, "nt", NC_UNLIMITED, &(nc->dimid[0]));
  err = nc_def_dim(nc->ncid, "nz", nk,           &(nc->dimid[1]));
  err = nc_def_dim(nc->ncid, "ny", nj,           &(nc->dimid[2]));
  handle_err(err);

  const int dimid2[2] = {nc->dimid[1], nc->dimid[2]};

  err = nc_def_var(nc->ncid, "x", NC_REAL_T, 2, dimid2, &(nc->varid[20]));
  err = nc_def_var(nc->ncid, "y", NC_REAL_T, 2, dimid2, &(nc->varid[21]));
  err = nc_def_var(nc->ncid, "z", NC_REAL_T, 2, dimid2, &(nc->varid[22]));

  err = nc_def_var(nc->ncid, "init_t0",  NC_REAL_T, 2, dimid2, &(nc->varid[10]));
  err = nc_def_var(nc->ncid, "C0",       NC_REAL_T, 2, dimid2, &(nc->varid[11]));
  err = nc_def_var(nc->ncid, "str_peak", NC_REAL_T, 2, dimid2, &(nc->varid[12]));

  err = nc_def_var(nc->ncid, "united",      NC_INT, 2, dimid2, &(nc->varid[13]));
  //err = nc_def_var(nc->ncid, "rup_index_y", NC_INT, 2, dimid2, &(nc->varid[14]));
  //err = nc_def_var(nc->ncid, "rup_index_z", NC_INT, 2, dimid2, &(nc->varid[15]));

  err = nc_def_var(nc->ncid, "a", NC_REAL_T, 2, dimid2, &(nc->varid[16]));
  err = nc_def_var(nc->ncid, "b", NC_REAL_T, 2, dimid2, &(nc->varid[17]));
  err = nc_def_var(nc->ncid, "Vw", NC_REAL_T, 2, dimid2, &(nc->varid[18]));

  handle_err(err);

  // define variables
  err = nc_def_var(nc->ncid, "Vs1"  , NC_REAL_T, 3, nc->dimid, &(nc->varid[0]));
  err = nc_def_var(nc->ncid, "Vs2"  , NC_REAL_T, 3, nc->dimid, &(nc->varid[1]));
  err = nc_def_var(nc->ncid, "tn"   , NC_REAL_T, 3, nc->dimid, &(nc->varid[2]));
  err = nc_def_var(nc->ncid, "ts1"  , NC_REAL_T, 3, nc->dimid, &(nc->varid[3]));
  err = nc_def_var(nc->ncid, "ts2"  , NC_REAL_T, 3, nc->dimid, &(nc->varid[4]));
  err = nc_def_var(nc->ncid, "Us0"  , NC_REAL_T, 3, nc->dimid, &(nc->varid[5]));
  err = nc_def_var(nc->ncid, "State", NC_REAL_T, 3, nc->dimid, &(nc->varid[6]));
  err = nc_def_var(nc->ncid, "rup_index_y", NC_INT, 3, nc->dimid, &(nc->varid[14]));
  err = nc_def_var(nc->ncid, "rup_index_z", NC_INT, 3, nc->dimid, &(nc->varid[15]));
  err = nc_def_var(nc->ncid, "rup_sensor", NC_REAL_T, 3, nc->dimid, &(nc->varid[30]));
  err = nc_def_var(nc->ncid, "TP_T", NC_REAL_T, 3, nc->dimid, &(nc->varid[31]));
  err = nc_def_var(nc->ncid, "TP_P", NC_REAL_T, 3, nc->dimid, &(nc->varid[32]));

  handle_err(err);

  err = nc_enddef(nc->ncid); handle_err(err);
  // end define ===============================================================

  // put some initial values
  real_t *data = (real_t *) malloc(sizeof(real_t)*nj*nk);
  int *dataint = (int *) malloc(sizeof(int)*nj*nk);

#ifdef DoublePrecision
  cudaMemcpy(data, F.C0, sizeof(double)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_var_double(nc->ncid, nc->varid[11], data);handle_err(err);

  cudaMemcpy(data, F.str_peak, sizeof(double)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_var_double(nc->ncid, nc->varid[12], data);handle_err(err);

  cudaMemcpy(dataint, F.united, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_var_int(nc->ncid, nc->varid[13], dataint);handle_err(err);

  //cudaMemcpy(dataint, F.rup_index_y, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  //err = nc_put_var_int(nc->ncid, nc->varid[14], dataint);handle_err(err);

  //cudaMemcpy(dataint, F.rup_index_z, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  //err = nc_put_var_int(nc->ncid, nc->varid[15], dataint);handle_err(err);

  cudaMemcpy(data, F.a, sizeof(double)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_var_double(nc->ncid, nc->varid[16], data);handle_err(err);

  cudaMemcpy(data, F.b, sizeof(double)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_var_double(nc->ncid, nc->varid[17], data);handle_err(err);

  cudaMemcpy(data, F.Vw, sizeof(double)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_var_double(nc->ncid, nc->varid[18], data);handle_err(err);

#else

  cudaMemcpy(data, F.C0, sizeof(float)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_var_float(nc->ncid, nc->varid[11], data);handle_err(err);

  cudaMemcpy(data, F.str_peak, sizeof(float)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_var_float(nc->ncid, nc->varid[12], data);handle_err(err);

  cudaMemcpy(dataint, F.united, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_var_int(nc->ncid, nc->varid[13], dataint);handle_err(err);

  //cudaMemcpy(dataint, F.rup_index_y, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  //err = nc_put_var_int(nc->ncid, nc->varid[14], dataint);handle_err(err);

  //cudaMemcpy(dataint, F.rup_index_z, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  //err = nc_put_var_int(nc->ncid, nc->varid[15], dataint);handle_err(err);

  cudaMemcpy(data, F.a, sizeof(float)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_var_float(nc->ncid, nc->varid[16], data);handle_err(err);

  cudaMemcpy(data, F.b, sizeof(float)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_var_float(nc->ncid, nc->varid[17], data);handle_err(err);

  cudaMemcpy(data, F.Vw, sizeof(float)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_var_float(nc->ncid, nc->varid[18], data);handle_err(err);
#endif

  free(data);
  free(dataint);
  return;
}

void nc_put_fault_coord(real_t *C, ncFile nc)
{
  if(!hostParams.faultnode) return;

  int ni = hostParams.ni;
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  int nx = ni + 6;
  int ny = nj + 6;
  int nz = nk + 6;

  real_t *x = (real_t *) malloc(sizeof(real_t)*nj*nk);
  real_t *y = (real_t *) malloc(sizeof(real_t)*nj*nk);
  real_t *z = (real_t *) malloc(sizeof(real_t)*nj*nk);

  int srci = hostParams.NX/2;

  int i = srci % ni; // local i

  for (int j = 0; j < nj; j++){
    for (int k = 0; k < nk; k++){
      int j1 = j + 3;
      int k1 = k + 3;
      int i1 = i + 3;
      int pos = j1 + k1 * ny + i1 * ny * nz;
      int nxyz = nx*ny*nz;
      x[j + k * nj] = C[pos + 0 * nxyz];
      y[j + k * nj] = C[pos + 1 * nxyz];
      z[j + k * nj] = C[pos + 2 * nxyz];
    }
  }

  int err;
#ifdef DoublePrecision
  err = nc_put_var_double(nc.ncid, nc.varid[20], x);
  err = nc_put_var_double(nc.ncid, nc.varid[21], y);
  err = nc_put_var_double(nc.ncid, nc.varid[22], z);
#else
  err = nc_put_var_float(nc.ncid, nc.varid[20], x);
  err = nc_put_var_float(nc.ncid, nc.varid[21], y);
  err = nc_put_var_float(nc.ncid, nc.varid[22], z);
#endif
  handle_err(err);

  free(x);
  free(y);
  free(z);

  return;
}

void nc_put_fault(Fault F, int it, ncFile nc)
{
  int nj = hostParams.nj;
  int nk = hostParams.nk;

  size_t start[3] = {it, 0, 0};
  size_t count[3] = {1, nk, nj};
  int err;

  size_t ibytes = sizeof(real_t)*nj*nk;
  real_t *data = (real_t *) malloc(ibytes);

#ifdef DoublePrecision
  cudaMemcpy(data, F.Vs1, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_double(nc.ncid, nc.varid[0], start, count, data);

  cudaMemcpy(data, F.Vs2, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_double(nc.ncid, nc.varid[1], start, count, data);

  cudaMemcpy(data, F.Tn,  ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_double(nc.ncid, nc.varid[2], start, count, data);

  cudaMemcpy(data, F.Ts1, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_double(nc.ncid, nc.varid[3], start, count, data);

  cudaMemcpy(data, F.Ts2, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_double(nc.ncid, nc.varid[4], start, count, data);

  cudaMemcpy(data, F.slip, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_double(nc.ncid, nc.varid[5], start, count, data);

  cudaMemcpy(data, F.State, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_double(nc.ncid, nc.varid[6], start, count, data);

  cudaMemcpy(data, F.init_t0, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_var_double(nc.ncid, nc.varid[10], data);
  handle_err(err);
#else
  cudaMemcpy(data, F.Vs1, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_float(nc.ncid, nc.varid[0], start, count, data);

  cudaMemcpy(data, F.Vs2, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_float(nc.ncid, nc.varid[1], start, count, data);

  cudaMemcpy(data, F.Tn,  ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_float(nc.ncid, nc.varid[2], start, count, data);

  cudaMemcpy(data, F.Ts1, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_float(nc.ncid, nc.varid[3], start, count, data);

  cudaMemcpy(data, F.Ts2, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_float(nc.ncid, nc.varid[4], start, count, data);

  cudaMemcpy(data, F.slip, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_float(nc.ncid, nc.varid[5], start, count, data);

  cudaMemcpy(data, F.State, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_vara_float(nc.ncid, nc.varid[6], start, count, data);

  cudaMemcpy(data, F.init_t0, ibytes, cudaMemcpyDeviceToHost);
  err = nc_put_var_float(nc.ncid, nc.varid[10], data);
  handle_err(err);

#endif

  int *dataint = (int *) malloc(sizeof(int)*nj*nk);

  cudaMemcpy(dataint, F.rup_index_y, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_vara_int(nc.ncid, nc.varid[14], start, count, dataint);

  cudaMemcpy(dataint, F.rup_index_z, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_vara_int(nc.ncid, nc.varid[15], start, count, dataint);

  cudaMemcpy(data, F.rup_sensor, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_vara_real_t(nc.ncid, nc.varid[30], start, count, data);

  cudaMemcpy(data, F.TP_T, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_vara_real_t(nc.ncid, nc.varid[31], start, count, data);
  cudaMemcpy(data, F.TP_P, sizeof(int)*nj*nk, cudaMemcpyDeviceToHost);
  err = nc_put_vara_real_t(nc.ncid, nc.varid[32], start, count, data);

  handle_err(err);

  nc_sync(nc.ncid);
  free(data); data = NULL;
  free(dataint);

  return;
}

void nc_end_fault(ncFile nc)
{
  int err;
  nc_sync(nc.ncid);
  err = nc_close(nc.ncid);
  handle_err(err);
  return;
}
#endif
