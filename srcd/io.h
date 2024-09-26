#ifndef IO_H
#define IO_H

#ifdef useNetCDF

#define MAX_NUM_NC_VARID 100
#define MAX_NUM_NC_DIMID 4

typedef struct {
  int ncid;
  int err;
  int varid[MAX_NUM_NC_VARID];
  int dimid[MAX_NUM_NC_DIMID];
} ncFile;

// For Fault output
extern void nc_def_fault(Fault F, ncFile *);
extern void nc_put_fault(Fault F, int it, ncFile);
extern void nc_put_fault_coord(real_t *C, ncFile nc);
extern void nc_end_fault(ncFile);

// For wave slice output

// slice xy, slice index range from 0 to NZ-1
extern void nc_def_wave_xy(int slice_index, ncFile *);
extern void nc_put_wave_xy(real_t *, int it, int slice_index, ncFile);
extern void nc_put_wave_xy_coord(real_t *C, int slice_index, ncFile);
extern void nc_end_wave_xy(int slice_index, ncFile);

// slice xz, slice index range from 0 to NY-1
extern void nc_def_wave_xz(int slice_index, ncFile *);
extern void nc_put_wave_xz(real_t *, int it, int slice_index, ncFile);
extern void nc_put_wave_xz_coord(real_t *C, int slice_index, ncFile);
extern void nc_end_wave_xz(int slice_index, ncFile);

// slice yz, slice index range from 0 to NZ-1
extern void nc_def_wave_yz(int slice_index, ncFile *);
extern void nc_put_wave_yz(real_t * C, int it, int slice_index, ncFile);
extern void nc_put_wave_yz_coord(real_t * C, int slice_index, ncFile);
extern void nc_end_wave_yz(int slice_index, ncFile);

extern void nc_def_recv(Recv R, Wave W, ncFile *);
extern void nc_end_recv(Recv R, ncFile);

extern void locate_recv(Recv *, realptr_t C);
#endif

#ifdef useHDF5
/* use hdf5 c++ api */
//#include <string.h>
//#include "H5Cpp.h"
//
//using std::cout;
//using std::endl;
//
//using namespace H5;
//#include "h5_io.c"
//#ifdef __cplusplus
//extern "C"{
//#endif
extern void h5_fault_io_init();
extern void h5_fault_save(Fault F, const int it);
extern void h5_fault_io_end();
//#ifdef __cplusplus
//}
//#endif
#endif
#endif
