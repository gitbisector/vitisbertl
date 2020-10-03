#define VDATA_SIZE 16

typedef signed short Dt;
typedef ap_int<32> It;
typedef struct v_datatype { Dt data[VDATA_SIZE]; } v_dt;

enum {
	mems = 16,
	cores = 64,
	Tp2 = 10,
	Tsize = 1<<Tp2,             /* 1024 */
	Msize = (Tsize/cores),      /* 16 */
	Vsize = Msize/VDATA_SIZE,   /* 1 */
	Vshift = 2,
	Veclen = 14,
};

typedef struct v_inWrrtype{
	Dt data[Msize];
} v_arr;
